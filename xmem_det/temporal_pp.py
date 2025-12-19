import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.models.detectors.pointpillar import PointPillar
from xmem_det.xmem_wrapper import XMemBackboneWrapper
from xmem_det.util import boxes_to_bev_masks


class TemporalPointPillar(PointPillar):
    def __init__(self, model_cfg, num_class, dataset, xmem_train_cfg, pc_range):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c_bev = self.backbone_2d.num_bev_features

        self.xmem = XMemBackboneWrapper(
            device=str(device),
            train_config=xmem_train_cfg,
            bev_channels=c_bev,
        )

        self.pc_range = pc_range
        self.temporal_fusion = nn.Conv2d(c_bev + self.xmem.hidden_dim, c_bev, kernel_size=1)
        self.motion_mask_tf = nn.Conv2d(1 + 6, 1, kernel_size=3, padding=1)
        self.aux_weight = float(xmem_train_cfg.get("aux_weight", 0.1))

    def reset_sequence(self, seq_id: int):
        self.xmem.clear_memory()

    def _build_scene_mask_from_bev(self, spatial_features_2d: torch.Tensor):
        with torch.no_grad():
            mag = spatial_features_2d.abs().sum(dim=1, keepdim=True)
            mask = (mag > 0).float()
        return mask

    def _motion6_map(self, T_rel: torch.Tensor, H: int, W: int, device, dtype):
        if T_rel is None:
            return torch.zeros(1, 6, H, W, device=device, dtype=dtype)

        T = T_rel.unsqueeze(0) if T_rel.dim() == 2 else T_rel

        r11 = T[:, 0, 0]
        r12 = T[:, 0, 1]
        r21 = T[:, 1, 0]
        r22 = T[:, 1, 1]
        tx = T[:, 0, 3]
        ty = T[:, 1, 3]

        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        sx = float(x_max - x_min)
        sy = float(y_max - y_min)
        tx = tx / sx
        ty = ty / sy

        v = torch.stack([r11, r12, r21, r22, tx, ty], dim=1).to(dtype)
        return v.view(v.shape[0], 6, 1, 1).expand(v.shape[0], 6, H, W)

    def _compensate_prev_scene_mask_logits(self, scene_mask_prev: torch.Tensor, T_rel: torch.Tensor):
        B, _, H, W = scene_mask_prev.shape
        motion = self._motion6_map(T_rel, H, W, scene_mask_prev.device, scene_mask_prev.dtype)
        inp = torch.cat([scene_mask_prev, motion], dim=1)
        logits = self.motion_mask_tf(inp)
        return logits

    def _gt_occ_mask(self, batch_dict, bev_h: int, bev_w: int):
        if "gt_boxes" not in batch_dict:
            return None
        gt = batch_dict["gt_boxes"]
        if not isinstance(gt, torch.Tensor):
            return None
        if gt.numel() == 0:
            return None

        if gt.shape[-1] < 7:
            return None

        gt7 = gt[..., :7]
        valid = (gt7.abs().sum(dim=-1) > 0).to(gt7.dtype)

        B, M, _ = gt7.shape
        scores = valid

        inst_masks = boxes_to_bev_masks(
            gt7,
            scores,
            bev_h,
            bev_w,
            self.pc_range,
            score_thresh=0.0,
        )

        occ = (inst_masks.sum(dim=1, keepdim=True) > 0).to(gt7.dtype)
        return occ


    def _build_det_masks(self, pred_dicts, batch_dict):
        batch_bev = batch_dict["spatial_features_2d"]
        bev_h, bev_w = batch_bev.shape[-2], batch_bev.shape[-1]
        B = len(pred_dicts)
        max_det = max([d["pred_boxes"].shape[0] for d in pred_dicts]) if B > 0 else 0

        if max_det == 0:
            return torch.zeros(B, 1, bev_h, bev_w, device=batch_bev.device)

        boxes_batch = []
        scores_batch = []
        for d in pred_dicts:
            boxes = d["pred_boxes"]
            scores = d["pred_scores"]
            if boxes.shape[0] < max_det:
                pad_n = max_det - boxes.shape[0]
                boxes = torch.cat([boxes, torch.zeros(pad_n, boxes.shape[1], device=boxes.device)], dim=0)
                scores = torch.cat([scores, torch.zeros(pad_n, device=scores.device)], dim=0)
            boxes_batch.append(boxes.unsqueeze(0))
            scores_batch.append(scores.unsqueeze(0))

        boxes_batch = torch.cat(boxes_batch, dim=0)
        scores_batch = torch.cat(scores_batch, dim=0)

        det_masks_next = boxes_to_bev_masks(
            boxes_batch,
            scores_batch,
            bev_h,
            bev_w,
            self.pc_range,
            score_thresh=0.3,
        )
        return det_masks_next

    def forward(self, batch_dict, t_seq: int = 0, det_instance_masks_prev: torch.Tensor = None, T_rel: torch.Tensor = None, alpha_temporal: float = 1.0):
        aux_loss = None

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

            if cur_module is self.backbone_2d:
                bev = batch_dict["spatial_features_2d"]
                bev_h, bev_w = bev.shape[-2], bev.shape[-1]

                if det_instance_masks_prev is not None:
                    scene_mask_prev = (det_instance_masks_prev.sum(dim=1, keepdim=True) > 0).float()
                    scene_logits = self._compensate_prev_scene_mask_logits(scene_mask_prev, T_rel)
                    scene_mask = torch.sigmoid(scene_logits)

                    if self.training:
                        gt_occ = self._gt_occ_mask(batch_dict, bev_h, bev_w)
                        if gt_occ is not None:
                            aux_loss = F.binary_cross_entropy_with_logits(scene_logits, gt_occ)
                else:
                    scene_mask = self._build_scene_mask_from_bev(bev)

                occ_logits, hidden_features = self.xmem.forward_step(
                    t_seq,
                    bev,
                    scene_mask=scene_mask,
                )

                gate = torch.sigmoid(occ_logits)
                a = float(alpha_temporal)
                gate_used = gate * a + (1.0 - a)
                hidden_used = hidden_features * a

                bev_gated = bev * gate_used
                bev_with_temporal = torch.cat([bev_gated, hidden_used], dim=1)
                bev_fused = self.temporal_fusion(bev_with_temporal)
                batch_dict["spatial_features_2d"] = bev_fused

        if self.training:
            loss_det, tb_dict, disp_dict = self.get_training_loss()
            if aux_loss is not None:
                loss = loss_det + self.aux_weight * aux_loss
                tb_dict["aux_motion_tf"] = aux_loss.detach()
            else:
                loss = loss_det

            ret_dict = {"loss": loss}

            fwd_ret = self.dense_head.forward_ret_dict
            batch_cls_preds, batch_box_preds = self.dense_head.generate_predicted_boxes(
                batch_size=batch_dict["batch_size"],
                cls_preds=fwd_ret["cls_preds"],
                box_preds=fwd_ret["box_preds"],
                dir_cls_preds=fwd_ret.get("dir_cls_preds", None),
            )

            batch_dict["batch_cls_preds"] = batch_cls_preds
            batch_dict["batch_box_preds"] = batch_box_preds
            batch_dict["cls_preds_normalized"] = False

            if isinstance(batch_cls_preds, list):
                batch_dict["multihead_label_mapping"] = [
                    self.dense_head.rpn_heads[i].head_label_indices for i in range(len(batch_cls_preds))
                ]

            pred_dicts, _ = self.post_processing(batch_dict)
            det_masks_next = self._build_det_masks(pred_dicts, batch_dict)
            return ret_dict, tb_dict, disp_dict, det_masks_next

        pred_dicts, recall_dicts = self.post_processing(batch_dict)
        det_masks_next = self._build_det_masks(pred_dicts, batch_dict)
        return pred_dicts, recall_dicts, det_masks_next

