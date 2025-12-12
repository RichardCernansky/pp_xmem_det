import torch
import torch.nn.functional as F
import torch.nn as nn

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


    def _build_scene_mask_from_bev(self, spatial_features_2d: torch.Tensor):
        with torch.no_grad():
            mag = spatial_features_2d.abs().sum(dim=1, keepdim=True)
            mask = (mag > 0).float()
        return mask

    def _build_det_masks(self, pred_dicts, batch_dict):
        batch_bev = batch_dict["spatial_features_2d"]
        bev_h, bev_w = batch_bev.shape[-2], batch_bev.shape[-1]
        B = len(pred_dicts)
        max_det = max([d["pred_boxes"].shape[0] for d in pred_dicts]) if B > 0 else 0

        if max_det == 0:
            det_masks_next = torch.zeros(
                B,
                1,
                bev_h,
                bev_w,
                device=batch_bev.device,
            )
            return det_masks_next

        boxes_batch = []
        scores_batch = []
        for d in pred_dicts:
            boxes = d["pred_boxes"]  # Changed from boxes_lidar
            scores = d["pred_scores"]  # Changed from scores
            if boxes.shape[0] < max_det:
                pad_n = max_det - boxes.shape[0]
                pad_boxes = torch.zeros(
                    pad_n,
                    boxes.shape[1],
                    device=boxes.device,
                )
                pad_scores = torch.zeros(
                    pad_n,
                    device=scores.device,
                )
                boxes = torch.cat([boxes, pad_boxes], dim=0)
                scores = torch.cat([scores, pad_scores], dim=0)
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

    def forward(
        self,
        batch_dict,
        t_seq: int = 0,
        det_instance_masks_prev: torch.Tensor = None,
    ):
        # Process through VFE, scatter, backbone_2d
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            
            # After backbone_2d, apply XMem gating
            if cur_module is self.backbone_2d:
                bev = batch_dict["spatial_features_2d"]

                if det_instance_masks_prev is not None:
                    scene_mask = (det_instance_masks_prev.sum(dim=1, keepdim=True) > 0).float()
                else:
                    scene_mask = self._build_scene_mask_from_bev(bev)

                # B, _, bev_h, bev_w = scene_mask.shape

                occ_logits, hidden_features = self.xmem.forward_step(
                    t_seq,
                    bev,
                    scene_mask=scene_mask,
                )

                # Step 1: Apply occupancy gate
                gate = torch.sigmoid(occ_logits)
                bev_gated = bev * gate  # (B, C_bev, H, W)
                
                # Step 2: Concatenate with temporal features
                bev_with_temporal = torch.cat([bev_gated, hidden_features], dim=1)
                # Shape: (B, C_bev + D_hidden, H, W)
                
                # Step 3: Fuse using 1x1 conv
                bev_fused = self.temporal_fusion(bev_with_temporal)  # (B, C_bev, H, W)
                
                # Update batch_dict with enhanced features
                batch_dict["spatial_features_2d"] = bev_fused
        
        # Call dense_head
        batch_dict = self.dense_head(batch_dict)

        if self.training:
            # Compute loss
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {"loss": loss}
            
            # Generate predictions from forward_ret_dict
            fwd_ret = self.dense_head.forward_ret_dict
            batch_cls_preds, batch_box_preds = self.dense_head.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'],
                cls_preds=fwd_ret['cls_preds'],
                box_preds=fwd_ret['box_preds'],
                dir_cls_preds=fwd_ret.get('dir_cls_preds', None)
            )
            
            # Add to batch_dict
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
            
            # Extract multihead_label_mapping from rpn_heads (just like OpenPCDet does)
            if isinstance(batch_cls_preds, list):
                multihead_label_mapping = []
                for idx in range(len(batch_cls_preds)):
                    multihead_label_mapping.append(self.dense_head.rpn_heads[idx].head_label_indices)
                batch_dict['multihead_label_mapping'] = multihead_label_mapping

            pred_dicts, _ = self.post_processing(batch_dict)
            det_masks_next = self._build_det_masks(pred_dicts, batch_dict)
            
            return ret_dict, tb_dict, disp_dict, det_masks_next

        # Eval mode
        pred_dicts, recall_dicts = self.post_processing(batch_dict)
        det_masks_next = self._build_det_masks(pred_dicts, batch_dict)
        return pred_dicts, recall_dicts, det_masks_next