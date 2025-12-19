import argparse
import os
import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from datasets.nuscenes_seq_dataset import NuScenesSeqDataset, collate_seq
from xmem_det.temporal_pp import TemporalPointPillar
from xmem_det.optimizer import build_stage_optimizer

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

from xmem_det.util import load_xmem_train_cfg

def set_trainable_prefixes(model, prefixes):
    for n, p in model.named_parameters():
        p.requires_grad = any(n.startswith(pref) for pref in prefixes)

def stage_plan(args):
    e1 = args.stage1_epochs
    e2a = e1 + args.stage2a_epochs
    e2b = e2a + args.stage2b_epochs
    e2c = e2b + args.stage2c_epochs
    e3 = e2c + args.stage3_epochs
    return e1, e2a, e2b, e2c, e3

def alpha_for_epoch(epoch, args):
    e1, e2a, e2b, e2c, _ = stage_plan(args)

    if epoch < e1:
        return 0.0
    if epoch < e2a:
        k = epoch - e1
        d = max(args.stage2a_epochs, 1)
        return 0.3 * (k + 1) / d
    if epoch < e2b:
        k = epoch - e2a
        d = max(args.stage2b_epochs, 1)
        return 0.3 + (1.0 - 0.3) * (k + 1) / d
    return 1.0

def stage_name(epoch, args):
    e1, e2a, e2b, e2c, e3 = stage_plan(args)
    if epoch < e1:
        return "stage1"
    if epoch < e2a:
        return "stage2a"
    if epoch < e2b:
        return "stage2b"
    if epoch < e2c:
        return "stage2c"
    if epoch < e3:
        return "stage3"
    return "stage3"


def rel_T_curr_prev(T_world_lidar: np.ndarray, t: int) -> np.ndarray:
    T_prev = T_world_lidar[t - 1]
    T_curr = T_world_lidar[t]
    return (np.linalg.inv(T_curr) @ T_prev).astype(np.float32)


def to_torch_batch_dict(frame_dict, device):
    batch_dict = {}
    for k, v in frame_dict.items():
        if isinstance(v, np.ndarray):
            if v.dtype.kind in ("U", "S", "O"):
                batch_dict[k] = v
            else:
                tensor = torch.from_numpy(v)
                batch_dict[k] = tensor.to(device, non_blocking=True)
        elif isinstance(v, torch.Tensor):
            batch_dict[k] = v.to(device, non_blocking=True)
        else:
            batch_dict[k] = v
    if "batch_size" not in batch_dict:
        batch_dict["batch_size"] = 1
    return batch_dict


def build_seq_loader(cfg, logger, workers, seq_len, stride):
    dataset_cfg = cfg.DATA_CONFIG
    class_names = cfg.CLASS_NAMES

    train_set = NuScenesSeqDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        training=True,
        root_path=None,
        logger=logger,
        seq_len=seq_len,
        stride=stride,
        nusc_version=dataset_cfg.VERSION,
        nusc_dataroot=dataset_cfg.DATA_PATH,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=min(workers, cfg.OPTIMIZATION.NUM_WORKERS),
        pin_memory=True,
        collate_fn=collate_seq,
        drop_last=False,
    )

    return train_set, train_loader


def train_one_epoch(model, optimizer, train_loader, epoch, total_epochs, logger, device, max_grad_norm, alpha_temporal: float):
    model.train()
    for seq_idx, seq in enumerate(train_loader):
        frames = seq["frames"]
        T_world_lidar = seq["T_world_lidar"]
        T = len(frames)

        if hasattr(model, "reset_sequence"):
            model.reset_sequence(seq_idx)

        det_instance_masks_prev = None
        total_loss = 0.0
        accumulated_tb_dict = {}
        accumulated_disp_dict = {}

        for t in range(T):
            frame = frames[t]
            batch_dict = to_torch_batch_dict(frame, device)

            if t == 0:
                T_rel = None
            else:
                T_rel_np = rel_T_curr_prev(T_world_lidar, t)
                T_rel = torch.from_numpy(T_rel_np).to(device, non_blocking=True)

            ret_dict, tb_dict, disp_dict, det_instance_masks_prev = model(
                batch_dict,
                t_seq=t,
                det_instance_masks_prev=det_instance_masks_prev,
                T_rel=T_rel,
                alpha_temporal=alpha_temporal,
            )

            if isinstance(det_instance_masks_prev, torch.Tensor):
                det_instance_masks_prev = det_instance_masks_prev.detach()

            loss_t = ret_dict["loss"]
            total_loss = total_loss + loss_t

            for key, val in tb_dict.items():
                v = val.item() if hasattr(val, "item") else float(val)
                accumulated_tb_dict[key] = accumulated_tb_dict.get(key, 0.0) + v

            for key, val in disp_dict.items():
                v = val.item() if hasattr(val, "item") else float(val)
                accumulated_disp_dict[key] = accumulated_disp_dict.get(key, 0.0) + v

        total_loss = total_loss / max(T, 1)

        for key in accumulated_tb_dict:
            accumulated_tb_dict[key] /= max(T, 1)
        for key in accumulated_disp_dict:
            accumulated_disp_dict[key] /= max(T, 1)

        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if (seq_idx + 1) % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            loss_str = f"epoch {epoch + 1}/{total_epochs}, seq {seq_idx + 1}/{len(train_loader)}, "
            loss_str += f"total_loss {total_loss.item():.4f}, "
            if "rpn_loss_cls" in accumulated_tb_dict:
                loss_str += f"cls {accumulated_tb_dict['rpn_loss_cls']:.4f}, "
            if "rpn_loss_loc" in accumulated_tb_dict:
                loss_str += f"loc {accumulated_tb_dict['rpn_loss_loc']:.4f}, "
            if "rpn_loss_dir" in accumulated_tb_dict:
                loss_str += f"dir {accumulated_tb_dict['rpn_loss_dir']:.4f}, "
            if "aux_motion_tf" in accumulated_tb_dict:
                loss_str += f"aux {accumulated_tb_dict['aux_motion_tf']:.4f}, "
            loss_str += f"lr {lr:.6e}"
            logger.info(loss_str)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, required=True)
    parser.add_argument("--xmem_cfg", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extra_tag", type=str, default="default")
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=35.0)

    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--pretrained_pp_ckpt", type=str, default=None)

    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument("--stage2a_epochs", type=int, default=2)
    parser.add_argument("--stage2b_epochs", type=int, default=6)
    parser.add_argument("--stage2c_epochs", type=int, default=5)
    parser.add_argument("--stage3_epochs", type=int, default=5)

    parser.add_argument("--head_lr_mult", type=float, default=0.1)
    parser.add_argument("--backbone2d_lr_mult", type=float, default=0.05)


    # SETUP 
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    xmem_train_cfg = load_xmem_train_cfg(args.xmem_cfg)

    os.makedirs("log", exist_ok=True)
    log_file = os.path.join("log", f"train_temporal_pp_{args.extra_tag}.txt")
    logger = common_utils.create_logger(log_file)

    logger.info(str(cfg))
    logger.info(f"seq_len={args.seq_len}, stride={args.stride}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, train_loader = build_seq_loader(
        cfg=cfg,
        logger=logger,
        workers=args.workers,
        seq_len=args.seq_len,
        stride=args.stride,
    )

    num_class = len(cfg.CLASS_NAMES)
    pc_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE

    model = TemporalPointPillar(
        model_cfg=cfg.MODEL,
        num_class=num_class,
        dataset=train_set.base,
        xmem_train_cfg=xmem_train_cfg,
        pc_range=pc_range,
    )
    model.to(device)
    names = [n for n, _ in model.named_parameters()]
    tops = sorted({n.split(".")[0] for n in names})
    print(tops)

    resume_blob = None
    start_epoch = 0

    if args.resume_ckpt:
        resume_blob = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(resume_blob["model_state"], strict=True)
        start_epoch = int(resume_blob.get("epoch", 0))
    elif args.pretrained_pp_ckpt:
        ckpt = torch.load(args.pretrained_pp_ckpt, map_location="cpu")
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=False)


    total_epochs = int(cfg.OPTIMIZATION.NUM_EPOCHS)




    # Training loop with staged training
    logger.info("Start training temporal PointPillar with cyclical XMem gating")

    e1, e2a, e2b, e2c, e3 = stage_plan(args)
    total_epochs = e3

    optimizer = None
    prev_stage = None

    for epoch in range(start_epoch, total_epochs):
        st = stage_name(epoch, args)
        alpha = alpha_for_epoch(epoch, args)

        if st != prev_stage:
            if st in ["stage1", "stage2a"]:
                set_trainable_prefixes(model, ["xmem", "motion_mask_tf", "temporal_fusion"])
            elif st == "stage2b":
                set_trainable_prefixes(model, ["xmem", "motion_mask_tf", "temporal_fusion", "dense_head"])
            else:
                set_trainable_prefixes(model, ["xmem", "motion_mask_tf", "temporal_fusion", "dense_head", "backbone_2d"])

            optimizer = build_stage_optimizer(model, cfg, args)

            if resume_blob is not None:
                try:
                    optimizer.load_state_dict(resume_blob["optimizer_state"])
                except Exception as e:
                    logger.info(f"Optimizer state not loaded (safe to ignore): {e}")
                resume_blob = None

            prev_stage = st

        logger.info(f"Epoch {epoch + 1}/{total_epochs} stage={st} alpha={alpha:.3f}")

        train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            epoch=epoch,
            total_epochs=total_epochs,
            logger=logger,
            device=device,
            max_grad_norm=args.max_grad_norm,
            alpha_temporal=alpha,
        )

        ckpt_path = os.path.join("log", f"ckpt_epoch_{epoch + 1}.pth")
        torch.save(
            {
                "model_state": model.state_dict(),
                "epoch": epoch + 1,
                "optimizer_state": optimizer.state_dict(),
                "stage_cfg": {
                    "stage1_epochs": args.stage1_epochs,
                    "stage2a_epochs": args.stage2a_epochs,
                    "stage2b_epochs": args.stage2b_epochs,
                    "stage2c_epochs": args.stage2c_epochs,
                    "stage3_epochs": args.stage3_epochs,
                },
            },
            ckpt_path,
        )
        logger.info(f"Saved checkpoint to {ckpt_path}")



if __name__ == "__main__":
    main()