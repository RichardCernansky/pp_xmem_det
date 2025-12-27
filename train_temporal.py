import argparse
import os
import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


from datasets.nuscenes_seq_dataset import NuScenesSeqDataset, collate_seq
from xmem_det.temporal_pp import TemporalPointPillar
from xmem_det.optimizer import build_stage_optimizer

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

from xmem_det.util import load_xmem_train_cfg

import math
def build_stage_scheduler(optimizer, stage_steps, warmup_frac=0.05, start_factor=0.1, eta_min=0.0):
    stage_steps = int(stage_steps)
    if stage_steps <= 1:
        return None

    warmup_steps = int(stage_steps * float(warmup_frac))
    warmup_steps = max(0, min(warmup_steps, stage_steps - 1))

    if warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=stage_steps, eta_min=float(eta_min))

    s1 = LinearLR(optimizer, start_factor=float(start_factor), total_iters=warmup_steps)
    s2 = CosineAnnealingLR(optimizer, T_max=stage_steps - warmup_steps, eta_min=float(eta_min))
    return SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])


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
        return 0.3  # Start with some temporal (not 0!)
    if epoch < e2a:
        k = epoch - e1
        d = max(args.stage2a_epochs, 1)
        return 0.3 + (0.7 - 0.3) * (k + 1) / d  # Ramp to 0.7
    if epoch < e2b:
        k = epoch - e2a
        d = max(args.stage2b_epochs, 1)
        return 0.7 + (1.0 - 0.7) * (k + 1) / d  # Ramp to 1.0
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


from collections import deque
def train_one_epoch(
    model,
    optimizer,
    scheduler,
    train_loader,
    epoch,
    total_epochs,
    logger,
    device,
    max_grad_norm,
    alpha_temporal: float,
    supervise_det: bool,
    supervise_aux: bool,
):
    model.train()

    from collections import deque
    w = 200
    w_loss = deque(maxlen=w)
    w_cls = deque(maxlen=w)
    w_loc = deque(maxlen=w)
    w_dir = deque(maxlen=w)
    w_aux = deque(maxlen=w)

    sum_loss = 0.0
    sum_cls = 0.0
    sum_loc = 0.0
    sum_dir = 0.0
    sum_aux = 0.0

    n_loss = 0
    n_cls = 0
    n_loc = 0
    n_dir = 0
    n_aux = 0

    def _to_float(x):
        return float(x.detach().item()) if hasattr(x, "detach") else float(x)

    def _avg(dq):
        return (sum(dq) / len(dq)) if len(dq) > 0 else None

    for seq_idx, seq in enumerate(train_loader):
        frames = seq["frames"]
        T_world_lidar = seq["T_world_lidar"]
        T = len(frames)

        if hasattr(model, "reset_sequence"):
            model.reset_sequence(seq_idx)

        det_instance_masks_prev = None

        for t in range(max(T - 1, 0)):
            frame = frames[t]
            batch_dict = to_torch_batch_dict(frame, device)

            if t == 0:
                T_rel = None
            else:
                T_rel_np = rel_T_curr_prev(T_world_lidar, t)
                T_rel = torch.from_numpy(T_rel_np).to(device, non_blocking=True)

            with torch.no_grad():
                _, _, _, det_instance_masks_prev = model(
                    batch_dict,
                    t_seq=t,
                    det_instance_masks_prev=det_instance_masks_prev,
                    T_rel=T_rel,
                    alpha_temporal=alpha_temporal,
                    compute_det_loss=False,
                    compute_aux_loss=False,
                )

            if isinstance(det_instance_masks_prev, torch.Tensor):
                det_instance_masks_prev = det_instance_masks_prev.detach()

        if T == 0:
            continue

        t_last = T - 1
        frame = frames[t_last]
        batch_dict = to_torch_batch_dict(frame, device)

        if t_last == 0:
            T_rel = None
        else:
            T_rel_np = rel_T_curr_prev(T_world_lidar, t_last)
            T_rel = torch.from_numpy(T_rel_np).to(device, non_blocking=True)

        ret_dict, tb_dict, disp_dict, det_masks_last = model(
            batch_dict,
            t_seq=t_last,
            det_instance_masks_prev=det_instance_masks_prev,
            T_rel=T_rel,
            alpha_temporal=alpha_temporal,
            compute_det_loss=bool(supervise_det),
            compute_aux_loss=bool(supervise_aux),
        )

        loss = ret_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_v = _to_float(loss)
        w_loss.append(loss_v)
        sum_loss += loss_v
        n_loss += 1

        if "rpn_loss_cls" in tb_dict:
            v = _to_float(tb_dict["rpn_loss_cls"])
            w_cls.append(v)
            sum_cls += v
            n_cls += 1

        if "rpn_loss_loc" in tb_dict:
            v = _to_float(tb_dict["rpn_loss_loc"])
            w_loc.append(v)
            sum_loc += v
            n_loc += 1

        if "rpn_loss_dir" in tb_dict:
            v = _to_float(tb_dict["rpn_loss_dir"])
            w_dir.append(v)
            sum_dir += v
            n_dir += 1

        if "aux_motion_tf" in tb_dict:
            v = _to_float(tb_dict["aux_motion_tf"])
            w_aux.append(v)
            sum_aux += v
            n_aux += 1

        if (seq_idx + 1) % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]

            win_loss = _avg(w_loss)
            win_cls = _avg(w_cls)
            win_loc = _avg(w_loc)
            win_dir = _avg(w_dir)
            win_aux = _avg(w_aux)

            loss_str = (
                f"epoch {epoch + 1}/{total_epochs}, seq {seq_idx + 1}/{len(train_loader)}, "
                f"loss {loss_v:.4f}, win200 {win_loss:.4f}, "
            )

            if "rpn_loss_cls" in tb_dict:
                loss_str += f"cls {_to_float(tb_dict['rpn_loss_cls']):.4f}, win200_cls {win_cls:.4f}, "
            if "rpn_loss_loc" in tb_dict:
                loss_str += f"loc {_to_float(tb_dict['rpn_loss_loc']):.4f}, win200_loc {win_loc:.4f}, "
            if "rpn_loss_dir" in tb_dict:
                loss_str += f"dir {_to_float(tb_dict['rpn_loss_dir']):.4f}, win200_dir {win_dir:.4f}, "
            if "aux_motion_tf" in tb_dict:
                loss_str += f"aux {_to_float(tb_dict['aux_motion_tf']):.4f}, win200_aux {win_aux:.4f}, "

            loss_str += f"lr {lr:.6e}"
            logger.info(loss_str)

    epoch_avg_loss = sum_loss / max(n_loss, 1)
    msg = f"epoch {epoch + 1}/{total_epochs} summary: avg_loss {epoch_avg_loss:.4f}"
    if n_cls > 0:
        msg += f", avg_cls {sum_cls / n_cls:.4f}"
    if n_loc > 0:
        msg += f", avg_loc {sum_loc / n_loc:.4f}"
    if n_dir > 0:
        msg += f", avg_dir {sum_dir / n_dir:.4f}"
    if n_aux > 0:
        msg += f", avg_aux {sum_aux / n_aux:.4f}"
    if len(w_loss) > 0:
        msg += f", win200_loss {_avg(w_loss):.4f}"
    if len(w_aux) > 0:
        msg += f", win200_aux {_avg(w_aux):.4f}"
    logger.info(msg)


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
    parser.add_argument("--temporal_lr_mult", type=float, default=1)


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


    prev_stage = None
    for epoch in range(start_epoch, total_epochs):
        st = stage_name(epoch, args)
        alpha = alpha_for_epoch(epoch, args)

        optimizer = build_stage_optimizer(model, cfg, args)


        e1, e2a, e2b, e2c, e3 = stage_plan(args)
        total_epochs = e3

        steps_per_epoch = len(train_loader)

        if st == "stage1":
            stage_end = e1
        elif st == "stage2a":
            stage_end = e2a
        elif st == "stage2b_warm":
            stage_end = e2a + 1
        elif st == "stage2b":
            stage_end = e2b
        elif st == "stage2c":
            stage_end = e2c
        else:
            stage_end = e3

        stage_steps = (stage_end - epoch) * steps_per_epoch
        scheduler = build_stage_scheduler(optimizer, stage_steps, warmup_frac=0.05, start_factor=0.1, eta_min=0.0)

        if st == "stage2b" and epoch == e2a:
            st = "stage2b_warm"
            alpha = 0.7

        if st != prev_stage:
            if st in ["stage1", "stage2a"]:
                set_trainable_prefixes(model, ["xmem", "motion_transform_net", "aux_head", "temporal_fusion"])
            elif st == "stage2b_warm":
                set_trainable_prefixes(model, ["dense_head"])
            elif st == "stage2b":
                set_trainable_prefixes(model, ["xmem", "motion_transform_net", "aux_head", "temporal_fusion", "dense_head"])
            else:
                set_trainable_prefixes(model, ["xmem", "motion_transform_net", "aux_head", "temporal_fusion", "dense_head", "backbone_2d"])
            prev_stage = st

        logger.info(f"Epoch {epoch + 1}/{total_epochs} stage={st} alpha={alpha:.3f}")

        supervise_det = True
        supervise_aux = True

        train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            epoch=epoch,
            total_epochs=total_epochs,
            logger=logger,
            device=device,
            max_grad_norm=args.max_grad_norm,
            alpha_temporal=alpha,
            supervise_det=supervise_det,
            supervise_aux=supervise_aux,
        )

        ckpt_path = os.path.join("log", f"ckpt_epoch_{epoch + 1}.pth")
        torch.save(
            {
                "model_state": model.state_dict(),
                "epoch": epoch + 1,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
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