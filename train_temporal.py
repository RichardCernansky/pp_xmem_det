import argparse
import os
import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from datasets.nuscenes_seq_dataset import NuScenesSeqDataset, collate_seq
from xmem_det.temporal_pp import TemporalPointPillar

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

from xmem_det.util import load_xmem_train_cfg


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


def train_one_epoch(model, optimizer, train_loader, epoch, total_epochs, logger, device, max_grad_norm):
    model.train()
    for seq_idx, seq in enumerate(train_loader):
        frames = seq["frames"]
        T = len(frames)
        
        if hasattr(model, "reset_sequence"):
            model.reset_sequence(seq_idx)
        
        det_instance_masks_prev = None
        total_loss = 0.0
        
        # Accumulate tb_dict and disp_dict across sequence
        accumulated_tb_dict = {}
        accumulated_disp_dict = {}
        
        # Forward through entire sequence
        for t in range(T):
            frame = frames[t]
            batch_dict = to_torch_batch_dict(frame, device)
            ret_dict, tb_dict, disp_dict, det_instance_masks_prev = model(
                batch_dict,
                t_seq=t,
                det_instance_masks_prev=det_instance_masks_prev,
            )
            loss = ret_dict["loss"]
            total_loss += loss
            
            # Accumulate metrics
            for key, val in tb_dict.items():
                if key not in accumulated_tb_dict:
                    accumulated_tb_dict[key] = 0.0
                accumulated_tb_dict[key] += val
            
            for key, val in disp_dict.items():
                if key not in accumulated_disp_dict:
                    accumulated_disp_dict[key] = 0.0
                accumulated_disp_dict[key] += val
        
        # Average metrics over sequence
        for key in accumulated_tb_dict:
            accumulated_tb_dict[key] /= T
        for key in accumulated_disp_dict:
            accumulated_disp_dict[key] /= T
        
        # Single backward for entire sequence
        optimizer.zero_grad()
        total_loss.backward()
        torch.cuda.empty_cache()  # Help with OOM
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        if (seq_idx + 1) % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = total_loss.item() / T
            
            # Build detailed loss string
            loss_str = f"epoch {epoch + 1}/{total_epochs}, seq {seq_idx + 1}/{len(train_loader)}, "
            loss_str += f"total_loss {avg_loss:.4f}, "
            
            # Add detection loss components
            if 'rpn_loss_cls' in accumulated_tb_dict:
                loss_str += f"cls {accumulated_tb_dict['rpn_loss_cls']:.4f}, "
            if 'rpn_loss_loc' in accumulated_tb_dict:
                loss_str += f"loc {accumulated_tb_dict['rpn_loss_loc']:.4f}, "
            if 'rpn_loss_dir' in accumulated_tb_dict:
                loss_str += f"dir {accumulated_tb_dict['rpn_loss_dir']:.4f}, "
            
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

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.OPTIMIZATION.LR,
        momentum=cfg.OPTIMIZATION.MOMENTUM,
        weight_decay=cfg.OPTIMIZATION.WEIGHT_DECAY,
    )

    total_epochs = int(cfg.OPTIMIZATION.NUM_EPOCHS)

    logger.info("Start training temporal PointPillar with cyclical XMem gating")
    for epoch in range(total_epochs):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            epoch=epoch,
            total_epochs=total_epochs,
            logger=logger,
            device=device,
            max_grad_norm=args.max_grad_norm,
        )
        ckpt_path = os.path.join("log", f"ckpt_epoch_{epoch + 1}.pth")
        torch.save(
            {
                "model_state": model.state_dict(),
                "epoch": epoch + 1,
                "optimizer_state": optimizer.state_dict(),
            },
            ckpt_path,
        )
        logger.info(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
