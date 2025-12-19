
import torch

def collect_params(model, prefixes):
    out = []
    for n, p in model.named_parameters():
        if p.requires_grad and any(n.startswith(pref) for pref in prefixes):
            out.append(p)
    return out


def build_stage_optimizer(model, cfg, args):
    base_lr = float(cfg.OPTIMIZATION.LR)

    temporal_prefixes = ("xmem", "motion_mask_tf", "temporal_fusion")
    head_prefixes = ("dense_head",)
    backbone2d_prefixes = ("backbone_2d",)

    temporal, head, backbone2d = [], [], []
    seen = set()

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue

        if n.startswith(temporal_prefixes):
            temporal.append(p); seen.add(pid)
        elif n.startswith(head_prefixes):
            head.append(p); seen.add(pid)
        elif n.startswith(backbone2d_prefixes):
            backbone2d.append(p); seen.add(pid)

    param_groups = []
    if temporal:
        param_groups.append({"params": temporal, "lr": base_lr})
    if head:
        param_groups.append({"params": head, "lr": base_lr * float(args.head_lr_mult)})
    if backbone2d:
        param_groups.append({"params": backbone2d, "lr": base_lr * float(args.backbone2d_lr_mult)})

    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer. Check requires_grad and stage prefixes.")

    return torch.optim.SGD(
        param_groups,
        lr=base_lr,
        momentum=float(cfg.OPTIMIZATION.MOMENTUM),
        weight_decay=float(cfg.OPTIMIZATION.WEIGHT_DECAY),
    )
