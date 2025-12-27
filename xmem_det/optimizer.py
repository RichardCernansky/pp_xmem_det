import torch

def build_stage_optimizer(model, cfg, args):
    base_lr = float(cfg.OPTIMIZATION.LR)

    temporal_prefixes = ("xmem", "motion_transform_net", "aux_head", "temporal_fusion")
    head_prefixes = ("dense_head",)
    backbone2d_prefixes = ("backbone_2d",)

    temporal, head, backbone2d, rest = [], [], [], []
    seen = set()

    for n, p in model.named_parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)

        if n.startswith(temporal_prefixes):
            temporal.append(p)
        elif n.startswith(head_prefixes):
            head.append(p)
        elif n.startswith(backbone2d_prefixes):
            backbone2d.append(p)
        else:
            rest.append(p)

    param_groups = []
    if temporal:
        param_groups.append({"params": temporal, "lr": base_lr})
    if head:
        param_groups.append({"params": head, "lr": base_lr * float(args.head_lr_mult)})
    if backbone2d:
        param_groups.append({"params": backbone2d, "lr": base_lr * float(args.backbone2d_lr_mult)})
    if rest:
        param_groups.append({"params": rest, "lr": base_lr})

    return torch.optim.SGD(
        param_groups,
        lr=base_lr,
        momentum=float(cfg.OPTIMIZATION.MOMENTUM),
        weight_decay=float(cfg.OPTIMIZATION.WEIGHT_DECAY),
    )
