from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch
import math
from torch.optim.lr_scheduler import LambdaLR


def set_trainable_prefixes(model, prefixes):
    prefixes = tuple(prefixes)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(prefixes)


def build_optimizer_trainable_only(model, cfg, lr):
    params = []
    seen = set()
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        params.append(p)

    return torch.optim.SGD(
        params,
        lr=float(lr),
        momentum=float(cfg.OPTIMIZATION.MOMENTUM),
        weight_decay=float(cfg.OPTIMIZATION.WEIGHT_DECAY),
    )

def build_warmup_cosine_scheduler(optimizer, steps_per_epoch, epochs, lr_start=1e-4, lr_max=1e-3, lr_end=1e-6, warmup_epochs=1):

    total_steps = int(steps_per_epoch) * int(epochs)
    warmup_steps = int(steps_per_epoch) * int(warmup_epochs)
    warmup_steps = max(1, min(warmup_steps, total_steps - 1))

    for pg in optimizer.param_groups:
        pg["lr"] = lr_max

    start_factor = float(lr_start) / float(lr_max)

    warmup = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    decay_steps = total_steps - warmup_steps

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=decay_steps,
        eta_min=lr_end,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


import torch

def build_optimizer_with_prefix_multipliers(model, cfg, base_lr, group_specs):
    named = list(model.named_parameters())
    selected = []
    seen = set()

    groups = []
    for prefixes, mult in group_specs:
        params = []
        for n, p in named:
            if not p.requires_grad:
                continue
            if not n.startswith(prefixes):
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            params.append(p)
        if params:
            groups.append({"params": params, "lr": float(base_lr) * float(mult)})

    leftovers = []
    for n, p in named:
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        leftovers.append(p)

    if leftovers:
        groups.append({"params": leftovers, "lr": float(base_lr)})

    return torch.optim.SGD(
        groups,
        lr=float(base_lr),
        momentum=float(cfg.OPTIMIZATION.MOMENTUM),
        weight_decay=float(cfg.OPTIMIZATION.WEIGHT_DECAY),
    )


def build_warmup_cosine_factor_scheduler(optimizer, steps_per_epoch, epochs, lr_start, lr_max, lr_end, warmup_epochs=1):
    total_steps = int(steps_per_epoch) * int(epochs)
    warmup_steps = int(steps_per_epoch) * int(warmup_epochs)
    warmup_steps = max(1, min(warmup_steps, total_steps))

    start_factor = float(lr_start) / float(lr_max)
    end_factor = float(lr_end) / float(lr_max)

    def factor(step):
        step = int(step)
        if warmup_steps >= total_steps:
            return 1.0
        if step < warmup_steps:
            if warmup_steps == 1:
                return 1.0
            x = step / float(warmup_steps - 1)
            return start_factor + (1.0 - start_factor) * x
        t = step - warmup_steps
        T = total_steps - warmup_steps
        if T <= 1:
            return end_factor
        x = t / float(T - 1)
        c = 0.5 * (1.0 + math.cos(math.pi * x))
        return end_factor + (1.0 - end_factor) * c

    return LambdaLR(optimizer, lr_lambda=[factor for _ in optimizer.param_groups])
