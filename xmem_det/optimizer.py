import math
import torch


class WarmRestartCosineDecay:
    def __init__(self, optimizer, steps_per_cycle, lr_max, lr_min, gamma):
        self.optimizer = optimizer
        self.steps_per_cycle = int(steps_per_cycle)
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)
        self.gamma = float(gamma)
        self.t = 0
        self.cycle = 0
        self._apply_lr(self.lr_max)

    def _apply_lr(self, lr):
        lr = float(lr)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def _lr_at_t(self):
        if self.steps_per_cycle <= 1:
            return self.lr_min
        x = self.t / (self.steps_per_cycle - 1)
        c = 0.5 * (1.0 + math.cos(math.pi * x))
        return self.lr_min + (self.lr_max - self.lr_min) * c

    def step(self):
        lr = self._lr_at_t()
        self._apply_lr(lr)

        self.t += 1
        if self.t >= self.steps_per_cycle:
            self.t = 0
            self.cycle += 1
            self.lr_max = max(self.lr_min, self.lr_max * self.gamma)
            self._apply_lr(self.lr_max)

        return lr

    def state_dict(self):
        return {
            "steps_per_cycle": self.steps_per_cycle,
            "lr_max": self.lr_max,
            "lr_min": self.lr_min,
            "gamma": self.gamma,
            "t": self.t,
            "cycle": self.cycle,
        }

    def load_state_dict(self, d):
        self.steps_per_cycle = int(d["steps_per_cycle"])
        self.lr_max = float(d["lr_max"])
        self.lr_min = float(d["lr_min"])
        self.gamma = float(d["gamma"])
        self.t = int(d["t"])
        self.cycle = int(d["cycle"])
        self._apply_lr(self._lr_at_t())


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
