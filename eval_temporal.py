import argparse
from pathlib import Path

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

import xmem_det


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()

    test_set, test_loader, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=test_set
    )
    ckpt = torch.load(args.ckpt, map_location="cuda")
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.cuda()
    model.eval()

    det_annos = []
    with torch.no_grad():
        for batch_dict in test_loader:
            for k, v in batch_dict.items():
                if isinstance(v, torch.Tensor):
                    batch_dict[k] = v.cuda(non_blocking=True)
            pred_dicts, recall_dicts = model(batch_dict)
            det_annos.extend(pred_dicts)

    output_dir = Path("output_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_str, result_dict = test_set.dataset.evaluation(
        det_annos,
        cfg.CLASS_NAMES,
        output_path=output_dir
    )
    logger.info(result_str)
    print(result_str)


if __name__ == "__main__":
    main()
