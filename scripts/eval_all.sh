#!/usr/bin/env bash
set -euo pipefail

ckpts=$(ls -1 log/phase1_ckpt_epoch_*.pth 2>/dev/null | sort -V)

if [ -z "${ckpts}" ]; then
  echo "No checkpoints found: log/phase1_ckpt_epoch_*.pth"
  exit 1
fi

while IFS= read -r ckpt; do
  base=$(basename "${ckpt}")
  epoch=${base#phase1_ckpt_epoch_}
  epoch=${epoch%.pth}
  tag="epoch${epoch}"

  python eval.py \
    --cfg_file xmem_det/configs/temporal_pp_xmem_nuscenes.yaml \
    --xmem_cfg xmem_det/configs/xmem.yaml \
    --ckpt "${ckpt}" \
    --split val \
    --seq_len 8 \
    --alpha 1.0 \
    --log_interval 100 \
    --extra_tag default \
    --eval_tag "${tag}"
done <<< "${ckpts}"
