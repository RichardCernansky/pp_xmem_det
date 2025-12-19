#!/usr/bin/env bash
set -euo pipefail

CFG_FILE="xmem_det/configs/temporal_pp_xmem_nuscenes.yaml"
XMEM_CFG="xmem_det/configs/xmem.yaml"
PP_BASELINE_CKPT="/home/cernanskyr/OpenPCDet/output/cfgs/nuscenes_models/cbgs_pp_multihead/default/ckpt/checkpoint_epoch_20.pth"

EXTRA_TAG="tp_xmem_staged"
WORKERS=4
SEQ_LEN=8
STRIDE=4
MAX_GRAD_NORM=35.0

STAGE1_EPOCHS=5
STAGE2A_EPOCHS=2
STAGE2B_EPOCHS=6
STAGE2C_EPOCHS=5
STAGE3_EPOCHS=5

HEAD_LR_MULT=0.1
BACKBONE2D_LR_MULT=0.05

python train_temporal.py \
  --cfg_file "$CFG_FILE" \
  --xmem_cfg "$XMEM_CFG" \
  --pretrained_pp_ckpt "$PP_BASELINE_CKPT" \
  --extra_tag "$EXTRA_TAG" \
  --workers "$WORKERS" \
  --seq_len "$SEQ_LEN" \
  --stride "$STRIDE" \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --stage1_epochs "$STAGE1_EPOCHS" \
  --stage2a_epochs "$STAGE2A_EPOCHS" \
  --stage2b_epochs "$STAGE2B_EPOCHS" \
  --stage2c_epochs "$STAGE2C_EPOCHS" \
  --stage3_epochs "$STAGE3_EPOCHS" \
  --head_lr_mult "$HEAD_LR_MULT" \
  --backbone2d_lr_mult "$BACKBONE2D_LR_MULT"
