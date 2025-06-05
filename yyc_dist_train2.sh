#!/usr/bin/env bash

# CONFIG=configs/convnext_moe/rawtMoE.py
CONFIG=configs/yolox/yolox_s_8xb8-300e_coco.py
GPUS=8
NNODES=1
NODE_RANK=0
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=1,2,3,4
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --launcher pytorch ${@:3} \
    --work-dir=work_dir/yolox_s_8xb8-300e_coco——cls3
    # --work-dir=work_dir/cas-t-raw-MoE # \
    # --resume=/home/dhw/yyc_workspace/ConvNeXt-MoE/work_dir/cas-t-raw/epoch_35.pth
