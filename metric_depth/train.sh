#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=120
bs=4
gpus=2
lr=0.000005
encoder=vitl
dataset=SNDataset # vkitti
img_size=518
min_depth=0.001
max_depth=20 # 80 for virtual kitti
pretrained_from=../checkpoints/depth_anything_v2_${encoder}.pth
save_path=exp/hypersim # exp/vkitti

mkdir -p $save_path

python3 train.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth --pretrained-from $pretrained_from \
    --port 20596 2>&1 | tee -a $save_path/$now.log
