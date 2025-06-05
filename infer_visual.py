from mmdet.apis import DetInferencer

import os
import glob
from pathlib import Path
# Initialize the DetInferencer
inferencer = DetInferencer(model='configs/convnext_moe/rawt.py',
                           weights='work_dir/cas-t-raw/epoch_36.pth',
                           device='cuda:0')

# 设置输入和输出目录
input_dir = '/home/dhw/yyc_workspace/ConvNeXt-MoE/dataset_coco/test/images'  # 修改为您的图片目录路径
output_dir = 'visual'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 支持的图片格式
img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']

# 获取所有图片文件
img_files = []
for ext in img_extensions:
    img_files.extend(glob.glob(os.path.join(input_dir, ext)))
    img_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

print(f"Found {len(img_files)} images in {input_dir}")

# 遍历所有图片进行推理
for i, img_file in enumerate(img_files):
    print(f"Processing [{i+1}/{len(img_files)}]: {os.path.basename(img_file)}")
    
    try:
        # 推理并保存结果
        result = inferencer(
            img_file, 
            show=False, 
            out_dir=output_dir, 
            no_save_pred=False,
            pred_score_thr=0.3  # 设置置信度阈值
        )
        
        print(f"✓ Saved result for {os.path.basename(img_file)}")
        
    except Exception as e:
        print(f"✗ Error processing {img_file}: {e}")
        continue

print(f"\nInference completed! Results saved to: {output_dir}")