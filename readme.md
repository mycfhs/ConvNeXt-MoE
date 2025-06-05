## ConvNext with MoE block



### install
`
conda create -n convnext_moe python=3.10
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26

pip install -U openmim
mim install mmengine
mim install mmcv==2.1

cd mmdetection
pip install -v -e .

pip install mmpretrain

pip install tensorboard

cd mmpretrain
mim install -e .

`

### shell
`python
python tools/train.py configs/convnext_moe/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py --work-dir cas-t-raw 


python tools/test.py configs/convnext_moe/rawt.py work_dir/cas-t-raw/epoch_36.pth  --work-dir work_dir/cas-t-raw-eval


<!-- https://blog.csdn.net/Lucy_wzw/article/details/144480725 -->
python tools/analysis_tools/analyze_logs.py plot_curve 'work_dir/cas-t-raw-MoE/20250604_120147/vis_data/20250604_120147.json' --keys bbox_mAP_50 bbox_mAP_75 bbox_mAP --out curve_map.png
python tools/analysis_tools/analyze_logs.py plot_curve 'work_dir/cas-t-raw-MoE/20250604_120147/vis_data/20250604_120147.json' --keys loss --out curv_losse.png
`


### 读数据

使用yolo2cocoDataset.ipynb把yolo格式数据集转为coco格式。 直接修改 configs/_base_/datasets/coco_detection.py为我们的数据路径

另外要把模型config里面的dict(type='LoadAnnotations', with_bbox=True, with_mask=True),  with_mask从True改为False

另外由于数据集没有mask标注 配置文件改为使用cascade-rcnn_r50_fpn.py 而非cascade-mask-rcnn_r50_fpn.py。 且把num_classes都改为了3

bug: 我们转换过的coco数据加载不进去。 debug发现在filter_data()的时候把数据都过滤掉了，原因在于需要把mmdet.datasets.coco里面的METAINFO改为自己的，他在数据过滤的时候会根据这个标注筛选错误标签。 另外coco数据集的class_id要从1开始（0是bg。 不知道不这样会不会报错）