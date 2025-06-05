_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# please install mmpretrain
# import mmpretrain.models to trigger register_module in mmpretrain
custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa

dataset_type = 'CocoDataset'
data_root = '/home/dhw/yyc_workspace/ConvNeXt-MoE/dataset_coco/'




model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')
        ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(bbox_head=[
        dict(
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
        dict(
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
        dict(
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.033, 0.033, 0.067, 0.067]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
    ]))

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', prob=0.5),
    # dict(
    #     type='RandomChoice',
    #     transforms=[[
    #         dict(
    #             type='RandomChoiceResize',
    #             scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
    #                     (608, 1333), (640, 1333), (672, 1333), (704, 1333),
    #                     (736, 1333), (768, 1333), (800, 1333)],
    #             keep_ratio=True)
    #     ],
    #                 [
    #                     dict(
    #                         type='RandomChoiceResize',
    #                         scales=[(400, 1333), (500, 1333), (600, 1333)],
    #                         keep_ratio=True),
    #                     dict(
    #                         type='RandomCrop',
    #                         crop_type='absolute_range',
    #                         crop_size=(384, 600),
    #                         allow_negative_crop=True),
    #                     dict(
    #                         type='RandomChoiceResize',
    #                         scales=[(480, 1333), (512, 1333), (544, 1333),
    #                                 (576, 1333), (608, 1333), (640, 1333),
    #                                 (672, 1333), (704, 1333), (736, 1333),
    #                                 (768, 1333), (800, 1333)],
    #                         keep_ratio=True)
    #                 ]]),
    dict(
            type='Resize',
            scale=(512, 512),
            keep_ratio=False),
    dict(type='PackDetInputs')
]


backend_args = None

# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4, # batch_size: 每个 GPU 的 batch size
    num_workers=4, # num_workers: 每个 GPU 的线程数
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='anno_train.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='anno_val.json',
        data_prefix=dict(img='val/images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

# ! visual
# vis_backends = [
# dict(type='LocalVisBackend'),
# dict(type='WandbVisBackend'),
# dict(type='TensorboardVisBackend')]

# visualization = _base_.default_hooks.visualization
# visualization.update(dict(draw=True, show=True))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'anno_val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
    classwise=True,)
test_evaluator = val_evaluator


max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 6
    },
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0002,
        betas=(0.9, 0.999),
        weight_decay=0.05))
