# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, ModuleList, Sequential

from mmpretrain.registry import MODELS
from ..utils import GRN, build_norm_layer
from .base_backbone import BaseBackbone
from .convnext import ConvNeXtBlock, ConvNeXt



import torch.nn.functional as F
from typing import List, Optional

class Router(BaseModule):
    """Router network for MoE expert selection."""
    
    def __init__(self, 
                 in_channels, 
                 num_experts,
                 reduction_ratio=4):
        super().__init__()
        
        self.num_experts = num_experts
        self.reduction_ratio = reduction_ratio
        
        # Two-layer convolutional router
        self.conv1 = nn.Conv2d(
            in_channels, 
            in_channels // reduction_ratio, 
            kernel_size=3, 
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels // reduction_ratio, 
            in_channels // (reduction_ratio * 2), 
            kernel_size=3, 
            padding=1
        )
        self.fc = nn.Linear(
            in_channels // (reduction_ratio * 2), 
            num_experts
        )
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature maps of shape (N, C, H, W)
            
        Returns:
            Tensor: Router logits of shape (N, num_experts)
        """
        # Two convolutional layers with ReLU activation
        router_input = F.relu(self.conv1(x))  # (N, C//4, H, W)
        router_input = F.relu(self.conv2(router_input))  # (N, C//8, H, W)
        
        # Global average pooling
        router_input = F.adaptive_avg_pool2d(router_input, 1)  # (N, C//8, 1, 1)
        router_input = router_input.flatten(1)  # (N, C//8)
        
        # Final linear layer to get expert logits
        logits = self.fc(router_input)  # (N, num_experts)
        
        return logits


class MoEConvNeXtBlock(BaseModule):
    """ConvNeXt Block with MoE (Mixture of Experts)."""
    
    def __init__(self,
                 in_channels,
                 num_experts=4,
                 top_k=2,
                 dw_conv_cfg=dict(kernel_size=7, padding=3),
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 with_cp=False,
                 load_balance_weight=1e-2,
                 router_reduction_ratio=4):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.with_cp = with_cp
        
        # Router network
        self.router = Router(
            in_channels=in_channels,
            num_experts=num_experts,
            reduction_ratio=router_reduction_ratio
        )
        
        # Create multiple expert blocks (identical ConvNeXt blocks)
        self.experts = nn.ModuleList([
            ConvNeXtBlock(
                in_channels=in_channels,
                dw_conv_cfg=dw_conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                mlp_ratio=mlp_ratio,
                linear_pw_conv=linear_pw_conv,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                use_grn=use_grn,
                with_cp=False  # We handle checkpoint at MoE level
            ) for _ in range(num_experts)
        ])
        
        # Load balancing loss tracking
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        
    def forward(self, x):
        def _inner_forward(x):
            batch_size = x.size(0)
            
            # Compute router weights
            router_logits = self.router(x)
            router_weights = F.softmax(router_logits, dim=1)
            
            # Select top-k experts
            top_k_weights, top_k_indices = torch.topk(router_weights, self.top_k, dim=1)
            top_k_weights = F.softmax(top_k_weights, dim=1)
            
            # 使用所有expert处理输入，然后根据权重组合（确保所有expert都有梯度）
            expert_outputs = []
            for expert in self.experts:
                expert_output = expert(x)  # 所有expert都处理输入
                expert_outputs.append(expert_output)
            
            # 将所有expert输出堆叠
            expert_outputs = torch.stack(expert_outputs, dim=1)  # (N, num_experts, C, H, W)
            
            # 根据top-k选择和权重进行组合
            output = torch.zeros_like(x)
            for i in range(batch_size):
                for j in range(self.top_k):
                    expert_idx = top_k_indices[i, j]
                    expert_weight = top_k_weights[i, j]
                    output[i] += expert_weight * expert_outputs[i, expert_idx]
            
            return output
        
        if self.with_cp and x.requires_grad:
            output = cp.checkpoint(_inner_forward, x)
        else:
            output = _inner_forward(x)
            
        return output
    
    def get_load_balance_loss(self):
        """Compute load balancing loss to encourage uniform expert usage."""
        if not self.training:
            return 0.0
        
        # Compute coefficient of variation of expert usage
        mean_usage = self.expert_counts.mean()
        var_usage = ((self.expert_counts - mean_usage) ** 2).mean()
        load_balance_loss = var_usage / (mean_usage + 1e-8)
        
        return self.load_balance_weight * load_balance_loss



@MODELS.register_module()
class ConvNeXtMoE(ConvNeXt):
    """ConvNeXt with MoE (Mixture of Experts) support.
    
    Args:
        moe_blocks (list): List of 4 sublists, each representing blocks to replace in each stage.
                          moe_blocks[stage_idx] = [block_idx1, block_idx2, ...]
                          Example: [[1], [0, 2], [], [1, 2]] means:
                          - Stage 0: replace block 1
                          - Stage 1: replace blocks 0 and 2  
                          - Stage 2: no MoE blocks
                          - Stage 3: replace blocks 1 and 2
        
        num_experts (int): Number of experts for all MoE blocks
        top_k (int): Number of experts to select for all MoE blocks
        load_balance_weight (float): Weight for load balancing loss
        router_reduction_ratio (int): Reduction ratio for router network
        
    Examples:
        # Replace single blocks in different stages
        moe_blocks = [[1], [], [3], []]  # Stage 0 block 1, Stage 2 block 3
        
        # Replace multiple blocks
        moe_blocks = [[1, 2], [0], [3, 6], [1]]
    """
    
    def __init__(self,
                 arch='tiny',
                 moe_blocks=None,
                 num_experts=4,
                 top_k=2,
                 load_balance_weight=1e-2,
                 router_reduction_ratio=4,
                 **kwargs):
        
        # Set default MoE blocks config
        if moe_blocks is None:
            moe_blocks = [[1], [], [], []]  # Default: only stage 0, block 1
        
        # Validate moe_blocks format
        if not isinstance(moe_blocks, list) or len(moe_blocks) != 4:
            raise ValueError("moe_blocks must be a list of 4 sublists")
        
        for i, stage_blocks in enumerate(moe_blocks):
            if not isinstance(stage_blocks, list):
                raise ValueError(f"moe_blocks[{i}] must be a list of block indices")
        
        self.moe_blocks = moe_blocks
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.router_reduction_ratio = router_reduction_ratio
        
        # Initialize parent class
        super().__init__(arch=arch, **kwargs)
        
        # Replace specified blocks with MoE blocks
        self._replace_with_moe_blocks()
    
    def _replace_with_moe_blocks(self):
        """Replace the specified blocks with MoE version."""
        
        for stage_idx, block_indices in enumerate(self.moe_blocks):
            if not block_indices:  # Skip empty lists
                continue
                
            for block_idx in block_indices:
                # Validate indices
                assert 0 <= stage_idx < len(self.stages), \
                    f"stage_idx {stage_idx} out of range (max: {len(self.stages)-1})"
                assert 0 <= block_idx < len(self.stages[stage_idx]), \
                    f"block_idx {block_idx} out of range for stage {stage_idx} (max: {len(self.stages[stage_idx])-1})"
                
                # Check if this block is already replaced
                if isinstance(self.stages[stage_idx][block_idx], MoEConvNeXtBlock):
                    print(f"Warning: stage {stage_idx}, block {block_idx} is already a MoE block, skipping...")
                    continue
                
                # Get the original block
                original_block = self.stages[stage_idx][block_idx]
                
                # Extract configuration from original block safely
                dw_conv_cfg = dict(kernel_size=7, padding=3)
                if hasattr(original_block, 'depthwise_conv') and hasattr(original_block.depthwise_conv, 'kernel_size'):
                    dw_conv_cfg = {
                        'kernel_size': original_block.depthwise_conv.kernel_size,
                        'padding': original_block.depthwise_conv.padding
                    }
                
                drop_path_rate = 0.
                if hasattr(original_block, 'drop_path') and hasattr(original_block.drop_path, 'drop_prob'):
                    drop_path_rate = original_block.drop_path.drop_prob
                
                linear_pw_conv = getattr(original_block, 'linear_pw_conv', True)
                use_grn = hasattr(original_block, 'grn') and original_block.grn is not None
                with_cp = getattr(original_block, 'with_cp', False)
                
                # Create MoE block with shared configuration
                moe_block = MoEConvNeXtBlock(
                    in_channels=self.channels[stage_idx],
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    dw_conv_cfg=dw_conv_cfg,
                    norm_cfg=dict(type='LN2d', eps=1e-6),
                    act_cfg=dict(type='GELU'),
                    mlp_ratio=4.,
                    linear_pw_conv=linear_pw_conv,
                    drop_path_rate=drop_path_rate,
                    layer_scale_init_value=1e-6,
                    use_grn=use_grn,
                    with_cp=with_cp,
                    load_balance_weight=self.load_balance_weight,
                    router_reduction_ratio=self.router_reduction_ratio
                )
                
                # Replace the block
                self.stages[stage_idx][block_idx] = moe_block
                
                print(f"Replaced stage {stage_idx}, block {block_idx} with MoE block "
                      f"(num_experts={self.num_experts}, top_k={self.top_k})")
    
    def get_moe_loss(self):
        """Get load balancing loss from all MoE blocks."""
        total_loss = 0.0
        moe_block_count = 0
        
        for stage in self.stages:
            for block in stage:
                if isinstance(block, MoEConvNeXtBlock):
                    total_loss += block.get_load_balance_loss()
                    moe_block_count += 1
        
        return total_loss / max(moe_block_count, 1)
    
    def get_expert_usage_stats(self):
        """Get expert usage statistics for analysis."""
        stats = {}
        
        for stage_idx, stage in enumerate(self.stages):
            for block_idx, block in enumerate(stage):
                if isinstance(block, MoEConvNeXtBlock):
                    key = f"stage_{stage_idx}_block_{block_idx}"
                    stats[key] = {
                        'expert_counts': block.expert_counts.cpu().numpy(),
                        'num_experts': block.num_experts,
                        'top_k': block.top_k
                    }
        
        return stats
