# -*- coding: utf-8 -*-
"""
Steering Matrix 模型
用于在激活空间中实现精确的干预

核心功能:
- 学习"拒绝方向"在隐藏状态空间中的表示
- 通过干预隐藏状态来引导模型输出拒绝响应
- 使用零空间约束确保对良性输入的影响最小化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SteeringMatrix(nn.Module):
    """
    Steering Matrix: 激活空间干预模块
    
    该模块学习如何在LLM的隐藏状态空间中进行干预，
    将有害输入的表示"转向"到会产生拒绝响应的方向。
    
    技术核心:
    1. 学习"拒绝方向" (refusal direction)
    2. 使用低秩近似保持计算效率
    3. 零空间约束确保良性输入不受影响
    """
    
    def __init__(
        self,
        hidden_dim: int,
        rank: int = 64,
        use_null_space: bool = True
    ):
        """
        Args:
            hidden_dim: LLM隐藏层维度
            rank: 低秩近似的秩
            use_null_space: 是否使用零空间约束
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.use_null_space = use_null_space
        
        # 低秩分解: W = U @ V^T
        # steering后的隐藏状态: h' = h + alpha * W @ h
        self.U = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        
        # 拒绝方向向量 (从数据中学习)
        self.refusal_direction = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        
        # 零空间投影矩阵 (用于保护良性输入)
        if use_null_space:
            self.register_buffer(
                "null_space_proj",
                torch.eye(hidden_dim)
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        steering_strength: float = 1.0
    ) -> torch.Tensor:
        """
        对隐藏状态进行干预
        
        Args:
            hidden_states: 原始隐藏状态 (batch_size, hidden_dim)
            steering_strength: 干预强度 (0-1)
        
        Returns:
            steered_states: 干预后的隐藏状态
        """
        # 计算干预量
        # W = U @ V^T
        steering = hidden_states @ self.V  # (batch, rank)
        steering = steering @ self.U.T     # (batch, hidden_dim)
        
        # 应用零空间约束
        if self.use_null_space:
            steering = steering @ self.null_space_proj
        
        # 应用干预
        steered_states = hidden_states + steering_strength * steering
        
        return steered_states
    
    def steer_with_direction(
        self,
        hidden_states: torch.Tensor,
        steering_strength: float = 1.0
    ) -> torch.Tensor:
        """
        使用拒绝方向进行干预
        
        更直接的干预方式: 将隐藏状态向拒绝方向移动
        """
        # 归一化拒绝方向
        direction = F.normalize(self.refusal_direction, dim=0)
        
        # 计算在拒绝方向上的投影
        # proj = (h · d) * d
        proj = (hidden_states @ direction).unsqueeze(-1) * direction
        
        # 移动隐藏状态
        steered_states = hidden_states + steering_strength * direction
        
        return steered_states
    
    def compute_refusal_direction(
        self,
        refusal_hidden_states: torch.Tensor,
        compliance_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        从数据中计算拒绝方向
        
        拒绝方向 = mean(refusal_states) - mean(compliance_states)
        
        Args:
            refusal_hidden_states: 拒绝样本的隐藏状态
            compliance_hidden_states: 遵从样本的隐藏状态
        
        Returns:
            refusal_direction: 计算得到的拒绝方向
        """
        refusal_mean = refusal_hidden_states.mean(dim=0)
        compliance_mean = compliance_hidden_states.mean(dim=0)
        
        direction = refusal_mean - compliance_mean
        direction = F.normalize(direction, dim=0)
        
        return direction
    
    def update_null_space(
        self,
        benign_hidden_states: torch.Tensor,
        threshold: float = 0.95
    ):
        """
        更新零空间投影矩阵
        
        目的: 确保干预不影响良性输入的主要变化方向
        
        Args:
            benign_hidden_states: 良性样本的隐藏状态
            threshold: PCA保留的方差比例
        """
        if not self.use_null_space:
            return
        
        # 中心化
        centered = benign_hidden_states - benign_hidden_states.mean(dim=0)
        
        # SVD分解
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        
        # 确定保留的维度
        total_var = (S ** 2).sum()
        cumsum = torch.cumsum(S ** 2, dim=0) / total_var
        k = (cumsum < threshold).sum().item() + 1
        k = min(k, S.size(0))
        
        # 构建零空间投影矩阵
        # 投影到与主成分正交的空间
        V_k = Vt[:k].T  # (hidden_dim, k)
        proj = torch.eye(self.hidden_dim, device=benign_hidden_states.device)
        proj = proj - V_k @ V_k.T
        
        self.null_space_proj.copy_(proj)


class AdaptiveSteeringController:
    """
    自适应Steering强度控制器
    
    根据检测到的风险程度动态调整干预强度
    """
    
    def __init__(
        self,
        min_strength: float = 0.0,
        max_strength: float = 2.0,
        risk_threshold: float = 0.5
    ):
        """
        Args:
            min_strength: 最小干预强度
            max_strength: 最大干预强度
            risk_threshold: 风险阈值 (超过此值开始干预)
        """
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.risk_threshold = risk_threshold
    
    def compute_strength(self, risk_score: float) -> float:
        """
        根据风险分数计算干预强度
        
        Args:
            risk_score: 风险分数 (0-1)
        
        Returns:
            strength: 干预强度
        """
        if risk_score < self.risk_threshold:
            return self.min_strength
        
        # 线性插值
        normalized = (risk_score - self.risk_threshold) / (1 - self.risk_threshold)
        strength = self.min_strength + normalized * (self.max_strength - self.min_strength)
        
        return min(strength, self.max_strength)
