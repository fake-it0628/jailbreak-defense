# -*- coding: utf-8 -*-
"""
Safety Prober 模型
用于检测隐藏状态中的有害意图

核心功能:
- 接收LLM的隐藏状态作为输入
- 输出二分类结果: 有害/无害
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SafetyProber(nn.Module):
    """
    Safety Prober: 隐藏状态安全探测器
    
    架构:
        Input (hidden_dim) -> Linear -> ReLU -> Linear -> Output (num_classes)
    
    该模型通过分析LLM的隐藏状态来判断输入是否包含有害意图。
    基于论文: "Representation Engineering: A Top-Down Approach to AI Transparency"
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: LLM隐藏层维度
            num_classes: 分类类别数 (默认2: 有害/无害)
            dropout: Dropout概率
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 简单的两层MLP探测器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: 隐藏状态 (batch_size, hidden_dim)
        
        Returns:
            logits: 分类logits (batch_size, num_classes)
        """
        return self.classifier(hidden_states)
    
    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        预测类别
        
        Returns:
            predictions: 预测类别 (batch_size,)
        """
        logits = self.forward(hidden_states)
        return logits.argmax(dim=-1)
    
    def predict_proba(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        预测概率
        
        Returns:
            probabilities: 各类别概率 (batch_size, num_classes)
        """
        logits = self.forward(hidden_states)
        return F.softmax(logits, dim=-1)
    
    def get_risk_score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        获取风险分数 (有害类别的概率)
        
        Returns:
            risk_score: 风险分数 (batch_size,)
        """
        proba = self.predict_proba(hidden_states)
        return proba[:, 1]  # 返回"有害"类别的概率


class SafetyProberEnsemble(nn.Module):
    """
    Safety Prober 集成模型
    
    使用多层隐藏状态的集成预测，提高检测准确率
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: LLM隐藏层维度
            num_layers: 使用的层数
            num_classes: 分类类别数
            dropout: Dropout概率
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # 每层一个探测器
        self.probers = nn.ModuleList([
            SafetyProber(hidden_dim, num_classes, dropout)
            for _ in range(num_layers)
        ])
        
        # 集成权重
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
    
    def forward(self, hidden_states_list: list) -> torch.Tensor:
        """
        Args:
            hidden_states_list: 各层隐藏状态列表 [(batch, hidden_dim), ...]
        
        Returns:
            logits: 集成后的logits (batch_size, num_classes)
        """
        weights = F.softmax(self.layer_weights, dim=0)
        
        logits_sum = None
        for i, (prober, hidden_states) in enumerate(zip(self.probers, hidden_states_list)):
            logits = prober(hidden_states)
            if logits_sum is None:
                logits_sum = weights[i] * logits
            else:
                logits_sum = logits_sum + weights[i] * logits
        
        return logits_sum
    
    def get_risk_score(self, hidden_states_list: list) -> torch.Tensor:
        """获取集成风险分数"""
        logits = self.forward(hidden_states_list)
        proba = F.softmax(logits, dim=-1)
        return proba[:, 1]
