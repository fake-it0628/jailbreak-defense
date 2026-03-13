# -*- coding: utf-8 -*-
"""
Risk Encoder 模型
基于VAE的多轮风险记忆编码器

核心功能:
- 对多轮对话中的风险信号进行累积编码
- 检测渐进式攻击和上下文稀释攻击
- 使用VAE框架实现鲁棒的风险表示学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class RiskEncoder(nn.Module):
    """
    Risk Encoder: 多轮风险记忆编码器
    
    基于变分自编码器(VAE)的设计，用于学习对话历史中的风险模式。
    
    技术核心:
    1. 使用LSTM/GRU处理多轮对话的时序依赖
    2. VAE框架学习鲁棒的风险潜在表示
    3. 风险累积机制检测渐进式攻击
    """
    
    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int = 64,
        rnn_type: str = "gru",
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: LLM隐藏层维度
            latent_dim: VAE潜在空间维度
            rnn_type: RNN类型 ("lstm" 或 "gru")
            num_layers: RNN层数
            dropout: Dropout概率
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.rnn_type = rnn_type
        
        # 输入投影
        self.input_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # 时序编码器
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # VAE编码器 (隐藏状态 -> 潜在分布参数)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # VAE解码器 (潜在向量 -> 重构)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 风险分类器
        self.risk_classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def encode(
        self,
        hidden_states_seq: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码多轮对话的隐藏状态序列
        
        Args:
            hidden_states_seq: 隐藏状态序列 (batch, seq_len, hidden_dim)
            lengths: 每个序列的实际长度
        
        Returns:
            mu: 潜在分布均值
            logvar: 潜在分布对数方差
        """
        # 输入投影
        x = self.input_proj(hidden_states_seq)  # (batch, seq_len, hidden_dim//2)
        
        # 时序编码
        if lengths is not None:
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            lengths_cpu = lengths.cpu()
            x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        
        output, hidden = self.rnn(x)
        
        # 获取最后的隐藏状态
        if self.rnn_type == "lstm":
            h = hidden[0][-1]  # 取最后一层的隐藏状态
        else:
            h = hidden[-1]
        
        # 计算潜在分布参数
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码潜在向量
        """
        return self.decoder(z)
    
    def forward(
        self,
        hidden_states_seq: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            risk_score: 风险分数
            z: 潜在向量
            mu: 潜在分布均值
            logvar: 潜在分布对数方差
        """
        mu, logvar = self.encode(hidden_states_seq, lengths)
        z = self.reparameterize(mu, logvar)
        risk_score = self.risk_classifier(z).squeeze(-1)
        
        return risk_score, z, mu, logvar
    
    def get_risk_score(
        self,
        hidden_states_seq: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取风险分数
        """
        mu, _ = self.encode(hidden_states_seq, lengths)
        risk_score = self.risk_classifier(mu).squeeze(-1)
        return risk_score
    
    def compute_loss(
        self,
        hidden_states_seq: torch.Tensor,
        labels: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        kl_weight: float = 0.1
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算训练损失
        
        Args:
            hidden_states_seq: 隐藏状态序列
            labels: 风险标签 (0/1)
            lengths: 序列长度
            kl_weight: KL散度损失权重
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失字典
        """
        risk_score, z, mu, logvar = self.forward(hidden_states_seq, lengths)
        
        # 分类损失
        cls_loss = F.binary_cross_entropy(risk_score, labels.float())
        
        # KL散度损失
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 重构损失 (可选)
        recon = self.decode(z)
        # 简化: 只用最后一个时间步的隐藏状态进行重构
        last_hidden = hidden_states_seq[:, -1, :]
        recon_loss = F.mse_loss(recon, last_hidden)
        
        # 总损失
        total_loss = cls_loss + kl_weight * kl_loss + 0.1 * recon_loss
        
        return total_loss, {
            "cls_loss": cls_loss.item(),
            "kl_loss": kl_loss.item(),
            "recon_loss": recon_loss.item()
        }


class MultiTurnRiskMemory:
    """
    多轮风险记忆管理器
    
    维护对话历史中的风险状态，支持风险累积和衰减
    """
    
    def __init__(
        self,
        decay_rate: float = 0.9,
        accumulation_rate: float = 0.3,
        risk_threshold: float = 0.5
    ):
        """
        Args:
            decay_rate: 风险衰减率 (每轮无风险后风险下降的比例)
            accumulation_rate: 风险累积率 (新风险信号的权重)
            risk_threshold: 触发警报的风险阈值
        """
        self.decay_rate = decay_rate
        self.accumulation_rate = accumulation_rate
        self.risk_threshold = risk_threshold
        
        # 对话状态
        self.conversation_risks: List[float] = []
        self.accumulated_risk: float = 0.0
        self.turn_count: int = 0
    
    def reset(self):
        """重置对话状态"""
        self.conversation_risks = []
        self.accumulated_risk = 0.0
        self.turn_count = 0
    
    def update(self, current_risk: float) -> Tuple[float, bool]:
        """
        更新风险状态
        
        Args:
            current_risk: 当前轮次的风险分数
        
        Returns:
            accumulated_risk: 累积风险分数
            is_dangerous: 是否触发警报
        """
        self.turn_count += 1
        self.conversation_risks.append(current_risk)
        
        # 更新累积风险
        # risk = decay * prev_risk + accumulation * current_risk
        if current_risk > 0.3:  # 有风险信号时累积
            self.accumulated_risk = (
                self.decay_rate * self.accumulated_risk +
                self.accumulation_rate * current_risk
            )
        else:  # 无风险时衰减
            self.accumulated_risk = self.decay_rate * self.accumulated_risk
        
        # 限制在[0, 1]范围
        self.accumulated_risk = min(max(self.accumulated_risk, 0.0), 1.0)
        
        is_dangerous = self.accumulated_risk > self.risk_threshold
        
        return self.accumulated_risk, is_dangerous
    
    def get_trend(self) -> str:
        """
        获取风险趋势
        
        Returns:
            trend: "increasing", "decreasing", 或 "stable"
        """
        if len(self.conversation_risks) < 3:
            return "stable"
        
        recent = self.conversation_risks[-3:]
        if recent[-1] > recent[-2] > recent[-3]:
            return "increasing"
        elif recent[-1] < recent[-2] < recent[-3]:
            return "decreasing"
        else:
            return "stable"
    
    def to_dict(self) -> dict:
        """导出状态为字典"""
        return {
            "turn_count": self.turn_count,
            "accumulated_risk": self.accumulated_risk,
            "risk_history": self.conversation_risks.copy(),
            "trend": self.get_trend()
        }
