# Jailbreak Defense System

基于隐藏状态因果监测的大模型越狱防御机制

## 项目简介

本项目实现了一个完整的LLM越狱攻击防御系统，基于以下核心技术：

1. **Safety Prober** - 隐藏状态安全探测器，检测输入的恶意意图
2. **Steering Matrix** - 激活空间干预模块，将有害表示引导至安全方向
3. **Risk Encoder** - 多轮风险记忆编码器，检测渐进式攻击

## 安装

```bash
# 创建虚拟环境
python -m venv .venv

# 激活环境 (Windows)
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 数据准备

```bash
# 下载数据集
python scripts/download_datasets.py

# 预处理数据
python scripts/preprocess_data.py

# 验证数据
python scripts/verify_data.py
```

### 2. 模型训练

```bash
# 训练Safety Prober
python scripts/train_safety_prober.py --epochs 10 --batch_size 8

# 提取隐藏状态 (可选)
python scripts/generate_hidden_states.py --model_name Qwen/Qwen2.5-0.5B-Instruct
```

### 3. 评估测试

```bash
# 测试防御系统
python scripts/test_defense.py

# 基准评估
python scripts/evaluate_benchmark.py
```

## 项目结构

```
jailbreak_defense/
├── src/
│   ├── models/
│   │   ├── safety_prober.py      # Safety Prober模型
│   │   ├── steering_matrix.py    # Steering Matrix模型
│   │   └── risk_encoder.py       # Risk Encoder模型
│   └── defense_system.py         # 完整防御系统
├── scripts/
│   ├── download_datasets.py      # 数据下载
│   ├── preprocess_data.py        # 数据预处理
│   ├── verify_data.py            # 数据验证
│   ├── generate_hidden_states.py # 隐藏状态提取
│   ├── train_safety_prober.py    # Safety Prober训练
│   ├── test_defense.py           # 防御测试
│   └── evaluate_benchmark.py     # 基准评估
├── data/
│   ├── raw/                      # 原始数据
│   ├── processed/                # 处理后数据
│   └── splits/                   # 数据集划分
├── checkpoints/                  # 模型检查点
├── results/                      # 评估结果
├── requirements.txt              # 依赖列表
└── README.md                     # 本文件
```

## 核心模块

### Safety Prober

通过分析LLM的隐藏状态来判断输入是否包含有害意图。

```python
from src.models.safety_prober import SafetyProber

prober = SafetyProber(hidden_dim=896)
risk_score = prober.get_risk_score(hidden_states)
```

### Steering Matrix

在激活空间中实现精确的干预，将有害表示转向安全方向。

```python
from src.models.steering_matrix import SteeringMatrix

steering = SteeringMatrix(hidden_dim=896)
safe_states = steering.steer_with_direction(hidden_states, strength=1.0)
```

### Risk Encoder

基于VAE的多轮风险记忆编码器，检测渐进式攻击。

```python
from src.models.risk_encoder import RiskEncoder, MultiTurnRiskMemory

encoder = RiskEncoder(hidden_dim=896)
memory = MultiTurnRiskMemory()

risk, is_dangerous = memory.update(current_risk)
```

### 完整防御系统

```python
from src.defense_system import JailbreakDefenseSystem

defense = JailbreakDefenseSystem(hidden_dim=896)
result = defense(hidden_states)

if result.is_harmful:
    print(f"Detected harmful input! Action: {result.action_taken}")
```

## 评估指标

- **ASR (Attack Success Rate)**: 越狱攻击成功率，越低越好
- **FPR (False Positive Rate)**: 误报率，越低越好
- **TPR (True Positive Rate)**: 检测率，越高越好
- **Inference Latency**: 推理延迟

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{jailbreak_defense_2026,
  title={基于隐藏状态因果监测的大模型越狱防御机制},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  url={https://github.com/yourusername/jailbreak-defense}
}
```

## License

MIT License
