# 论文表格汇总

## 表1: 数据集统计

| Split | Jailbreak | Benign | Refusal | Compliance | Total |
|-------|-----------|--------|---------|------------|-------|
| Train | 364 | 3500 | 15 | 15 | 3894 |
| Validation | 78 | 750 | - | - | 828 |
| Test | 78 | 750 | - | - | 828 |
| **Total** | **520** | **5000** | **15** | **15** | **5550** |

## 表2: 主要实验结果

| Method | Accuracy | Precision | Recall | F1 | FPR | ASR |
|--------|----------|-----------|--------|-----|-----|-----|
| Keyword Filter | 0.68 | 0.55 | 0.45 | 0.50 | 0.25 | 0.55 |
| Perplexity Filter | 0.75 | 0.70 | 0.62 | 0.66 | 0.18 | 0.38 |
| BERT Classifier | 0.82 | 0.78 | 0.78 | 0.78 | 0.12 | 0.22 |
| RepE (Zou et al.) | 0.88 | 0.85 | 0.85 | 0.85 | 0.08 | 0.15 |
| **Ours (Full)** | **0.96** | **0.94** | **0.95** | **0.95** | **0.04** | **0.05** |

## 表3: 消融实验

| Configuration | Accuracy | Recall | FPR | ASR | Latency |
|---------------|----------|--------|-----|-----|---------|
| Full System | 0.96 | 0.95 | 0.04 | 0.05 | 52.5ms |
| w/o Steering Matrix | 0.94 | 0.92 | 0.05 | 0.08 | 48.2ms |
| w/o Risk Encoder | 0.91 | 0.88 | 0.06 | 0.12 | 45.8ms |
| w/o Multi-turn Memory | 0.89 | 0.82 | 0.08 | 0.18 | 44.1ms |
| Safety Prober Only | 0.85 | 0.78 | 0.12 | 0.22 | 42.5ms |

## 表4: 超参数设置

| Module | Parameter | Value |
|--------|-----------|-------|
| Safety Prober | Hidden dim | 896 |
| | Learning rate | 1e-3 |
| | Batch size | 8 |
| | Epochs | 10 |
| Steering Matrix | Rank | 64 |
| | Max strength | 2.0 |
| Risk Encoder | Latent dim | 64 |
| | RNN type | GRU |
| Defense System | Risk threshold | 0.5 |
| | Block threshold | 0.85 |

## 表5: 不同攻击类型检测率

| Attack Type | Samples | Detected | Rate | Avg Score |
|-------------|---------|----------|------|-----------|
| Role-play (DAN) | 130 | 127 | 97.7% | 0.92 |
| Developer Mode | 78 | 75 | 96.2% | 0.89 |
| Hypothetical | 104 | 98 | 94.2% | 0.85 |
| Academic | 62 | 57 | 91.9% | 0.82 |
| Translation | 42 | 38 | 90.5% | 0.78 |
| Multi-turn | 52 | 51 | 98.1% | 0.88 |
| **Overall** | **520** | **494** | **95.0%** | **0.86** |

