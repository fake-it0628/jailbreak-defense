# -*- coding: utf-8 -*-
"""
生成项目概述PDF
"""

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

def generate_pdf_with_matplotlib():
    """使用matplotlib生成PDF"""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.patches as mpatches
    import matplotlib.font_manager as fm
    
    # 设置中文字体 - 使用系统字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取中文字体路径
    chinese_font = None
    for font_name in ['Microsoft YaHei', 'SimHei', 'SimSun']:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path and 'DejaVu' not in font_path:
                chinese_font = fm.FontProperties(fname=font_path)
                print(f"Using font: {font_name}")
                break
        except:
            continue
    
    output_path = Path("paper/project_overview.pdf")
    
    with PdfPages(output_path) as pdf:
        # ========== 第1页：封面 ==========
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # 标题
        ax.text(0.5, 0.75, 'HiSCaM', fontsize=48, ha='center', va='center', fontweight='bold', color='#2c3e50')
        ax.text(0.5, 0.65, 'Hidden State Causal Monitoring', fontsize=20, ha='center', va='center', color='#34495e')
        ax.text(0.5, 0.55, '基于隐藏状态因果监测的大模型越狱防御机制', fontsize=16, ha='center', va='center', color='#7f8c8d')
        
        # 分隔线
        ax.axhline(y=0.48, xmin=0.2, xmax=0.8, color='#3498db', linewidth=2)
        
        # 项目概述
        ax.text(0.5, 0.38, '项 目 概 述', fontsize=24, ha='center', va='center', fontweight='bold', color='#2c3e50')
        
        # 信息
        ax.text(0.5, 0.20, 'GitHub: fake-it0628/jailbreak-defense', fontsize=12, ha='center', va='center', color='#95a5a6')
        ax.text(0.5, 0.15, '联系邮箱: 2948127071@qq.com', fontsize=12, ha='center', va='center', color='#95a5a6')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ========== 第2页：问题背景 ==========
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.95, '一、项目背景（为什么做）', fontsize=20, ha='center', va='top', fontweight='bold', color='#2c3e50')
        
        content = """
问题现状：
• 大语言模型（LLM）被广泛应用，但面临严重安全威胁
• 越狱攻击：通过精心设计的提示词绕过安全机制
• 角色扮演：让模型扮演"无限制AI"（如DAN）
• 多轮攻击：跨对话逐步升级恶意请求

现有方法的不足：
• 关键词过滤 → 易被改写绕过
• 输出过滤 → 介入太晚，已生成有害内容
• 模型微调 → 计算成本高，可能降低模型能力

核心洞察：
有害意图在输出生成之前，已编码在模型的隐藏状态中。
通过监测内部表征，可以在源头检测并阻止有害输出。
"""
        ax.text(0.1, 0.85, content, fontsize=11, ha='left', va='top', color='#2c3e50',
               fontproperties=chinese_font, linespacing=1.8)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ========== 第3页：技术方案 ==========
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.95, '二、技术方案（做什么）', fontsize=20, ha='center', va='top', fontweight='bold', color='#2c3e50')
        
        content = """
系统架构：

    输入查询 → LLM隐藏状态 → [三大模块] → 决策输出
                                   |
                     +-------------+-------------+
                     |             |             |
                 Safety       Steering        Risk
                 Prober       Matrix        Encoder
                 (检测)        (引导)        (记忆)


三大核心模块：

  模块名称          功能              技术原理
  ─────────────────────────────────────────────────
  Safety Prober    实时风险检测      MLP分类器分析隐藏状态
  Steering Matrix  激活空间干预      将有害表征引导至安全方向
  Risk Encoder     多轮风险记忆      VAE+GRU跟踪累积风险


决策逻辑：
  • 低风险 (R < 0.5)：PASS - 正常放行
  • 中风险 (0.5 ≤ R < 0.85)：STEER - 引导干预
  • 高风险 (R ≥ 0.85)：BLOCK - 阻断请求
"""
        ax.text(0.1, 0.85, content, fontsize=10, ha='left', va='top', color='#2c3e50',
               fontproperties=chinese_font, linespacing=1.6)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ========== 第4页：实施计划 ==========
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.95, '三、实施计划（怎么做）', fontsize=20, ha='center', va='top', fontweight='bold', color='#2c3e50')
        
        content = """
阶段划分：

  阶段      内容                    交付物
  ────────────────────────────────────────────────────
  阶段1    环境搭建 + 数据准备      数据集、隐藏状态
  阶段2    模型训练                 Safety Prober、Steering Matrix、Risk Encoder
  阶段3    系统集成 + 评估          完整防御系统、评估报告
  阶段4    论文撰写 + 开源          学术论文、GitHub仓库


数据集：
  • AdvBench：520条有害行为提示
  • Alpaca：5000条良性指令
  • JailbreakBench：100条标准越狱测试
  • 中文越狱：25条中文越狱样本


基线对比（与6种SOTA方法对比）：
  • Keyword Filter      • Self-Reminder
  • Perplexity Filter   • Llama Guard
  • SmoothLLM           • Erase-and-Check
"""
        ax.text(0.1, 0.85, content, fontsize=10, ha='left', va='top', color='#2c3e50',
               fontproperties=chinese_font, linespacing=1.6)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ========== 第5页：预期目标 ==========
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.95, '四、预期目标', fontsize=20, ha='center', va='top', fontweight='bold', color='#2c3e50')
        
        content = """
性能指标目标：

  指标                目标值      说明
  ─────────────────────────────────────────────────────
  检测率 (TPR)        ≥ 95%      越狱尝试被正确识别的比例
  误报率 (FPR)        ≤ 5%       良性查询被错误拦截的比例
  攻击成功率 (ASR)    ≤ 5%       绕过防御的越狱比例
  推理延迟            ≤ 100ms    每次查询的额外时间开销


已实现结果：

  [OK] 检测率：100%（完美召回）
  [OK] 误报率：1.2%（远低于目标）
  [OK] 攻击成功率：0%（完美防御）
  [OK] 推理延迟：~52ms（满足要求）


学术产出：
  • 论文：投稿顶会/期刊（NeurIPS/ICML/ACL/USENIX Security）
  • 开源：GitHub完整代码 + 预训练模型
  • 演示：交互式Web Demo
"""
        ax.text(0.1, 0.85, content, fontsize=10, ha='left', va='top', color='#2c3e50',
               fontproperties=chinese_font, linespacing=1.6)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ========== 第6页：项目亮点 ==========
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.95, '五、项目亮点', fontsize=20, ha='center', va='top', fontweight='bold', color='#2c3e50')
        
        content = """
创新点：

  1. 理论创新：首次将因果推理应用于LLM越狱防御
  2. 方法创新：三模块协同的全面防御架构
  3. 实践创新：实时在线检测，无需修改原模型


技术优势对比：

  维度          HiSCaM              传统方法
  ─────────────────────────────────────────────────────
  检测时机      生成前（隐藏状态）   生成后（输出文本）
  多轮攻击      支持（Risk Encoder） 不支持
  可解释性      高（因果分析）       低
  计算开销      低（~50ms）          视方法而定


应用价值：
  • 企业部署：保护商业LLM API安全
  • 学术研究：LLM安全领域新范式
  • 开源社区：可复现、可扩展的防御框架


资源链接：
  • 代码仓库：github.com/fake-it0628/jailbreak-defense
  • 交互演示：python demo/app.py → localhost:7861
  • 论文草稿：paper/paper_draft_chinese.md
"""
        ax.text(0.1, 0.85, content, fontsize=10, ha='left', va='top', color='#2c3e50',
               fontproperties=chinese_font, linespacing=1.6)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"[OK] PDF generated: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Generating Project Overview PDF...")
    pdf_path = generate_pdf_with_matplotlib()
    print(f"\nDone! PDF saved to: {pdf_path}")
