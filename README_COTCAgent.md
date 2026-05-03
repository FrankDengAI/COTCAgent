# COTCAgent: 时序保健数据思维链补全Agent

## 项目概述

COTCAgent是一个先进的医疗AI系统，专门用于分析患者的时序保健数据，通过思维链推理提供个性化的健康评估和主动问诊建议。该系统实现了完整的医疗决策支持流程，从患者症状描述到疾病风险评估，再到针对性的医疗询问。

## 核心功能

### 1. 高级时序数据分析
- **统计检验方法**:
  - 配对t检验：比较时间点测量值
  - 重复测量ANOVA：分析时间序列方差
  - Wilcoxon检验：非参数时间序列比较
  - 贝叶斯变点检测：识别结构突变

- **高级趋势分析**:
  - STL分解：季节和趋势分解使用Loess
  - 混合效应模型：考虑个体变异
  - 高斯过程回归：建模复杂时间模式
  - 贝叶斯结构时间序列：不确定性量化

- **多变量分析技术**:
  - 向量自回归（VAR）：建模相互依赖关系
  - 格兰杰因果检验：预测关系分析
  - 动态时间规整：序列相似性测量
  - 正则相关分析：多变量关系

- **生存分析方法**:
  - Cox比例风险模型：时变协变量建模
  - 联合模型：纵向-生存数据整合
  - 时间依赖ROC：预测准确性评估
  - 竞争风险模型：多结局分析

- **频域分析**:
  - 小波变换：时频定位
  - 多重分形DFA：相关性特征化
  - 经验模态分解：非线性信号分析
  - 庞加莱图分析：动态系统特征化

### 2. 疾病风险评估
- **IDF权重算法**: 实现逆疾病频率（Inverse Disease Frequency）权重计算
- **加权匹配分数**: 使用公式计算疾病与患者症状的匹配度：
  ```
  R_i = Σ(w_j) for j in (S_d_i ∩ S_p) / Σ(w_k) for k in S_d_i
  ```
- **概率解释**: 将匹配分数解释为相对风险概率
- **前10名风险排序**: 返回最相关的潜在疾病列表

### 3. 主动问诊系统
- **缺失症状识别**: 分析疾病与患者症状的匹配度，找出缺失的关键症状
- **针对性问题生成**: 基于最高风险疾病生成精确的医疗询问
- **对话流程优化**: 提高诊断效率和准确性

### 4. DeepSeek API集成
- **自然语言指令生成**: 将分析需求转换为详细的编程指令
- **代码自动生成**: 基于指令自动编写复杂的统计分析代码
- **临时代码执行**: 生成并执行分析代码，完成后自动清理

## 技术架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  DeepSeek API    │───▶│   Code Writing  │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Temporal Health │───▶│ Statistical      │───▶│ Disease Risk    │
│ Data Analysis   │    │ Analysis Engine  │    │ Calculator      │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Risk Assessment│───▶│ Active Inquiry   │───▶│   Final Report  │
│ & Prioritization│    │ System           │    │                 │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 数学框架

### 1. 统计检验方法
**配对t检验**: 用于比较时间点测量值的统计显著性
```math
t = \frac{\bar{d}}{s_d / \sqrt{n}}, \quad df = n-1
```

**贝叶斯变点检测**: 识别时间序列中的结构突变
```math
P(change|data) = \frac{P(data|change) \cdot P(change)}{P(data)}
```

### 2. 高级趋势分析
**高斯过程回归**: 非参数贝叶斯框架
```math
f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))
```
**后验预测分布**:
```math
f_* | \mathbf{X}, \mathbf{y}, \mathbf{x}_* \sim \mathcal{N}(\bar{f}_*, \mathbb{V}[f_*])
```
其中:
```math
\bar{f}_* = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2\mathbf{I})^{-1} \mathbf{y}
```
```math
\mathbb{V}[f_*] = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2\mathbf{I})^{-1} \mathbf{k}_*
```

### 3. 多变量分析
**向量自回归（VAR）模型**:
```math
\mathbf{y}_t = \mathbf{A}_1 \mathbf{y}_{t-1} + \mathbf{A}_2 \mathbf{y}_{t-2} + \cdots + \mathbf{A}_p \mathbf{y}_{t-p} + \boldsymbol{\epsilon}_t
```
**正则化估计**:
```math
\hat{\mathbf{A}} = \arg\min_{\mathbf{A}} \left\{ \sum_{t=p+1}^T \|\mathbf{y}_t - \sum_{j=1}^p \mathbf{A}_j \mathbf{y}_{t-j}\|_2^2 + \lambda_1 \sum_{j=1}^p \|\mathbf{A}_j\|_1 + \lambda_2 \sum_{j=1}^p \|\mathbf{A}_j\|_F^2 \right\}
```

### 4. 生存分析
**Cox比例风险模型**:
```math
\lambda(t|\mathbf{Z}(t)) = \lambda_0(t) \exp\left(\boldsymbol{\beta}^T \mathbf{Z}(t) + \boldsymbol{\gamma}^T \mathbf{X}\right)
```
**偏似然函数**:
```math
L(\boldsymbol{\beta}, \boldsymbol{\gamma}) = \prod_{i=1}^n \left[ \frac{\exp\left(\boldsymbol{\beta}^T \mathbf{Z}_i(t_i) + \boldsymbol{\gamma}^T \mathbf{X}_i\right)}{\sum_{j \in R(t_i)} \exp\left(\boldsymbol{\beta}^T \mathbf{Z}_j(t_i) + \boldsymbol{\gamma}^T \mathbf{X}_j\right)} \right]^{\delta_i}
```

### 5. 频域分析
**小波变换**:
```math
W_x(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t-b}{a}\right) dt
```
**小波相干性**:
```math
R_{xy}(a, b) = \frac{|S(a^{-1} W_{xy}(a, b))|^2}{S(a^{-1} |W_x(a, b)|^2) S(a^{-1} |W_y(a, b)|^2)}
```

### 6. 疾病风险评估
**IDF权重计算**:
```math
w_j = \log\left( \frac{N + \alpha}{n_j + \beta} \right) + \gamma
```
其中：
- `N`: 疾病总数
- `n_j`: 包含症状 `s_j` 的疾病数
- `α, β = 1`: 拉普拉斯平滑参数
- `γ = 1`: 正权重保证

**加权匹配分数**:
```math
R_i = \frac{ \sum_{s_j \in (S_{d_i} \cap S_p) } w_j }{ \sum_{s_k \in S_{d_i} } w_k }
```
其中：
- `R_i`: 疾病 `d_i` 的匹配分数
- `S_{d_i}`: 疾病 `d_i` 的症状集合
- `S_p`: 患者症状集合

**贝叶斯风险更新**:
```math
P(disease|symptoms) = \frac{P(symptoms|disease) \times P(disease)}{P(symptoms)}
```

## 使用方法

### 1. 环境配置
```bash
pip install numpy pandas scipy scikit-learn aiohttp
```

### 2. API配置
```python
config = DeepSeekConfig(
    api_key='YOUR_API_KEY',
    api_base='https://api.modelarts-maas.com/v1/chat/completions',
    model='DeepSeek-V3'
)
```

### 3. 初始化Agent
```python
agent = COTCAgent(config)
```

### 4. 处理患者查询
```python
patient_data = json.load(open('patient_data/patient_0001.json'))
user_query = "我最近肠胃老是疼，而且头也经常晕，晚上睡不着觉"

result = await agent.process_user_query(user_query, patient_data)
```

### 5. 查看结果
```python
print("Disease Risks:")
for risk in result['disease_risks']:
    print(f"- {risk['disease_name']}: {risk['risk_score']:.3f}")

print("\nActive Inquiry Questions:")
for question in result['active_inquiry_questions']:
    print(f"- {question}")
```

## 测试案例

运行完整测试套件：
```bash
python test_cotc_agent.py
```

测试输出包括：
- 时序数据分析结果
- 统计趋势识别
- 疾病风险评估
- 主动问诊问题生成
- 数学分析质量评估

## 数据格式

### 患者数据结构
```json
{
  "基础体征": {
    "症状名称": {
      "id": "指标ID",
      "时间序列": ["2024-01-01", "2024-01-02", ...],
      "测量值": [数值1, 数值2, ...]
    }
  },
  "patient_info": {
    "id": "patient_0001",
    "diseases": [...],
    "total_indicators": 16
  }
}
```

### 疾病数据库结构
```json
{
  "疾病库": [
    {
      "疾病ID": "D000001",
      "疾病名称": "疾病名称",
      "症状列表": [
        {
          "symptom_id": "S000001",
          "symptom_name": "症状名称",
          "disease_id": "D000001"
        }
      ],
      "疾病解释": "详细医学描述"
    }
  ]
}
```

## 性能特点

1. **数学严谨性**: 所有分析都基于统计学原理，有效性和可靠性高
2. **可解释性**: 提供详细的数学推导和统计检验结果
3. **实时性**: 支持快速的在线诊断和风险评估
4. **可扩展性**: 易于集成新的分析方法和医疗数据库
5. **安全性**: 代码执行在沙箱环境中，确保系统安全

## 应用场景

- **远程医疗**: 为在线医疗平台提供AI辅助诊断
- **健康监测**: 分析可穿戴设备收集的时序健康数据
- **医疗研究**: 辅助临床研究和流行病学分析
- **个性化医疗**: 提供基于患者历史数据的定制化建议

## 开发路线图

- [ ] 集成更多统计模型（ARIMA、机器学习算法）
- [ ] 扩展医疗数据库支持
- [ ] 添加可视化分析模块
- [ ] 实现多语言支持
- [ ] 开发移动端应用接口

## 贡献指南

欢迎提交问题报告、功能建议或代码贡献。请确保：
1. 代码符合项目的数学严谨性要求
2. 包含详细的注释和数学推导
3. 提供相应的测试案例
4. 遵循项目的编码规范

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

---

**作者**: AI Assistant
**版本**: 1.0.0
**更新日期**: 2025年09月28日
