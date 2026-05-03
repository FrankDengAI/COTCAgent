"""
Example Usage of COTCAgent
Demonstrating a complete patient-agent conversation workflow
"""

import json
import asyncio
from cotc_agent import COTCAgent, DeepSeekConfig


async def demonstrate_cotc_agent():
    """Demonstrate COTCAgent with a realistic patient conversation"""

    print("COTCAgent 演示：时序保健数据思维链补全Agent")
    print("=" * 60)

    # Initialize the agent with your DeepSeek configuration
    config = DeepSeekConfig(
        api_key='xxx',
        api_base="https://api.deepseek.com/v1/chat/completions",
        model="deepseek-chat",
        max_tokens=2000,
        temperature=0.7,
        timeout=180,
        save_temp_files=True  # 启用临时代码保存功能
    )

    agent = COTCAgent(config)

    # Load patient data (random patient from your dataset)
    with open('patient_data/patient_0001.json', 'r', encoding='utf-8') as f:
        patient_data = json.load(f)

    print(f"患者信息: {patient_data['patient_info']['id']}")
    print(f"总指标数: {patient_data['patient_info']['total_indicators']}")
    print(f"现有诊断: {len(patient_data['patient_info']['diseases'])} 种疾病")
    print()

    # Simulated conversation
    conversation = [
        {
            'role': 'patient',
            'message': '我最近肠胃老是疼，而且头也经常晕，晚上睡不着觉，不知道怎么回事，你能帮我看看吗？'
        },
        {
            'role': 'agent',
            'message': '您好！我理解您的不适。基于您描述的症状，我将分析您的时序健康数据来提供专业的医疗建议。请稍候...'
        }
    ]

    # Process patient query
    print("处理患者查询...")
    user_query = conversation[0]['message']

    try:
        result = await agent.process_user_query(user_query, patient_data)

        print("分析完成！")
        print()

        # Display results
        print("分析结果:")
        print("-" * 40)

        # Temporal analysis summary
        temporal = result.get('temporal_analysis', {})
        print(f" 时序分析: {temporal.get('summary', 'N/A')}")

        if 'trends' in temporal:
            print(f" 发现 {len(temporal['trends'])} 个显著趋势:")
            for trend in temporal['trends'][:3]:
                print(f"   • {trend.get('metric', 'Unknown')}: {trend.get('trend_direction', 'unknown')} "
                      f"(斜率: {trend.get('slope', 0):.3f})")

        # Disease risks
        disease_risks = result.get('disease_risks', [])
        print(f"\n 疾病风险评估 (前3名):")
        for i, risk in enumerate(disease_risks[:3], 1):
            print(f"   {i}. {risk.disease_name}")
            print(f"      风险分数: {risk.risk_score:.1%}")
            print(f"      置信度: {risk.confidence:.1%}")
            print(f"      匹配症状: {', '.join(risk.matched_symptoms)}")
            print(f"      缺失症状: {', '.join(risk.missing_symptoms)}")

        # Active inquiry questions
        questions = result.get('active_inquiry_questions', [])
        print(f"\n 主动问诊问题 ({len(questions)} 个问题):")
        for i, question in enumerate(questions[:5], 1):
            print(f"   {i}. {question}")

        # Detailed analysis
        detailed = result.get('detailed_analysis', {})
        if 'risk_assessment' in detailed:
            risk = detailed['risk_assessment']
            print(f"\n 综合风险评估:")
            print(f"   • 风险等级: {risk.get('risk_level', 'unknown')}")
            print(f"   • 后验概率: {risk.get('posterior_probability', 0):.1%}")
            print(f"   • 贝叶斯证据: {risk.get('bayesian_evidence', 'unknown')}")

        print("\n" + "=" * 60)
        print(" 医疗建议:")
        print("-" * 40)
        print("基于上述分析，建议您:")
        print("1. 密切观察症状变化，记录发作频率和严重程度")
        print("2. 考虑咨询消化内科和神经内科医生")
        print("3. 保持规律作息，避免刺激性食物")
        print("4. 如症状加重，及时就医")

        # Simulate follow-up conversation
        print("\n" + "=" * 60)
        print(" 模拟后续对话:")
        print("-" * 40)

        # Patient responds to questions
        patient_responses = [
            "是的，我偶尔会呕吐，主要是吃油腻食物后。",
            "肠胃疼痛是间歇性的，大概每天发作2-3次。",
            "最近饮食没有太大变化，就是工作压力比较大。"
        ]

        for i, response in enumerate(patient_responses, 1):
            print(f"患者回答 {i}: {response}")
            print(f"Agent: 感谢您的详细描述。这有助于我们更准确地评估风险...")

        print("\n 最终评估:")
        print("基于您的完整描述和时序数据，系统将进行更精确的风险重新计算...")

    except Exception as e:
        print(f" 处理过程中出现错误: {e}")
        print("请检查 DeepSeek API 配置和网络连接。")

    print("\n" + "=" * 60)
    print(" COTCAgent 演示完成！")
    print("=" * 60)


def show_system_capabilities():
    """Display system capabilities and mathematical framework"""

    print("\nCOTCAgent 核心能力:")
    print("=" * 50)

    capabilities = [
        {
            "category": "时序数据分析",
            "description": "线性回归、趋势识别、异常检测",
            "methods": "Linear Regression, IQR, Z-score, Moving Averages"
        },
        {
            "category": "统计检验",
            "description": "相关性分析、因果检验、假设检验",
            "methods": "Pearson/Spearman, Granger Causality, T-tests"
        },
        {
            "category": "风险评估",
            "description": "贝叶斯更新、IDF权重、加权匹配",
            "methods": "Bayesian Inference, IDF Weighting, Weighted Matching Score"
        },
        {
            "category": "主动问诊",
            "description": "缺失症状识别、针对性问题生成",
            "methods": "Symptom Gap Analysis, Targeted Question Generation"
        },
        {
            "category": "数学框架",
            "description": "严格的统计学方法和概率模型",
            "methods": "Statistical Rigor, Probabilistic Interpretation, Confidence Intervals"
        }
    ]

    for cap in capabilities:
        print(f"\n- {cap['category']}:")
        print(f"   {cap['description']}")
        print(f"   数学方法: {cap['methods']}")

    print("\n关键算法:")
    print("1. 逆疾病频率 (IDF) 权重计算")
    print("   w_j = log((N + α)/(n_j + β)) + γ")
    print("2. 加权匹配分数")
    print("   R_i = Σ(w_j) for j in (S_d_i ∩ S_p) / Σ(w_k) for k in S_d_i")
    print("3. 贝叶斯风险更新")
    print("   P(disease|symptoms) = P(symptoms|disease) × P(disease) / P(symptoms)")


def show_temp_code_demo():
    """演示临时代码保存功能"""
    print("\n" + "=" * 60)
    print("临时代码查看演示")
    print("=" * 60)

    print("\n要查看AI生成的临时代码，请按以下步骤操作：")
    print("\n1. 修改配置启用临时代码保存：")
    print("   config = DeepSeekConfig(")
    print("       api_key='your_api_key',")
    print("       save_temp_files=True  # 启用此选项")
    print("   )")

    print("\n2. 运行程序后，查看控制台输出中的文件路径")
    print("   示例输出: '临时代码已保存到: C:\\Users\\用户名\\AppData\\Local\\Temp\\tmpXXXXXX.py'")

    print("\n3. 打开该文件查看AI生成的完整分析代码")
    print("   - 时序分析代码：包含统计检验、趋势分析等")
    print("   - 高级分析代码：包含贝叶斯方法、机器学习等")

    print("\n4. 代码执行完成后，临时文件会保留（不会自动删除）")
    print("   方便您研究和调试AI生成的代码逻辑")


if __name__ == "__main__":
    print("COTCAgent 使用示例")
    print("这是一个完整的医疗AI系统演示，展示从患者查询到主动问诊的完整流程")

    # Show system capabilities first
    show_system_capabilities()

    # Show temp code demo
    show_temp_code_demo()

    # Run the demonstration
    asyncio.run(demonstrate_cotc_agent())

    print("\n 使用说明:")
    print("1. 确保已配置 DeepSeek API 密钥")
    print("2. 准备患者时序健康数据文件")
    print("3. 调用 agent.process_user_query() 处理查询")
    print("4. 查看返回的疾病风险和问诊建议")
    print("5. 如需查看AI生成的临时代码，请设置 save_temp_files=True")
    print("\n 核心文件:")
    print("- cotc_agent.py: 主要Agent实现")
    print("- test_cotc_agent.py: 完整的测试套件")
    print("- README_COTCAgent.md: 详细文档")
    print("\n 临时代码保存:")
    print("- 代码保存在系统临时目录中（通常为 C:\\Users\\用户名\\AppData\\Local\\Temp）")
    print("- 文件名格式为随机字符串 + .py")
    print("- 执行完成后自动删除（除非启用 save_temp_files=True）")
