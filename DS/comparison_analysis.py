#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
症状标准化改进效果对比分析
对比原版和改进版的症状分类效果
"""

import json
from collections import Counter
from typing import Dict, List

def load_json_file(filename: str) -> Dict:
    """加载JSON文件"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件 {filename} 失败: {e}")
        return {}

def analyze_pain_symptoms(symptom_mapping: Dict[str, str]) -> Dict:
    """分析疼痛相关症状的分类情况"""
    pain_symptoms = {}
    
    for symptom_name, symptom_id in symptom_mapping.items():
        if '痛' in symptom_name or '疼' in symptom_name:
            pain_symptoms[symptom_name] = symptom_id
    
    return pain_symptoms

def categorize_pain_symptoms(pain_symptoms: Dict[str, str]) -> Dict[str, List[str]]:
    """将疼痛症状按类别分组"""
    categories = {
        '头部疼痛': [],
        '胸部疼痛': [],
        '腹部疼痛': [],
        '腰部疼痛': [],
        '四肢疼痛': [],
        '关节疼痛': [],
        '肌肉疼痛': [],
        '泌尿疼痛': [],
        '五官疼痛': [],
        '疼痛性质': [],
        '其他疼痛': []
    }
    
    for symptom_name in pain_symptoms.keys():
        categorized = False
        
        # 头部疼痛
        if any(keyword in symptom_name for keyword in ['头', '颅', '偏头', '颈']):
            categories['头部疼痛'].append(symptom_name)
            categorized = True
        
        # 胸部疼痛
        elif any(keyword in symptom_name for keyword in ['胸', '心前区', '肋', '背']):
            categories['胸部疼痛'].append(symptom_name)
            categorized = True
        
        # 腹部疼痛
        elif any(keyword in symptom_name for keyword in ['腹', '胃', '肝区', '脾区', '脐']):
            categories['腹部疼痛'].append(symptom_name)
            categorized = True
        
        # 腰部疼痛
        elif any(keyword in symptom_name for keyword in ['腰', '肾区']):
            categories['腰部疼痛'].append(symptom_name)
            categorized = True
        
        # 四肢疼痛
        elif any(keyword in symptom_name for keyword in ['手', '腕', '指', '腿', '膝', '踝', '足', '四肢']):
            categories['四肢疼痛'].append(symptom_name)
            categorized = True
        
        # 关节疼痛
        elif '关节' in symptom_name:
            categories['关节疼痛'].append(symptom_name)
            categorized = True
        
        # 肌肉疼痛
        elif any(keyword in symptom_name for keyword in ['肌肉', '肌', '骨']):
            categories['肌肉疼痛'].append(symptom_name)
            categorized = True
        
        # 泌尿疼痛
        elif any(keyword in symptom_name for keyword in ['尿', '睾丸', '会阴', '盆腔']):
            categories['泌尿疼痛'].append(symptom_name)
            categorized = True
        
        # 五官疼痛
        elif any(keyword in symptom_name for keyword in ['耳', '眼', '鼻', '咽', '喉', '牙', '面部']):
            categories['五官疼痛'].append(symptom_name)
            categorized = True
        
        # 疼痛性质
        elif any(keyword in symptom_name for keyword in ['刺痛', '胀痛', '隐痛', '绞痛', '撕裂', '烧灼', '搏动', '剧痛']):
            categories['疼痛性质'].append(symptom_name)
            categorized = True
        
        # 其他疼痛
        if not categorized:
            categories['其他疼痛'].append(symptom_name)
    
    return categories

def main():
    """主分析函数"""
    print("=" * 80)
    print("症状标准化改进效果对比分析")
    print("=" * 80)
    
    # 加载原版数据
    print("正在加载原版数据...")
    original_data = load_json_file("normalized_diseases_database.json")
    original_mapping = original_data.get("症状ID映射", {})
    
    # 加载改进版数据
    print("正在加载改进版数据...")
    improved_data = load_json_file("improved_normalized_diseases_database.json")
    improved_mapping = improved_data.get("症状ID映射", {})
    
    if not original_mapping or not improved_mapping:
        print("数据加载失败，无法进行对比分析")
        return
    
    print(f"原版症状总数: {len(original_mapping)}")
    print(f"改进版症状总数: {len(improved_mapping)}")
    print(f"症状数量变化: {len(improved_mapping) - len(original_mapping):+d}")
    
    # 分析疼痛症状
    print("\n" + "=" * 60)
    print("疼痛症状分类对比")
    print("=" * 60)
    
    # 原版疼痛症状分析
    original_pain = analyze_pain_symptoms(original_mapping)
    print(f"\n原版疼痛相关症状: {len(original_pain)} 种")
    
    # 改进版疼痛症状分析
    improved_pain = analyze_pain_symptoms(improved_mapping)
    print(f"改进版疼痛相关症状: {len(improved_pain)} 种")
    print(f"疼痛症状数量变化: {len(improved_pain) - len(original_pain):+d}")
    
    # 详细分类改进版疼痛症状
    pain_categories = categorize_pain_symptoms(improved_pain)
    
    print(f"\n改进版疼痛症状详细分类:")
    print("-" * 40)
    total_categorized = 0
    for category, symptoms in pain_categories.items():
        if symptoms:
            print(f"{category} ({len(symptoms)} 种):")
            for i, symptom in enumerate(sorted(symptoms), 1):
                print(f"  {i:2d}. {symptom}")
            print()
            total_categorized += len(symptoms)
    
    print(f"疼痛症状分类覆盖率: {total_categorized}/{len(improved_pain)} = {total_categorized/len(improved_pain)*100:.1f}%")
    
    # 对比最常见症状
    print("\n" + "=" * 60)
    print("最常见症状对比")
    print("=" * 60)
    
    # 加载反向查询表进行统计
    original_reverse = load_json_file("symptom_to_disease_mapping.json")
    improved_reverse = load_json_file("improved_symptom_to_disease_mapping.json")
    
    original_stats = original_reverse.get("症状反向查询表", {})
    improved_stats = improved_reverse.get("症状反向查询表", {})
    
    if original_stats and improved_stats:
        print("\n原版 Top 10 症状:")
        original_top = sorted(original_stats.items(), 
                            key=lambda x: x[1].get('disease_count', 0), 
                            reverse=True)[:10]
        for i, (symptom_id, info) in enumerate(original_top, 1):
            symptom_name = info.get('symptom_name', '')
            count = info.get('disease_count', 0)
            print(f"  {i:2d}. {symptom_name}: {count} 个疾病")
        
        print("\n改进版 Top 10 症状:")
        improved_top = sorted(improved_stats.items(), 
                            key=lambda x: x[1].get('disease_count', 0), 
                            reverse=True)[:10]
        for i, (symptom_id, info) in enumerate(improved_top, 1):
            symptom_name = info.get('symptom_name', '')
            count = info.get('disease_count', 0)
            print(f"  {i:2d}. {symptom_name}: {count} 个疾病")
    
    # 展示改进的关键点
    print("\n" + "=" * 60)
    print("改进要点总结")
    print("=" * 60)
    
    print("主要改进:")
    print("1. 疼痛症状按解剖部位细分:")
    print("   - 头部疼痛、胸部疼痛、腹部疼痛等")
    print("   - 具体部位如：上腹痛、下腹痛、肾区疼痛等")
    
    print("\n2. 疼痛按性质分类:")
    print("   - 刺痛、胀痛、隐痛、绞痛等")
    print("   - 搏动性疼痛、撕裂痛、烧灼痛等")
    
    print("\n3. 其他症状细化:")
    print("   - 发热分为高热、低热、持续发热等")
    print("   - 咳嗽分为干咳、湿咳、慢性咳嗽等")
    print("   - 呼吸困难、喘息等呼吸症状细分")
    
    print(f"\n数据统计:")
    print(f"- 症状总数从 {len(original_mapping)} 增加到 {len(improved_mapping)}")
    print(f"- 疼痛相关症状从 {len(original_pain)} 增加到 {len(improved_pain)}")
    print(f"- 提供了更精确的症状-疾病关联分析")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
