#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
疾病-症状查询示例脚本
演示如何使用标准化后的数据库进行双向查询
"""

import json
from typing import List, Dict, Optional

class DiseaseSymptomQuery:
    def __init__(self):
        self.diseases_db = None
        self.symptom_reverse_db = None
        self.load_databases()
    
    def load_databases(self):
        """加载数据库文件"""
        try:
            # 加载标准化疾病数据库
            with open('normalized_diseases_database.json', 'r', encoding='utf-8') as f:
                diseases_data = json.load(f)
                self.diseases_db = diseases_data.get('疾病库', [])
            
            # 加载症状反向查询表
            with open('symptom_to_disease_mapping.json', 'r', encoding='utf-8') as f:
                reverse_data = json.load(f)
                self.symptom_reverse_db = reverse_data.get('症状反向查询表', {})
            
            print(f"数据库加载成功:")
            print(f"  - 疾病数量: {len(self.diseases_db)}")
            print(f"  - 症状数量: {len(self.symptom_reverse_db)}")
            
        except Exception as e:
            print(f"数据库加载失败: {e}")
    
    def query_disease_by_name(self, disease_name: str) -> Optional[Dict]:
        """根据疾病名称查询疾病信息"""
        for disease in self.diseases_db:
            if disease.get('疾病名称') == disease_name:
                return disease
        return None
    
    def query_symptoms_by_disease(self, disease_name: str) -> List[Dict]:
        """根据疾病名称查询所有症状"""
        disease = self.query_disease_by_name(disease_name)
        if disease:
            return disease.get('症状列表', [])
        return []
    
    def query_diseases_by_symptom_name(self, symptom_name: str) -> List[Dict]:
        """根据症状名称查询相关疾病"""
        # 查找包含该症状名称的症状ID
        matching_symptoms = []
        for symptom_id, symptom_info in self.symptom_reverse_db.items():
            if symptom_name in symptom_info.get('symptom_name', ''):
                matching_symptoms.append(symptom_info)
        
        return matching_symptoms
    
    def query_diseases_by_symptom_id(self, symptom_id: str) -> Optional[Dict]:
        """根据症状ID查询相关疾病"""
        return self.symptom_reverse_db.get(symptom_id)
    
    def find_common_diseases(self, symptom_names: List[str]) -> List[Dict]:
        """根据多个症状查找共同的疾病"""
        if not symptom_names:
            return []
        
        # 获取每个症状对应的疾病
        symptom_diseases = []
        for symptom_name in symptom_names:
            diseases = self.query_diseases_by_symptom_name(symptom_name)
            if diseases:
                # 合并所有匹配症状的疾病
                all_diseases = []
                for symptom_info in diseases:
                    all_diseases.extend(symptom_info.get('diseases', []))
                symptom_diseases.append(set(d['disease_id'] for d in all_diseases))
        
        if not symptom_diseases:
            return []
        
        # 找到所有症状共同的疾病ID
        common_disease_ids = set.intersection(*symptom_diseases)
        
        # 获取共同疾病的详细信息
        common_diseases = []
        for disease in self.diseases_db:
            if disease.get('疾病ID') in common_disease_ids:
                common_diseases.append(disease)
        
        return common_diseases
    
    def get_top_symptoms(self, top_n: int = 10) -> List[Dict]:
        """获取最常见的症状"""
        symptoms_with_count = []
        for symptom_id, symptom_info in self.symptom_reverse_db.items():
            symptoms_with_count.append({
                'symptom_id': symptom_id,
                'symptom_name': symptom_info.get('symptom_name'),
                'disease_count': symptom_info.get('disease_count', 0)
            })
        
        # 按疾病数量排序
        symptoms_with_count.sort(key=lambda x: x['disease_count'], reverse=True)
        return symptoms_with_count[:top_n]

def demo_queries():
    """演示查询功能"""
    print("=" * 60)
    print("疾病-症状双向查询系统演示")
    print("=" * 60)
    
    # 初始化查询系统
    query_system = DiseaseSymptomQuery()
    
    if not query_system.diseases_db or not query_system.symptom_reverse_db:
        print("数据库加载失败，无法进行演示")
        return
    
    print("\n1. 根据疾病查询症状:")
    print("-" * 30)
    disease_name = "喉炎"
    symptoms = query_system.query_symptoms_by_disease(disease_name)
    print(f"疾病: {disease_name}")
    print(f"症状数量: {len(symptoms)}")
    for i, symptom in enumerate(symptoms[:5], 1):  # 只显示前5个
        print(f"  {i}. {symptom.get('symptom_name')} (ID: {symptom.get('symptom_id')})")
    if len(symptoms) > 5:
        print(f"  ... 还有 {len(symptoms) - 5} 个症状")
    
    print("\n2. 根据症状查询疾病:")
    print("-" * 30)
    symptom_name = "发热"
    diseases = query_system.query_diseases_by_symptom_name(symptom_name)
    print(f"症状: {symptom_name}")
    if diseases:
        for symptom_info in diseases:
            print(f"标准化症状名: {symptom_info.get('symptom_name')}")
            print(f"相关疾病数: {symptom_info.get('disease_count')}")
            diseases_list = symptom_info.get('diseases', [])
            for i, disease in enumerate(diseases_list[:5], 1):  # 只显示前5个
                print(f"  {i}. {disease.get('disease_name')} (ID: {disease.get('disease_id')})")
            if len(diseases_list) > 5:
                print(f"  ... 还有 {len(diseases_list) - 5} 个疾病")
            break  # 只显示第一个匹配的症状
    
    print("\n3. 根据多个症状查找共同疾病:")
    print("-" * 30)
    symptom_list = ["发热", "咳嗽", "疲劳"]
    common_diseases = query_system.find_common_diseases(symptom_list)
    print(f"症状组合: {', '.join(symptom_list)}")
    print(f"共同疾病数: {len(common_diseases)}")
    for i, disease in enumerate(common_diseases[:5], 1):  # 只显示前5个
        print(f"  {i}. {disease.get('疾病名称')} (ID: {disease.get('疾病ID')})")
    if len(common_diseases) > 5:
        print(f"  ... 还有 {len(common_diseases) - 5} 个疾病")
    
    print("\n4. 最常见的症状:")
    print("-" * 30)
    top_symptoms = query_system.get_top_symptoms(10)
    for i, symptom in enumerate(top_symptoms, 1):
        print(f"  {i}. {symptom.get('symptom_name')} - {symptom.get('disease_count')} 个疾病")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)

def interactive_query():
    """交互式查询"""
    query_system = DiseaseSymptomQuery()
    
    if not query_system.diseases_db or not query_system.symptom_reverse_db:
        print("数据库加载失败")
        return
    
    print("\n交互式查询模式 (输入 'quit' 退出)")
    print("支持的查询类型:")
    print("1. 疾病查症状: disease <疾病名称>")
    print("2. 症状查疾病: symptom <症状名称>")
    print("3. 多症状查疾病: multi <症状1> <症状2> ...")
    
    while True:
        try:
            user_input = input("\n请输入查询: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            parts = user_input.split()
            if len(parts) < 2:
                print("输入格式错误，请重新输入")
                continue
            
            query_type = parts[0].lower()
            
            if query_type == 'disease':
                disease_name = ' '.join(parts[1:])
                symptoms = query_system.query_symptoms_by_disease(disease_name)
                if symptoms:
                    print(f"\n疾病 '{disease_name}' 的症状:")
                    for symptom in symptoms:
                        print(f"  - {symptom.get('symptom_name')}")
                else:
                    print(f"未找到疾病 '{disease_name}'")
            
            elif query_type == 'symptom':
                symptom_name = ' '.join(parts[1:])
                diseases = query_system.query_diseases_by_symptom_name(symptom_name)
                if diseases:
                    print(f"\n症状 '{symptom_name}' 相关的疾病:")
                    for symptom_info in diseases:
                        print(f"标准化症状: {symptom_info.get('symptom_name')}")
                        for disease in symptom_info.get('diseases', [])[:10]:
                            print(f"  - {disease.get('disease_name')}")
                        break
                else:
                    print(f"未找到症状 '{symptom_name}'")
            
            elif query_type == 'multi':
                symptom_names = parts[1:]
                common_diseases = query_system.find_common_diseases(symptom_names)
                if common_diseases:
                    print(f"\n症状 {symptom_names} 的共同疾病:")
                    for disease in common_diseases[:10]:
                        print(f"  - {disease.get('疾病名称')}")
                else:
                    print(f"未找到症状 {symptom_names} 的共同疾病")
            
            else:
                print("不支持的查询类型")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"查询出错: {e}")
    
    print("退出交互式查询")

if __name__ == "__main__":
    # 运行演示
    demo_queries()
    
    # 可选：运行交互式查询
    # interactive_query()
