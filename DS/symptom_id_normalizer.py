#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
症状ID标准化和反向查询表生成器
功能：
1. 统一相同症状名称的ID
2. 合并语义相近的症状
3. 生成症状到疾病的反向查询表
"""

import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import hashlib

class SymptomNormalizer:
    def __init__(self):
        # 症状名称标准化映射
        self.symptom_mappings = {}
        # 语义相近症状的合并规则
        self.semantic_groups = self._create_semantic_groups()
        # 统一后的症状ID映射
        self.unified_symptom_ids = {}
        # 症状到疾病的反向映射
        self.symptom_to_diseases = defaultdict(list)
        
    def _create_semantic_groups(self) -> Dict[str, List[str]]:
        """
        创建语义相近症状的分组规则
        """
        return {
            # 疼痛相关
            "疼痛": ["疼痛", "痛", "疼", "酸痛", "刺痛", "胀痛", "隐痛", "钝痛", "剧痛", "剧烈疼痛"],
            "头痛": ["头痛", "头疼", "偏头痛", "头部疼痛", "颅内压增高", "头胀痛"],
            "腹痛": ["腹痛", "腹部疼痛", "肚子痛", "胃痛", "胃部疼痛", "上腹痛", "下腹痛", "上腹部疼痛", "下腹部疼痛"],
            "胸痛": ["胸痛", "胸部疼痛", "心前区疼痛", "胸闷痛"],
            "关节痛": ["关节疼痛", "关节痛", "关节酸痛", "关节胀痛"],
            "肌肉痛": ["肌肉疼痛", "肌肉痛", "肌肉酸痛", "肌痛"],
            "腰痛": ["腰痛", "腰部疼痛", "腰酸", "腰背痛"],
            "耳痛": ["耳痛", "耳部疼痛", "耳朵痛"],
            "咽痛": ["咽痛", "咽喉痛", "喉咙痛", "吞咽疼痛", "咽部疼痛"],
            
            # 发热相关
            "发热": ["发热", "发烧", "体温升高", "高热", "低热", "中等热度", "持续发热"],
            "寒战": ["寒战", "畏寒", "怕冷", "恶寒"],
            
            # 呼吸相关
            "咳嗽": ["咳嗽", "咳", "干咳", "湿咳", "慢性咳嗽", "持续咳嗽", "刺激性咳嗽"],
            "咳痰": ["咳痰", "咯痰", "痰多", "有痰", "痰液增多", "脓痰", "脓性痰"],
            "咯血": ["咯血", "咳血", "痰中带血", "血痰"],
            "呼吸困难": ["呼吸困难", "气短", "气促", "呼吸急促", "喘息", "喘", "气喘"],
            "胸闷": ["胸闷", "胸部闷胀", "胸部压迫感"],
            
            # 消化相关
            "恶心": ["恶心", "想吐", "恶心感"],
            "呕吐": ["呕吐", "吐", "呕", "干呕"],
            "腹泻": ["腹泻", "拉肚子", "大便次数增多", "稀便", "水样便"],
            "便秘": ["便秘", "大便干燥", "排便困难", "大便秘结"],
            "腹胀": ["腹胀", "腹部胀满", "肚子胀", "胃胀"],
            "食欲不振": ["食欲不振", "食欲减退", "不想吃饭", "厌食"],
            "便血": ["便血", "大便带血", "血便", "黑便", "柏油样便"],
            
            # 神经相关
            "头晕": ["头晕", "眩晕", "头昏", "晕眩", "眩晕感"],
            "失眠": ["失眠", "睡眠障碍", "入睡困难", "睡眠不好"],
            "疲劳": ["疲劳", "乏力", "无力", "疲倦", "精神不振", "体力下降"],
            "意识障碍": ["意识障碍", "意识模糊", "神志不清", "昏迷", "嗜睡"],
            
            # 皮肤相关
            "皮疹": ["皮疹", "皮肤红疹", "红疹", "疹子", "皮肤疹"],
            "瘙痒": ["瘙痒", "痒", "皮肤瘙痒", "皮痒"],
            "红肿": ["红肿", "肿胀", "水肿", "浮肿"],
            "皮肤干燥": ["皮肤干燥", "皮肤粗糙", "皮肤脱屑", "脱皮"],
            
            # 泌尿相关
            "尿频": ["尿频", "小便频繁", "排尿次数增多"],
            "尿急": ["尿急", "急迫性尿意", "尿意急迫"],
            "尿痛": ["尿痛", "排尿疼痛", "小便疼痛", "尿道疼痛", "排尿时疼痛"],
            "血尿": ["血尿", "尿血", "小便带血", "尿液发红"],
            
            # 心血管相关
            "心悸": ["心悸", "心慌", "心跳加快", "心律不齐", "心跳不规律"],
            "胸痛": ["胸痛", "心前区疼痛", "胸部疼痛"],
            
            # 眼部相关
            "视力模糊": ["视力模糊", "视物模糊", "看东西模糊", "视力下降"],
            "眼痛": ["眼痛", "眼部疼痛", "眼睛疼痛"],
            "畏光": ["畏光", "怕光", "光敏感"],
            
            # 其他常见症状
            "体重下降": ["体重下降", "体重减轻", "消瘦", "体重降低"],
            "出汗": ["出汗", "多汗", "盗汗", "自汗", "汗多"],
            "鼻塞": ["鼻塞", "鼻堵", "鼻子不通气"],
            "流鼻涕": ["流鼻涕", "鼻涕", "流涕"],
        }
    
    def _normalize_symptom_name(self, symptom_name: str) -> str:
        """
        标准化症状名称，将语义相近的症状合并
        """
        # 去除空格和标点
        cleaned_name = re.sub(r'[^\w\u4e00-\u9fff]', '', symptom_name)
        
        # 检查是否属于某个语义组
        for standard_name, variants in self.semantic_groups.items():
            for variant in variants:
                if variant in cleaned_name or cleaned_name in variant:
                    return standard_name
        
        return cleaned_name
    
    def _generate_symptom_id(self, normalized_name: str) -> str:
        """
        为标准化的症状名称生成统一的ID
        """
        # 使用症状名称的哈希值生成ID，确保相同症状名称得到相同ID
        hash_obj = hashlib.md5(normalized_name.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()[:8]  # 取前8位
        return f"SYM_{hash_hex.upper()}"
    
    def analyze_symptoms(self, diseases_data: List[Dict]) -> Dict:
        """
        分析所有症状，统计频率和分布
        """
        symptom_counter = Counter()
        disease_symptom_pairs = []
        
        print("正在分析症状数据...")
        
        for disease in diseases_data:
            disease_name = disease.get('疾病名称', '')
            symptoms = disease.get('症状列表', [])
            
            for symptom in symptoms:
                symptom_name = symptom.get('symptom_name', '')
                if symptom_name:
                    normalized_name = self._normalize_symptom_name(symptom_name)
                    symptom_counter[normalized_name] += 1
                    disease_symptom_pairs.append((disease_name, normalized_name, symptom))
        
        print(f"发现 {len(symptom_counter)} 种不同的标准化症状")
        print(f"最常见的症状:")
        for symptom, count in symptom_counter.most_common(10):
            print(f"   - {symptom}: {count}次")
        
        return {
            'symptom_counter': symptom_counter,
            'disease_symptom_pairs': disease_symptom_pairs
        }
    
    def normalize_disease_data(self, diseases_data: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        标准化疾病数据中的症状ID
        """
        print("正在标准化症状ID...")
        
        # 分析症状数据
        analysis = self.analyze_symptoms(diseases_data)
        
        # 为每个标准化症状生成统一ID
        for normalized_name in analysis['symptom_counter'].keys():
            if normalized_name not in self.unified_symptom_ids:
                self.unified_symptom_ids[normalized_name] = self._generate_symptom_id(normalized_name)
        
        # 更新疾病数据
        updated_diseases = []
        
        for disease in diseases_data:
            updated_disease = disease.copy()
            updated_symptoms = []
            
            for symptom in disease.get('症状列表', []):
                original_name = symptom.get('symptom_name', '')
                normalized_name = self._normalize_symptom_name(original_name)
                unified_id = self.unified_symptom_ids.get(normalized_name)
                
                if unified_id:
                    updated_symptom = symptom.copy()
                    updated_symptom['symptom_id'] = unified_id
                    updated_symptom['normalized_name'] = normalized_name
                    updated_symptoms.append(updated_symptom)
                    
                    # 构建反向映射
                    disease_info = {
                        'disease_id': disease.get('疾病ID'),
                        'disease_name': disease.get('疾病名称'),
                        'specificity': symptom.get('specificity', False)
                    }
                    self.symptom_to_diseases[unified_id].append(disease_info)
            
            updated_disease['症状列表'] = updated_symptoms
            updated_diseases.append(updated_disease)
        
        print(f"标准化完成，生成 {len(self.unified_symptom_ids)} 个统一症状ID")
        
        return updated_diseases, self.unified_symptom_ids
    
    def create_reverse_mapping(self) -> Dict:
        """
        创建症状到疾病的反向查询表
        """
        print("正在创建反向查询表...")
        
        reverse_mapping = {}
        
        for symptom_id, diseases in self.symptom_to_diseases.items():
            # 去重疾病
            unique_diseases = []
            seen_disease_ids = set()
            
            for disease in diseases:
                disease_id = disease['disease_id']
                if disease_id not in seen_disease_ids:
                    unique_diseases.append(disease)
                    seen_disease_ids.add(disease_id)
            
            # 找到对应的标准化症状名称
            normalized_name = None
            for name, sid in self.unified_symptom_ids.items():
                if sid == symptom_id:
                    normalized_name = name
                    break
            
            reverse_mapping[symptom_id] = {
                'symptom_name': normalized_name,
                'disease_count': len(unique_diseases),
                'diseases': unique_diseases
            }
        
        print(f"反向查询表创建完成，包含 {len(reverse_mapping)} 个症状")
        
        return reverse_mapping

def main():
    """
    主处理函数
    """
    print("=" * 80)
    print("症状ID标准化和反向查询表生成器")
    print("=" * 80)
    
    # 读取合并后的疾病数据
    input_file = "merged_processed_diseases.json"
    print(f"正在读取文件: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        diseases_data = data.get('疾病库', [])
        print(f"读取到 {len(diseases_data)} 个疾病")
        
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 初始化标准化器
    normalizer = SymptomNormalizer()
    
    # 标准化疾病数据
    updated_diseases, symptom_id_mapping = normalizer.normalize_disease_data(diseases_data)
    
    # 创建反向查询表
    reverse_mapping = normalizer.create_reverse_mapping()
    
    # 保存标准化后的疾病数据
    output_diseases_file = "normalized_diseases_database.json"
    normalized_data = {
        "疾病库": updated_diseases,
        "症状ID映射": symptom_id_mapping,
        "统计信息": {
            "疾病总数": len(updated_diseases),
            "症状总数": len(symptom_id_mapping),
            "处理时间": "2024-10-02"
        }
    }
    
    print(f"正在保存标准化疾病数据: {output_diseases_file}")
    with open(output_diseases_file, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)
    
    # 保存反向查询表
    output_reverse_file = "symptom_to_disease_mapping.json"
    reverse_data = {
        "症状反向查询表": reverse_mapping,
        "统计信息": {
            "症状总数": len(reverse_mapping),
            "平均每症状对应疾病数": sum(item['disease_count'] for item in reverse_mapping.values()) / len(reverse_mapping) if reverse_mapping else 0,
            "处理时间": "2024-10-02"
        }
    }
    
    print(f"正在保存反向查询表: {output_reverse_file}")
    with open(output_reverse_file, 'w', encoding='utf-8') as f:
        json.dump(reverse_data, f, ensure_ascii=False, indent=2)
    
    # 生成统计报告
    print("\n" + "=" * 80)
    print("处理完成统计报告")
    print("=" * 80)
    print(f"原始疾病数: {len(diseases_data)}")
    print(f"标准化后疾病数: {len(updated_diseases)}")
    print(f"统一症状ID数: {len(symptom_id_mapping)}")
    print(f"反向映射条目数: {len(reverse_mapping)}")
    
    # 显示症状分布统计
    disease_counts = [item['disease_count'] for item in reverse_mapping.values()]
    if disease_counts:
        print(f"症状对应疾病数统计:")
        print(f"   - 平均: {sum(disease_counts) / len(disease_counts):.2f}")
        print(f"   - 最多: {max(disease_counts)}")
        print(f"   - 最少: {min(disease_counts)}")
    
    print(f"\n输出文件:")
    print(f"   - 标准化疾病数据库: {output_diseases_file}")
    print(f"   - 症状反向查询表: {output_reverse_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
