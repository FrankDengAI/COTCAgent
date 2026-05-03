#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版症状ID标准化器 - 更细致的疼痛分类
功能：
1. 将疼痛按照具体部位和类型进行细分
2. 统一相同症状名称的ID
3. 合并语义相近的症状
4. 生成症状到疾病的反向查询表
"""

import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import hashlib

class ImprovedSymptomNormalizer:
    def __init__(self):
        # 症状名称标准化映射
        self.symptom_mappings = {}
        # 语义相近症状的合并规则
        self.semantic_groups = self._create_detailed_semantic_groups()
        # 统一后的症状ID映射
        self.unified_symptom_ids = {}
        # 症状到疾病的反向映射
        self.symptom_to_diseases = defaultdict(list)
        
    def _create_detailed_semantic_groups(self) -> Dict[str, List[str]]:
        """
        创建更详细的语义相近症状分组规则，特别是疼痛的细分
        """
        return {
            # ========== 疼痛相关 - 按部位和类型细分 ==========
            
            # 头部疼痛
            "头痛": ["头痛", "头疼", "头部疼痛", "头胀痛", "头部胀痛"],
            "偏头痛": ["偏头痛", "偏侧头痛", "单侧头痛", "搏动性头痛"],
            "颅内压增高性头痛": ["颅内压增高", "颅压增高", "脑压增高头痛"],
            "紧张性头痛": ["紧张性头痛", "紧张型头痛", "压迫性头痛"],
            
            # 面部疼痛
            "面部疼痛": ["面部疼痛", "面痛", "脸痛", "面部不适"],
            "牙痛": ["牙痛", "牙疼", "牙齿疼痛", "牙齿痛"],
            "下颌痛": ["下颌疼痛", "下颌痛", "颞下颌关节痛"],
            
            # 颈部疼痛
            "颈痛": ["颈痛", "颈部疼痛", "脖子痛", "颈椎痛", "颈肩痛"],
            "咽痛": ["咽痛", "咽喉痛", "喉咙痛", "吞咽疼痛", "咽部疼痛", "喉部疼痛"],
            
            # 胸部疼痛
            "胸痛": ["胸痛", "胸部疼痛", "前胸痛"],
            "心前区疼痛": ["心前区疼痛", "心前区痛", "心脏区疼痛", "心口痛"],
            "胸闷痛": ["胸闷痛", "胸部闷痛", "胸部压迫性疼痛"],
            "肋间神经痛": ["肋间神经痛", "肋间痛", "肋骨痛"],
            "背痛": ["背痛", "背部疼痛", "后背痛", "脊背痛"],
            
            # 腹部疼痛
            "上腹痛": ["上腹痛", "上腹部疼痛", "上腹疼痛", "胃区疼痛", "胃痛", "胃部疼痛"],
            "下腹痛": ["下腹痛", "下腹部疼痛", "下腹疼痛", "小腹痛", "小腹疼痛"],
            "腹痛": ["腹痛", "腹部疼痛", "肚子痛", "肚痛"],
            "右上腹痛": ["右上腹痛", "右上腹部疼痛", "肝区疼痛"],
            "左上腹痛": ["左上腹痛", "左上腹部疼痛", "脾区疼痛"],
            "脐周痛": ["脐周痛", "脐周疼痛", "肚脐周围痛"],
            "腹绞痛": ["腹绞痛", "绞痛", "痉挛性腹痛"],
            
            # 腰部疼痛
            "腰痛": ["腰痛", "腰部疼痛", "腰酸", "腰酸痛"],
            "腰背痛": ["腰背痛", "腰背部疼痛", "腰脊痛"],
            "肾区疼痛": ["肾区疼痛", "肾区痛", "腰肾区痛"],
            
            # 四肢疼痛
            "关节痛": ["关节疼痛", "关节痛", "关节酸痛", "关节胀痛"],
            "肌肉痛": ["肌肉疼痛", "肌肉痛", "肌肉酸痛", "肌痛", "肌肉胀痛"],
            "骨痛": ["骨痛", "骨疼", "骨骼疼痛", "骨头痛"],
            "四肢痛": ["四肢痛", "四肢疼痛", "手脚痛"],
            
            # 手部疼痛
            "手痛": ["手痛", "手部疼痛", "手疼"],
            "腕痛": ["腕痛", "手腕痛", "腕关节痛"],
            "指痛": ["指痛", "手指痛", "手指疼痛"],
            
            # 腿部疼痛
            "腿痛": ["腿痛", "腿部疼痛", "下肢痛"],
            "膝痛": ["膝痛", "膝盖痛", "膝关节痛", "膝部疼痛"],
            "踝痛": ["踝痛", "脚踝痛", "踝关节痛"],
            "足痛": ["足痛", "脚痛", "足部疼痛"],
            
            # 盆腔疼痛
            "盆腔痛": ["盆腔痛", "盆腔疼痛", "盆腔不适"],
            "会阴痛": ["会阴痛", "会阴疼痛", "会阴部疼痛"],
            
            # 头颈部其他疼痛
            "耳痛": ["耳痛", "耳部疼痛", "耳朵痛", "耳内疼痛"],
            "眼痛": ["眼痛", "眼部疼痛", "眼睛疼痛", "眼球痛"],
            "鼻痛": ["鼻痛", "鼻部疼痛", "鼻子痛"],
            
            # 泌尿生殖系统疼痛
            "尿痛": ["尿痛", "排尿疼痛", "小便疼痛", "尿道疼痛", "排尿时疼痛", "尿道灼痛"],
            "睾丸痛": ["睾丸痛", "睾丸疼痛", "阴囊痛"],
            
            # 疼痛性质分类
            "刺痛": ["刺痛", "针刺样疼痛", "锐痛"],
            "胀痛": ["胀痛", "胀疼", "胀满痛"],
            "隐痛": ["隐痛", "隐隐作痛", "钝痛"],
            "绞痛": ["绞痛", "痉挛性疼痛", "绞扭样疼痛"],
            "撕裂痛": ["撕裂痛", "撕裂样疼痛", "撕扯痛"],
            "烧灼痛": ["烧灼痛", "灼痛", "烧灼感"],
            "搏动性疼痛": ["搏动性疼痛", "跳痛", "搏动痛"],
            "剧痛": ["剧痛", "剧烈疼痛", "剧疼", "严重疼痛"],
            
            # ========== 发热相关 ==========
            "发热": ["发热", "发烧", "体温升高"],
            "高热": ["高热", "高烧", "高度发热"],
            "低热": ["低热", "低烧", "微热", "轻度发热"],
            "中等热度": ["中等热度", "中度发热", "中等发热"],
            "持续发热": ["持续发热", "持续发烧", "连续发热"],
            "间歇性发热": ["间歇性发热", "间歇发热", "阵发性发热"],
            "寒战": ["寒战", "畏寒", "怕冷", "恶寒", "打寒战"],
            
            # ========== 呼吸相关 ==========
            "咳嗽": ["咳嗽", "咳"],
            "干咳": ["干咳", "无痰咳嗽", "刺激性咳嗽"],
            "湿咳": ["湿咳", "有痰咳嗽"],
            "慢性咳嗽": ["慢性咳嗽", "持续咳嗽", "长期咳嗽"],
            "咳痰": ["咳痰", "咯痰", "痰多", "有痰", "痰液增多"],
            "脓痰": ["脓痰", "脓性痰", "黄痰"],
            "血痰": ["血痰", "痰中带血"],
            "咯血": ["咯血", "咳血", "吐血"],
            "呼吸困难": ["呼吸困难", "气短", "气促", "呼吸急促"],
            "喘息": ["喘息", "喘", "气喘", "哮喘"],
            "胸闷": ["胸闷", "胸部闷胀", "胸部压迫感", "憋气"],
            
            # ========== 消化相关 ==========
            "恶心": ["恶心", "想吐", "恶心感", "反胃"],
            "呕吐": ["呕吐", "吐", "呕", "呕出"],
            "干呕": ["干呕", "干吐", "空呕"],
            "腹泻": ["腹泻", "拉肚子", "大便次数增多", "稀便"],
            "水样便": ["水样便", "水泻", "水样腹泻"],
            "便秘": ["便秘", "大便干燥", "排便困难", "大便秘结"],
            "腹胀": ["腹胀", "腹部胀满", "肚子胀", "胃胀", "腹部胀气"],
            "食欲不振": ["食欲不振", "食欲减退", "不想吃饭", "厌食", "食欲差"],
            "便血": ["便血", "大便带血", "血便"],
            "黑便": ["黑便", "柏油样便", "黑色大便"],
            "里急后重": ["里急后重", "便意频繁", "排便不尽感"],
            
            # ========== 神经相关 ==========
            "头晕": ["头晕", "头昏", "晕"],
            "眩晕": ["眩晕", "晕眩", "眩晕感", "旋转感"],
            "失眠": ["失眠", "睡眠障碍", "入睡困难", "睡眠不好", "不能入睡"],
            "嗜睡": ["嗜睡", "过度睡眠", "睡意浓"],
            "疲劳": ["疲劳", "乏力", "无力", "疲倦", "精神不振", "体力下降", "疲乏"],
            "意识障碍": ["意识障碍", "意识模糊", "神志不清"],
            "昏迷": ["昏迷", "不省人事", "失去意识"],
            "抽搐": ["抽搐", "痉挛", "抽筋"],
            "癫痫发作": ["癫痫发作", "癫痫", "抽风"],
            
            # ========== 皮肤相关 ==========
            "皮疹": ["皮疹", "皮肤红疹", "红疹", "疹子", "皮肤疹", "出疹"],
            "瘙痒": ["瘙痒", "痒", "皮肤瘙痒", "皮痒", "发痒"],
            "红肿": ["红肿", "肿胀", "红胀"],
            "水肿": ["水肿", "浮肿", "肿胀"],
            "皮肤干燥": ["皮肤干燥", "皮肤粗糙", "皮肤脱屑", "脱皮", "皮肤起皮"],
            "皮肤发红": ["皮肤发红", "皮肤潮红", "红斑"],
            "皮肤苍白": ["皮肤苍白", "面色苍白", "苍白"],
            "皮肤发黄": ["皮肤发黄", "黄疸", "巩膜黄染"],
            
            # ========== 泌尿相关 ==========
            "尿频": ["尿频", "小便频繁", "排尿次数增多", "尿次增多"],
            "尿急": ["尿急", "急迫性尿意", "尿意急迫", "憋不住尿"],
            "血尿": ["血尿", "尿血", "小便带血", "尿液发红"],
            "蛋白尿": ["蛋白尿", "尿蛋白", "尿液泡沫"],
            "尿潴留": ["尿潴留", "排尿困难", "尿不出"],
            "夜尿增多": ["夜尿增多", "夜间多尿", "夜尿频繁"],
            
            # ========== 心血管相关 ==========
            "心悸": ["心悸", "心慌", "心跳加快", "心跳快"],
            "心律不齐": ["心律不齐", "心跳不规律", "心律失常"],
            "心动过速": ["心动过速", "心跳过快", "心率快"],
            "心动过缓": ["心动过缓", "心跳过慢", "心率慢"],
            
            # ========== 眼部相关 ==========
            "视力模糊": ["视力模糊", "视物模糊", "看东西模糊", "视力下降", "视物不清"],
            "复视": ["复视", "看东西重影", "双影"],
            "畏光": ["畏光", "怕光", "光敏感", "见光流泪"],
            "眼干": ["眼干", "眼睛干涩", "干眼"],
            "流泪": ["流泪", "泪水增多", "溢泪"],
            "眼红": ["眼红", "眼睛发红", "结膜充血"],
            
            # ========== 耳鼻喉相关 ==========
            "听力下降": ["听力下降", "听力减退", "耳聋"],
            "耳鸣": ["耳鸣", "耳内响声"],
            "耳溢液": ["耳溢液", "耳朵流水", "耳朵流脓"],
            "鼻塞": ["鼻塞", "鼻堵", "鼻子不通气", "鼻阻塞"],
            "流鼻涕": ["流鼻涕", "鼻涕", "流涕", "鼻分泌物"],
            "打喷嚏": ["打喷嚏", "喷嚏", "连续打喷嚏"],
            "嗅觉减退": ["嗅觉减退", "嗅觉丧失", "闻不到味道"],
            "声音嘶哑": ["声音嘶哑", "嗓子哑", "声嘶"],
            
            # ========== 其他常见症状 ==========
            "体重下降": ["体重下降", "体重减轻", "消瘦", "体重降低", "瘦了"],
            "体重增加": ["体重增加", "体重增长", "发胖", "长胖"],
            "出汗": ["出汗", "多汗", "汗多"],
            "盗汗": ["盗汗", "夜间出汗", "睡觉出汗"],
            "畏寒": ["畏寒", "怕冷", "寒冷"],
            "怕热": ["怕热", "畏热", "热感"],
            "口渴": ["口渴", "渴", "想喝水"],
            "口干": ["口干", "嘴干", "口腔干燥"],
            "多尿": ["多尿", "尿量增多", "小便多"],
            "少尿": ["少尿", "尿量减少", "小便少"],
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
        for symptom, count in symptom_counter.most_common(15):
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
    print("改进版症状ID标准化器 - 细化疼痛分类")
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
    
    # 初始化改进版标准化器
    normalizer = ImprovedSymptomNormalizer()
    
    # 标准化疾病数据
    updated_diseases, symptom_id_mapping = normalizer.normalize_disease_data(diseases_data)
    
    # 创建反向查询表
    reverse_mapping = normalizer.create_reverse_mapping()
    
    # 保存标准化后的疾病数据
    output_diseases_file = "improved_normalized_diseases_database.json"
    normalized_data = {
        "疾病库": updated_diseases,
        "症状ID映射": symptom_id_mapping,
        "统计信息": {
            "疾病总数": len(updated_diseases),
            "症状总数": len(symptom_id_mapping),
            "处理时间": "2024-10-02",
            "版本": "改进版 - 细化疼痛分类"
        }
    }
    
    print(f"正在保存标准化疾病数据: {output_diseases_file}")
    with open(output_diseases_file, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)
    
    # 保存反向查询表
    output_reverse_file = "improved_symptom_to_disease_mapping.json"
    reverse_data = {
        "症状反向查询表": reverse_mapping,
        "统计信息": {
            "症状总数": len(reverse_mapping),
            "平均每症状对应疾病数": sum(item['disease_count'] for item in reverse_mapping.values()) / len(reverse_mapping) if reverse_mapping else 0,
            "处理时间": "2024-10-02",
            "版本": "改进版 - 细化疼痛分类"
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
    
    # 显示疼痛相关症状的分类情况
    pain_symptoms = []
    for symptom_name in symptom_id_mapping.keys():
        if '痛' in symptom_name or '疼' in symptom_name:
            pain_symptoms.append(symptom_name)
    
    print(f"\n疼痛相关症状分类 (共 {len(pain_symptoms)} 种):")
    for i, pain_symptom in enumerate(sorted(pain_symptoms), 1):
        print(f"   {i:2d}. {pain_symptom}")
    
    print(f"\n输出文件:")
    print(f"   - 改进版疾病数据库: {output_diseases_file}")
    print(f"   - 改进版反向查询表: {output_reverse_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
