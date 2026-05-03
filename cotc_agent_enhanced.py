#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COTCAgent - 医疗时序数据分析思维链补全系统 (增强版)
实现真正的思维链补全逻辑：症状匹配 -> 概率计算 -> 验证追问 -> 诊断输出
"""

import json
import asyncio
import aiohttp
import logging
import tempfile
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
    handlers=[
        logging.FileHandler('cotc_agent.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SymptomIndicator:
    """症状指标"""
    name: str
    value: float
    unit: str
    time_series: List[datetime]
    trend: str  # 'increasing', 'decreasing', 'stable'

@dataclass
class DiseaseRisk:
    """疾病风险"""
    disease_id: str
    disease_name: str
    probability: float
    matched_symptoms: List[str]
    missing_symptoms: List[str]
    extra_symptoms: List[str]
    reasoning: str
    confidence_level: str  # 'high', 'medium', 'low'

@dataclass
class PatientProfile:
    """患者档案"""
    patient_id: str
    age: int
    gender: str
    medical_history: List[str]
    current_symptoms: List[str]
    lifestyle_factors: Dict[str, str]

class DiseaseMatcher:
    """疾病症状匹配器"""
    
    def __init__(self, disease_db_path: str):
        self.disease_db = self.load_disease_database(disease_db_path)
        self.symptom_to_diseases = self.build_symptom_mapping()
    
    def load_disease_database(self, db_path: str) -> Dict:
        """加载疾病数据库"""
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载疾病数据库失败: {e}")
            return {"疾病库": []}
    
    def build_symptom_mapping(self) -> Dict[str, List[str]]:
        """构建症状到疾病的映射"""
        symptom_mapping = {}
        for disease in self.disease_db.get("疾病库", []):
            disease_id = disease.get("疾病ID", "")
            disease_name = disease.get("疾病名称", "")
            for symptom in disease.get("症状列表", []):
                symptom_name = symptom.get("symptom_name", "")
                if symptom_name:
                    if symptom_name not in symptom_mapping:
                        symptom_mapping[symptom_name] = []
                    symptom_mapping[symptom_name].append({
                        "disease_id": disease_id,
                        "disease_name": disease_name
                    })
        return symptom_mapping
    
    def calculate_disease_probability(self, patient_symptoms: List[str], 
                                    max_diseases: int = 5) -> List[DiseaseRisk]:
        """计算疾病概率"""
        disease_scores = {}
        
        # 为每个症状找到相关疾病
        for symptom in patient_symptoms:
            if symptom in self.symptom_to_diseases:
                for disease_info in self.symptom_to_diseases[symptom]:
                    disease_id = disease_info["disease_id"]
                    if disease_id not in disease_scores:
                        disease_scores[disease_id] = {
                            "disease_name": disease_info["disease_name"],
                            "matched_symptoms": [],
                            "total_symptoms": 0,
                            "score": 0
                        }
                    disease_scores[disease_id]["matched_symptoms"].append(symptom)
                    disease_scores[disease_id]["score"] += 1
        
        # 获取每个疾病的完整症状列表
        for disease in self.disease_db.get("疾病库", []):
            disease_id = disease.get("疾病ID", "")
            if disease_id in disease_scores:
                all_symptoms = [s.get("symptom_name", "") for s in disease.get("症状列表", [])]
                disease_scores[disease_id]["total_symptoms"] = len(all_symptoms)
                disease_scores[disease_id]["all_symptoms"] = all_symptoms
        
        # 计算概率和缺失症状
        disease_risks = []
        for disease_id, info in disease_scores.items():
            matched_count = len(info["matched_symptoms"])
            total_count = info["total_symptoms"]
            
            if total_count > 0:
                probability = (matched_count / total_count) * 100
                missing_symptoms = [s for s in info["all_symptoms"] 
                                 if s not in info["matched_symptoms"]]
                extra_symptoms = [s for s in info["matched_symptoms"] 
                                if s not in info["all_symptoms"]]
                
                # 确定置信度
                if probability >= 80:
                    confidence = "high"
                elif probability >= 60:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                disease_risks.append(DiseaseRisk(
                    disease_id=disease_id,
                    disease_name=info["disease_name"],
                    probability=probability,
                    matched_symptoms=info["matched_symptoms"],
                    missing_symptoms=missing_symptoms,
                    extra_symptoms=extra_symptoms,
                    reasoning=f"匹配症状: {matched_count}/{total_count}",
                    confidence_level=confidence
                ))
        
        # 按概率排序，返回前N个
        disease_risks.sort(key=lambda x: x.probability, reverse=True)
        return disease_risks[:max_diseases]

class COTCAgent:
    """思维链补全Agent"""
    
    def __init__(self, disease_db_path: str):
        self.disease_matcher = DiseaseMatcher(disease_db_path)
        self.conversation_history = []
        self.current_analysis_step = 0
        self.max_verification_rounds = 5
        self.verification_round = 0
    
    def process_user_query(self, user_input: str, patient_profile: PatientProfile) -> Dict:
        """处理用户查询 - 实现思维链补全逻辑"""
        logger.info(f"开始思维链补全分析: {user_input}")
        
        # 步骤1: 初始症状分析
        initial_symptoms = self.extract_symptoms_from_input(user_input)
        patient_profile.current_symptoms.extend(initial_symptoms)
        
        # 步骤2: 计算疾病概率
        disease_risks = self.disease_matcher.calculate_disease_probability(
            patient_profile.current_symptoms
        )
        
        # 步骤3: 思维链补全验证
        verification_result = self.chain_of_thought_verification(
            disease_risks, patient_profile, user_input
        )
        
        # 步骤4: 生成最终诊断
        final_diagnosis = self.generate_final_diagnosis(verification_result)
        
        # 记录对话历史
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "analysis_step": self.current_analysis_step,
            "disease_risks": [self._serialize_disease_risk(dr) for dr in disease_risks],
            "verification_result": verification_result,
            "final_diagnosis": final_diagnosis
        })
        
        return {
            "status": "success",
            "analysis_step": self.current_analysis_step,
            "disease_risks": disease_risks,
            "verification_result": verification_result,
            "final_diagnosis": final_diagnosis,
            "next_questions": self.generate_follow_up_questions(verification_result)
        }
    
    def extract_symptoms_from_input(self, user_input: str) -> List[str]:
        """从用户输入中提取症状"""
        # 症状关键词匹配 - 更精确的匹配
        symptom_mapping = {
            '头疼': ['头痛', '头疼'],
            '头痛': ['头痛', '头疼'],
            '胸闷': ['胸闷', '胸痛'],
            '胸痛': ['胸闷', '胸痛'],
            '脚后跟疼': ['足跟痛', '脚跟痛', '足痛'],
            '脚疼': ['足痛', '脚痛'],
            '腹痛': ['腹痛', '肚子疼'],
            '肚子疼': ['腹痛', '肚子疼'],
            '发热': ['发热', '发烧'],
            '发烧': ['发热', '发烧'],
            '恶心': ['恶心', '想吐'],
            '呕吐': ['呕吐', '吐'],
            '头晕': ['头晕', '眩晕'],
            '失眠': ['失眠', '睡不着'],
            '咳嗽': ['咳嗽', '咳'],
            '呼吸困难': ['呼吸困难', '气短'],
            '心悸': ['心悸', '心慌'],
            '腹泻': ['腹泻', '拉肚子'],
            '便秘': ['便秘', '排便困难']
        }
        
        extracted_symptoms = []
        user_input_lower = user_input.lower()
        
        # 精确匹配用户描述的症状
        for user_symptom, medical_symptoms in symptom_mapping.items():
            if user_symptom in user_input:
                extracted_symptoms.extend(medical_symptoms)
        
        # 如果没有匹配到，尝试通用关键词
        if not extracted_symptoms:
            generic_keywords = ['疼痛', '痛', '疼', '发热', '恶心', '头晕', '失眠']
            for keyword in generic_keywords:
                if keyword in user_input:
                    extracted_symptoms.append(keyword)
        
        return list(set(extracted_symptoms))  # 去重
    
    def chain_of_thought_verification(self, disease_risks: List[DiseaseRisk], 
                                    patient_profile: PatientProfile, 
                                    user_input: str) -> Dict:
        """思维链补全验证过程"""
        self.verification_round += 1
        
        # 如果没有找到任何疾病，直接返回无匹配
        if not disease_risks:
            return {
                "selected_disease": None,
                "match_type": "no_match",
                "confidence": "none",
                "reasoning": "未找到匹配的疾病模式",
                "verification_questions": [
                    "您能更详细地描述一下症状吗？",
                    "这些症状持续多长时间了？",
                    "症状的严重程度如何？"
                ]
            }
        
        # 检查是否有症状匹配度足够高的疾病
        for i, disease_risk in enumerate(disease_risks):
            logger.info(f"验证疾病 {i+1}/{len(disease_risks)}: {disease_risk.disease_name}")
            
            # 检查症状匹配度
            match_analysis = self.analyze_symptom_match(disease_risk, patient_profile)
            
            # 只有当症状匹配度足够高时才考虑诊断
            if match_analysis["is_perfect_match"] and disease_risk.probability >= 70:
                return {
                    "selected_disease": disease_risk,
                    "match_type": "perfect",
                    "confidence": "high",
                    "reasoning": f"症状完全匹配，概率: {disease_risk.probability:.1f}%"
                }
            elif match_analysis["is_partial_match"] and disease_risk.probability >= 50:
                # 需要进一步验证
                verification_questions = self.generate_verification_questions(
                    disease_risk, match_analysis
                )
                return {
                    "selected_disease": disease_risk,
                    "match_type": "partial",
                    "confidence": "medium",
                    "reasoning": f"部分症状匹配，需要进一步确认",
                    "verification_questions": verification_questions,
                    "missing_symptoms": match_analysis["missing_symptoms"],
                    "lifestyle_factors": match_analysis["lifestyle_factors"]
                }
        
        # 如果所有疾病匹配度都不够，返回需要更多信息
        return {
            "selected_disease": None,
            "match_type": "insufficient_info",
            "confidence": "low",
            "reasoning": "基于现有症状信息，无法确定具体疾病",
            "verification_questions": [
                "您能更详细地描述一下症状吗？",
                "这些症状是什么时候开始的？",
                "症状的严重程度如何？",
                "您最近的生活习惯有什么变化吗？"
            ],
            "suggested_symptoms": [
                "是否发热？",
                "是否有恶心呕吐？",
                "是否有其他不适？"
            ]
        }
    
    def analyze_symptom_match(self, disease_risk: DiseaseRisk, 
                            patient_profile: PatientProfile) -> Dict:
        """分析症状匹配度"""
        matched_symptoms = disease_risk.matched_symptoms
        missing_symptoms = disease_risk.missing_symptoms
        patient_symptoms = patient_profile.current_symptoms
        
        # 计算匹配度
        match_ratio = len(matched_symptoms) / len(patient_symptoms) if patient_symptoms else 0
        
        # 判断匹配类型
        is_perfect_match = (
            len(missing_symptoms) == 0 and 
            len(disease_risk.extra_symptoms) == 0 and 
            match_ratio >= 0.8
        )
        
        is_partial_match = (
            match_ratio >= 0.5 and 
            len(missing_symptoms) <= 3
        )
        
        # 分析生活方式因素
        lifestyle_factors = self.analyze_lifestyle_factors(disease_risk, patient_profile)
        
        return {
            "is_perfect_match": is_perfect_match,
            "is_partial_match": is_partial_match,
            "match_ratio": match_ratio,
            "missing_symptoms": missing_symptoms,
            "lifestyle_factors": lifestyle_factors
        }
    
    def analyze_lifestyle_factors(self, disease_risk: DiseaseRisk, 
                               patient_profile: PatientProfile) -> List[str]:
        """分析生活方式因素"""
        lifestyle_factors = []
        
        # 根据疾病类型分析可能的生活方式因素
        disease_name = disease_risk.disease_name.lower()
        
        if "肠胃" in disease_name or "消化" in disease_name:
            lifestyle_factors.extend([
                "饮食习惯变化",
                "压力水平",
                "作息规律",
                "药物使用"
            ])
        elif "头痛" in disease_name or "偏头痛" in disease_name:
            lifestyle_factors.extend([
                "睡眠质量",
                "工作压力",
                "用眼习惯",
                "环境因素"
            ])
        elif "失眠" in disease_name or "睡眠" in disease_name:
            lifestyle_factors.extend([
                "睡前习惯",
                "环境噪音",
                "心理状态",
                "药物影响"
            ])
        
        return lifestyle_factors
    
    def generate_verification_questions(self, disease_risk: DiseaseRisk, 
                                      match_analysis: Dict) -> List[str]:
        """生成验证问题"""
        questions = []
        
        # 关于缺失症状的问题
        for symptom in match_analysis["missing_symptoms"][:2]:  # 最多问2个
            questions.append(f"您是否还出现了{symptom}的症状？")
        
        # 关于生活方式因素的问题
        for factor in match_analysis["lifestyle_factors"][:2]:  # 最多问2个
            if factor == "饮食习惯变化":
                questions.append("您最近的饮食习惯有什么变化吗？")
            elif factor == "压力水平":
                questions.append("您最近的工作或生活压力大吗？")
            elif factor == "睡眠质量":
                questions.append("您的睡眠质量如何？")
            elif factor == "工作压力":
                questions.append("您的工作强度如何？")
        
        return questions[:3]  # 最多3个问题
    
    def generate_final_diagnosis(self, verification_result: Dict) -> Dict:
        """生成最终诊断"""
        match_type = verification_result["match_type"]
        
        if match_type == "no_match" or match_type == "insufficient_info":
            return {
                "diagnosis": "基于您描述的症状，我需要更多信息才能进行准确分析",
                "confidence": "low",
                "recommendation": "请回答以下问题，帮助我更好地了解您的症状",
                "next_questions": verification_result.get("verification_questions", []),
                "suggested_symptoms": verification_result.get("suggested_symptoms", [])
            }
        
        if not verification_result["selected_disease"]:
            return {
                "diagnosis": "未找到明确匹配的疾病",
                "recommendation": "建议咨询专业医生进行详细检查",
                "confidence": "none"
            }
        
        disease = verification_result["selected_disease"]
        
        if match_type == "perfect":
            return {
                "diagnosis": f"基于症状分析，您可能有{disease.probability:.1f}%的概率患有{disease.disease_name}",
                "confidence": "high",
                "recommendation": "建议尽快就医确诊并接受治疗",
                "matched_symptoms": disease.matched_symptoms
            }
        elif match_type == "partial":
            return {
                "diagnosis": f"初步分析显示您可能患有{disease.disease_name}，但需要进一步确认",
                "confidence": "medium",
                "recommendation": "建议回答更多问题以确认诊断，或咨询专业医生",
                "next_steps": "需要进一步症状确认",
                "verification_questions": verification_result.get("verification_questions", [])
            }
        else:  # uncertain
            return {
                "diagnosis": f"基于现有信息，{disease.disease_name}的可能性最高，但存在疑点",
                "confidence": "low",
                "recommendation": "建议进行更详细的医学检查以排除其他可能",
                "concerns": verification_result.get("concerns", "症状匹配度不够理想")
            }
    
    def generate_follow_up_questions(self, verification_result: Dict) -> List[str]:
        """生成后续问题"""
        if verification_result.get("verification_questions"):
            return verification_result["verification_questions"]
        
        # 默认后续问题
        return [
            "您还有其他症状需要补充吗？",
            "这些症状持续多长时间了？",
            "症状的严重程度如何？"
        ]
    
    def _serialize_disease_risk(self, disease_risk: DiseaseRisk) -> Dict:
        """序列化疾病风险对象"""
        return {
            "disease_id": disease_risk.disease_id,
            "disease_name": disease_risk.disease_name,
            "probability": disease_risk.probability,
            "matched_symptoms": disease_risk.matched_symptoms,
            "missing_symptoms": disease_risk.missing_symptoms,
            "extra_symptoms": disease_risk.extra_symptoms,
            "reasoning": disease_risk.reasoning,
            "confidence_level": disease_risk.confidence_level
        }

def main():
    """主函数 - 演示增强版COTCAgent"""
    print("COTCAgent 增强版 - 思维链补全医疗诊断系统")
    print("=" * 50)
    
    # 初始化Agent
    agent = COTCAgent("disease_symptom_database.json")
    
    # 创建患者档案
    patient = PatientProfile(
        patient_id="patient_0001",
        age=35,
        gender="女",
        medical_history=["无重大疾病史"],
        current_symptoms=[],
        lifestyle_factors={}
    )
    
    # 模拟用户输入
    user_input = "我现在头疼，然后胸闷，然后走路脚后跟疼得很"
    
    print(f"患者输入: {user_input}")
    print("\n开始思维链补全分析...")
    
    # 处理查询
    result = agent.process_user_query(user_input, patient)
    
    # 显示结果
    print("\n" + "="*50)
    print("思维链补全分析结果")
    print("="*50)
    
    print(f"\n分析步骤: {result['analysis_step']}")
    print(f"匹配类型: {result['verification_result']['match_type']}")
    print(f"置信度: {result['verification_result']['confidence']}")
    
    if result['disease_risks']:
        print(f"\n疾病风险评估 (前3个):")
        for i, risk in enumerate(result['disease_risks'][:3]):
            print(f"{i+1}. {risk.disease_name}: {risk.probability:.1f}%")
            print(f"   匹配症状: {', '.join(risk.matched_symptoms)}")
            print(f"   缺失症状: {', '.join(risk.missing_symptoms)}")
            print(f"   置信度: {risk.confidence_level}")
            print()
    
    print(f"\n最终诊断:")
    diagnosis = result['final_diagnosis']
    print(f"诊断: {diagnosis['diagnosis']}")
    print(f"置信度: {diagnosis['confidence']}")
    print(f"建议: {diagnosis['recommendation']}")
    
    if result['next_questions']:
        print(f"\n后续问题:")
        for i, question in enumerate(result['next_questions'], 1):
            print(f"{i}. {question}")

if __name__ == "__main__":
    main()
