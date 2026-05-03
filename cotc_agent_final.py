#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COTCAgent - 医疗时序数据分析思维链补全系统 (最终版)
实现真正的思维链补全：100%匹配才诊断，可视化匹配过程，通过询问增加匹配率
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
class SymptomMatch:
    """症状匹配状态"""
    symptom_name: str
    is_confirmed: bool
    confirmation_source: str  # 'user_input', 'user_confirmation', 'lifestyle_factor'
    match_confidence: float  # 0.0-1.0

@dataclass
class DiseaseMatchProgress:
    """疾病匹配进度"""
    disease_id: str
    disease_name: str
    total_symptoms: int
    confirmed_symptoms: int
    match_percentage: float
    missing_symptoms: List[str]
    lifestyle_factors: List[str]
    current_questions: List[str]

@dataclass
class PatientProfile:
    """患者档案"""
    patient_id: str
    age: int
    gender: str
    medical_history: List[str]
    confirmed_symptoms: List[SymptomMatch]
    lifestyle_factors: Dict[str, str]
    conversation_history: List[Dict]

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
    
    def get_disease_symptoms(self, disease_id: str) -> List[str]:
        """获取疾病的完整症状列表"""
        for disease in self.disease_db.get("疾病库", []):
            if disease.get("疾病ID") == disease_id:
                return [s.get("symptom_name", "") for s in disease.get("症状列表", [])]
        return []
    
    def find_potential_diseases(self, confirmed_symptoms: List[SymptomMatch]) -> List[DiseaseMatchProgress]:
        """找到潜在疾病并计算匹配进度"""
        disease_scores = {}
        
        # 统计每个疾病的症状匹配情况
        for symptom_match in confirmed_symptoms:
            if symptom_match.is_confirmed and symptom_match.symptom_name in self.symptom_to_diseases:
                for disease_info in self.symptom_to_diseases[symptom_match.symptom_name]:
                    disease_id = disease_info["disease_id"]
                    if disease_id not in disease_scores:
                        disease_scores[disease_id] = {
                            "disease_name": disease_info["disease_name"],
                            "confirmed_symptoms": [],
                            "total_symptoms": 0
                        }
                    disease_scores[disease_id]["confirmed_symptoms"].append(symptom_match.symptom_name)
        
        # 获取每个疾病的完整症状列表
        for disease_id in disease_scores:
            all_symptoms = self.get_disease_symptoms(disease_id)
            disease_scores[disease_id]["total_symptoms"] = len(all_symptoms)
            disease_scores[disease_id]["all_symptoms"] = all_symptoms
        
        # 计算匹配进度
        disease_progress = []
        for disease_id, info in disease_scores.items():
            confirmed_count = len(info["confirmed_symptoms"])
            total_count = info["total_symptoms"]
            match_percentage = (confirmed_count / total_count * 100) if total_count > 0 else 0
            
            missing_symptoms = [s for s in info["all_symptoms"] 
                              if s not in info["confirmed_symptoms"]]
            
            disease_progress.append(DiseaseMatchProgress(
                disease_id=disease_id,
                disease_name=info["disease_name"],
                total_symptoms=total_count,
                confirmed_symptoms=confirmed_count,
                match_percentage=match_percentage,
                missing_symptoms=missing_symptoms,
                lifestyle_factors=[],
                current_questions=[]
            ))
        
        # 按匹配度排序
        disease_progress.sort(key=lambda x: x.match_percentage, reverse=True)
        return disease_progress

class COTCAgent:
    """思维链补全Agent"""
    
    def __init__(self, disease_db_path: str):
        self.disease_matcher = DiseaseMatcher(disease_db_path)
        self.current_analysis_step = 0
        self.max_verification_rounds = 5
        self.verification_round = 0
        self.current_disease_focus = None  # 当前重点分析的疾病
    
    def process_user_query(self, user_input: str, patient_profile: PatientProfile) -> Dict:
        """处理用户查询 - 实现真正的思维链补全逻辑"""
        logger.info(f"开始思维链补全分析: {user_input}")
        
        # 步骤1: 提取和确认症状
        new_symptoms = self.extract_and_confirm_symptoms(user_input, patient_profile)
        
        # 步骤2: 更新患者档案
        patient_profile.confirmed_symptoms.extend(new_symptoms)
        
        # 步骤3: 计算疾病匹配进度
        disease_progress = self.disease_matcher.find_potential_diseases(patient_profile.confirmed_symptoms)
        
        print(f"找到 {len(disease_progress)} 个潜在疾病")
        
        # 步骤4: 检查是否有100%匹配的疾病
        perfect_matches = [d for d in disease_progress if d.match_percentage >= 100.0]
        
        if perfect_matches:
            # 找到100%匹配的疾病，可以给出诊断
            return self.generate_final_diagnosis(perfect_matches[0], patient_profile)
        else:
            # 没有100%匹配，需要继续询问
            return self.generate_verification_questions(disease_progress, patient_profile)
    
    def extract_and_confirm_symptoms(self, user_input: str, patient_profile: PatientProfile) -> List[SymptomMatch]:
        """提取并确认症状 - 更严格的匹配逻辑"""
        # 症状关键词映射 - 更精确的匹配
        symptom_mapping = {
            '头疼': ['头痛'],
            '头痛': ['头痛'],
            '胸闷': ['胸闷'],
            '胸痛': ['胸痛'],
            '脚后跟疼': ['足跟痛'],
            '脚疼': ['足痛'],
            '腹痛': ['腹痛'],
            '肚子疼': ['腹痛'],
            '发热': ['发热'],
            '发烧': ['发热'],
            '恶心': ['恶心'],
            '呕吐': ['呕吐'],
            '头晕': ['头晕'],
            '失眠': ['失眠'],
            '咳嗽': ['咳嗽'],
            '呼吸困难': ['呼吸困难'],
            '心悸': ['心悸'],
            '腹泻': ['腹泻'],
            '便秘': ['便秘']
        }
        
        confirmed_symptoms = []
        
        # 从用户输入中提取症状 - 只提取明确提到的症状
        for user_symptom, medical_symptoms in symptom_mapping.items():
            if user_symptom in user_input:
                for medical_symptom in medical_symptoms:
                    # 检查是否已经确认过这个症状
                    if not any(s.symptom_name == medical_symptom for s in patient_profile.confirmed_symptoms):
                        confirmed_symptoms.append(SymptomMatch(
                            symptom_name=medical_symptom,
                            is_confirmed=True,
                            confirmation_source='user_input',
                            match_confidence=1.0
                        ))
        
        return confirmed_symptoms
    
    def generate_verification_questions(self, disease_progress: List[DiseaseMatchProgress], 
                                     patient_profile: PatientProfile) -> Dict:
        """生成验证问题 - 真正的思维链补全"""
        # 选择匹配度最高的疾病作为重点分析对象
        if disease_progress:
            focus_disease = disease_progress[0]
            self.current_disease_focus = focus_disease
            
            # 检查是否已经达到100%匹配
            if focus_disease.match_percentage >= 100.0:
                return self.generate_final_diagnosis(focus_disease, patient_profile)
            
            # 生成针对性的问题
            questions = self.generate_targeted_questions(focus_disease, patient_profile)
            
            return {
                "status": "verification_needed",
                "analysis_step": self.current_analysis_step,
                "focus_disease": {
                    "disease_id": focus_disease.disease_id,
                    "disease_name": focus_disease.disease_name,
                    "match_percentage": focus_disease.match_percentage,
                    "confirmed_symptoms": [s.symptom_name for s in patient_profile.confirmed_symptoms if s.is_confirmed],
                    "missing_symptoms": focus_disease.missing_symptoms[:3],  # 最多显示3个
                    "total_symptoms": focus_disease.total_symptoms
                },
                "verification_questions": questions,
                "match_visualization": self.generate_match_visualization(focus_disease, patient_profile)
            }
        else:
            return {
                "status": "no_matches",
                "analysis_step": self.current_analysis_step,
                "message": "未找到匹配的疾病模式",
                "suggested_questions": [
                    "您能更详细地描述一下症状吗？",
                    "这些症状是什么时候开始的？",
                    "症状的严重程度如何？"
                ]
            }
    
    def generate_targeted_questions(self, focus_disease: DiseaseMatchProgress, 
                                  patient_profile: PatientProfile) -> List[str]:
        """生成针对性问题 - 更精确的追问"""
        questions = []
        
        # 关于缺失症状的问题 - 只问最重要的缺失症状
        important_missing = focus_disease.missing_symptoms[:1]  # 只问1个最重要的
        for symptom in important_missing:
            questions.append(f"您是否还出现了{symptom}的症状？")
        
        # 关于生活方式的问题 - 根据疾病类型选择
        if "肠胃" in focus_disease.disease_name or "消化" in focus_disease.disease_name:
            questions.append("您最近的饮食习惯有什么变化吗？")
        elif "头痛" in focus_disease.disease_name or "偏头痛" in focus_disease.disease_name:
            questions.append("您最近的工作压力大吗？")
        elif "失眠" in focus_disease.disease_name or "睡眠" in focus_disease.disease_name:
            questions.append("您的睡眠质量如何？")
        else:
            questions.append("您最近的生活习惯有什么变化吗？")
        
        return questions[:2]  # 最多2个问题
    
    def generate_match_visualization(self, focus_disease: DiseaseMatchProgress, 
                                   patient_profile: PatientProfile) -> Dict:
        """生成匹配可视化信息"""
        confirmed_symptoms = [s.symptom_name for s in patient_profile.confirmed_symptoms if s.is_confirmed]
        
        return {
            "disease_name": focus_disease.disease_name,
            "progress_bar": {
                "current": focus_disease.confirmed_symptoms,
                "total": focus_disease.total_symptoms,
                "percentage": focus_disease.match_percentage
            },
            "symptom_status": {
                "confirmed": confirmed_symptoms,
                "missing": focus_disease.missing_symptoms[:3],
                "total_needed": focus_disease.total_symptoms
            },
            "next_steps": f"需要确认 {focus_disease.total_symptoms - focus_disease.confirmed_symptoms} 个症状才能完成诊断"
        }
    
    def process_user_response(self, user_response: str, patient_profile: PatientProfile) -> Dict:
        """处理用户回答"""
        # 分析用户回答，提取新的症状确认
        new_confirmations = self.analyze_user_response(user_response, patient_profile)
        
        # 更新患者档案
        patient_profile.confirmed_symptoms.extend(new_confirmations)
        
        # 重新计算疾病匹配进度
        disease_progress = self.disease_matcher.find_potential_diseases(patient_profile.confirmed_symptoms)
        
        # 检查是否有100%匹配
        perfect_matches = [d for d in disease_progress if d.match_percentage >= 100.0]
        
        if perfect_matches:
            return self.generate_final_diagnosis(perfect_matches[0], patient_profile)
        else:
            return self.generate_verification_questions(disease_progress, patient_profile)
    
    def analyze_user_response(self, user_response: str, patient_profile: PatientProfile) -> List[SymptomMatch]:
        """分析用户回答，提取新的症状确认 - 更严格的确认逻辑"""
        new_confirmations = []
        
        # 检查用户是否确认了缺失的症状 - 更严格的确认
        if self.current_disease_focus:
            for missing_symptom in self.current_disease_focus.missing_symptoms:
                # 只有当用户明确确认时才添加症状
                if (missing_symptom in user_response and 
                    ("是" in user_response or "有" in user_response or "确实" in user_response)):
                    new_confirmations.append(SymptomMatch(
                        symptom_name=missing_symptom,
                        is_confirmed=True,
                        confirmation_source='user_confirmation',
                        match_confidence=1.0
                    ))
        
        # 检查生活方式因素
        lifestyle_factors = {
            "压力": ["压力", "紧张", "焦虑", "工作"],
            "饮食": ["饮食", "吃", "食物", "油腻"],
            "睡眠": ["睡眠", "睡觉", "失眠", "休息"],
            "运动": ["运动", "锻炼", "活动", "走路"]
        }
        
        for factor, keywords in lifestyle_factors.items():
            if any(keyword in user_response for keyword in keywords):
                patient_profile.lifestyle_factors[factor] = user_response
        
        return new_confirmations
    
    def generate_final_diagnosis(self, disease: DiseaseMatchProgress, patient_profile: PatientProfile) -> Dict:
        """生成最终诊断"""
        confirmed_symptoms = [s.symptom_name for s in patient_profile.confirmed_symptoms if s.is_confirmed]
        
        return {
            "status": "diagnosis_complete",
            "analysis_step": self.current_analysis_step,
            "diagnosis": {
                "disease_name": disease.disease_name,
                "confidence": "high",
                "match_percentage": disease.match_percentage,
                "confirmed_symptoms": confirmed_symptoms,
                "reasoning": f"基于症状分析，您患有{disease.disease_name}的概率很高"
            },
            "recommendations": [
                "建议尽快就医确诊并接受治疗",
                "密切观察症状变化",
                "如有症状加重，及时就医"
            ],
            "lifestyle_advice": self.generate_lifestyle_advice(patient_profile.lifestyle_factors)
        }
    
    def generate_lifestyle_advice(self, lifestyle_factors: Dict[str, str]) -> List[str]:
        """生成生活方式建议"""
        advice = []
        
        if "压力" in lifestyle_factors:
            advice.append("建议适当减压，保持心情愉快")
        if "饮食" in lifestyle_factors:
            advice.append("注意饮食规律，避免刺激性食物")
        if "睡眠" in lifestyle_factors:
            advice.append("保持规律作息，确保充足睡眠")
        if "运动" in lifestyle_factors:
            advice.append("适当运动，增强体质")
        
        return advice

def main():
    """主函数 - 演示最终版COTCAgent"""
    print("COTCAgent 最终版 - 真正的思维链补全医疗诊断系统")
    print("=" * 60)
    
    # 初始化Agent
    agent = COTCAgent("disease_symptom_database.json")
    
    # 创建患者档案
    patient = PatientProfile(
        patient_id="patient_0001",
        age=35,
        gender="女",
        medical_history=["无重大疾病史"],
        confirmed_symptoms=[],
        lifestyle_factors={},
        conversation_history=[]
    )
    
    # 模拟用户输入
    user_input = "我现在头疼，然后胸闷，然后走路脚后跟疼得很"
    
    print(f"患者输入: {user_input}")
    print("\n开始思维链补全分析...")
    
    # 第一轮分析
    result = agent.process_user_query(user_input, patient)
    
    print("\n" + "="*60)
    print("思维链补全分析结果")
    print("="*60)
    
    if result["status"] == "verification_needed":
        print(f"\n当前分析疾病: {result['focus_disease']['disease_name']}")
        print(f"匹配进度: {result['focus_disease']['match_percentage']:.1f}%")
        print(f"已确认症状: {', '.join(result['focus_disease']['confirmed_symptoms'])}")
        print(f"缺失症状: {', '.join(result['focus_disease']['missing_symptoms'])}")
        print(f"总症状数: {result['focus_disease']['total_symptoms']}")
        
        print(f"\n需要进一步确认的问题:")
        for i, question in enumerate(result['verification_questions'], 1):
            print(f"{i}. {question}")
        
        # 模拟多轮对话
        print(f"\n=== 第一轮追问 ===")
        user_response1 = "是的，我最近工作压力很大"
        print(f"患者回答: {user_response1}")
        
        result2 = agent.process_user_response(user_response1, patient)
        
        if result2["status"] == "diagnosis_complete":
            print(f"\n[SUCCESS] 诊断完成!")
            print(f"诊断结果: {result2['diagnosis']['disease_name']}")
            print(f"置信度: {result2['diagnosis']['confidence']}")
            print(f"匹配度: {result2['diagnosis']['match_percentage']:.1f}%")
            print(f"确认症状: {', '.join(result2['diagnosis']['confirmed_symptoms'])}")
        elif result2["status"] == "verification_needed":
            print(f"\n当前匹配度: {result2['focus_disease']['match_percentage']:.1f}%")
            print(f"还需要确认的症状: {', '.join(result2['focus_disease']['missing_symptoms'])}")
            
            print(f"\n=== 第二轮追问 ===")
            user_response2 = "是的，我确实有失眠的症状"
            print(f"患者回答: {user_response2}")
            
            result3 = agent.process_user_response(user_response2, patient)
            
            if result3["status"] == "diagnosis_complete":
                print(f"\n[SUCCESS] 诊断完成!")
                print(f"诊断结果: {result3['diagnosis']['disease_name']}")
                print(f"置信度: {result3['diagnosis']['confidence']}")
                print(f"匹配度: {result3['diagnosis']['match_percentage']:.1f}%")
                print(f"确认症状: {', '.join(result3['diagnosis']['confirmed_symptoms'])}")
            else:
                print(f"\n需要继续询问...")
                print(f"当前匹配度: {result3['focus_disease']['match_percentage']:.1f}%")
    
    elif result["status"] == "no_matches":
        print(f"\n未找到匹配的疾病模式")
        print(f"建议问题:")
        for question in result['suggested_questions']:
            print(f"- {question}")
    
    elif result["status"] == "diagnosis_complete":
        print(f"\n[SUCCESS] 诊断完成!")
        print(f"诊断结果: {result['diagnosis']['disease_name']}")
        print(f"置信度: {result['diagnosis']['confidence']}")
        print(f"匹配度: {result['diagnosis']['match_percentage']:.1f}%")
        print(f"确认症状: {', '.join(result['diagnosis']['confirmed_symptoms'])}")
        
        print(f"\n医疗建议:")
        for rec in result['recommendations']:
            print(f"- {rec}")
        
        if result.get('lifestyle_advice'):
            print(f"\n生活方式建议:")
            for advice in result['lifestyle_advice']:
                print(f"- {advice}")
    
    else:
        print(f"\n未知状态: {result.get('status', 'unknown')}")
        print(f"结果: {result}")

if __name__ == "__main__":
    main()
