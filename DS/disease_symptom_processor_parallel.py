import json
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import queue
from datetime import datetime

class ParallelDiseaseSymptomProcessor:
    def __init__(self, api_key: str, api_base: str, max_workers: int = 5):
        self.api_key = api_key
        self.api_base = api_base
        self.max_workers = max_workers
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # 用于控制API调用频率的锁
        self.rate_limit_lock = threading.Lock()
        self.last_call_time = 0
        self.min_interval = 0.2  # 最小间�?00ms
    
    def call_deepseek_api(self, disease_data: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
        """
        调用DeepSeek API来验证和补全疾病症状信息，支持重试机�?
        """
        disease_name = disease_data['疾病名称']
        
        for attempt in range(max_retries):
            try:
                # 控制API调用频率
                with self.rate_limit_lock:
                    current_time = time.time()
                    time_since_last_call = current_time - self.last_call_time
                    if time_since_last_call < self.min_interval:
                        time.sleep(self.min_interval - time_since_last_call)
                    self.last_call_time = time.time()
                
                prompt = self.create_prompt(disease_data)
                
                payload = {
                    "model": "DeepSeek-V3",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000
                }
                
                # 根据重试次数调整超时时间
                timeout = 30 + (attempt * 10)  # 第一�?0秒，第二�?0秒，以此类推
                
                response = requests.post(self.api_base, headers=self.headers, json=payload, timeout=timeout)
                response.raise_for_status()
                
                result = response.json()
                return result['choices'][0]['message']['content']
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 递增等待时间�?秒，4秒，6秒，8�?
                    print(f"API调用失败 (疾病: {disease_name}, 第{attempt + 1}次尝�?: {e}")
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"API调用最终失�?(疾病: {disease_name}, 已重试{max_retries}�?: {e}")
                    return None
    
    def create_prompt(self, disease_data: Dict[str, Any]) -> str:
        """
        创建用于DeepSeek的提示词
        """
        disease_name = disease_data['疾病名称']
        current_symptoms = disease_data['症状列表']
        
        symptoms_text = ""
        for symptom in current_symptoms:
            symptoms_text += f"- {symptom['symptom_name']} (ID: {symptom['symptom_id']})\n"
        
        prompt = f"""
你是一位专业的医学专家。请分析以下疾病及其症状信息，并进行验证和补全：

疾病名称：{disease_name}
当前症状列表�?
{symptoms_text}

请完成以下任务：

1. 验证当前症状是否与疾病准确对�?
2. 如果症状不准确，请修正症状名称（保持症状ID不变�?
3. 补全该疾病的主要症状，确保症状描述简洁准�?
4. 为每个症状添�?specificity"字段，判断该症状是否具有特异性（即出现该症状是否强烈提示该疾病）

请以JSON格式返回结果，格式如下：
{{
    "疾病ID": "{disease_data['疾病ID']}",
    "疾病名称": "{disease_name}",
    "症状列表": [
        {{
            "symptom_id": "症状ID",
            "symptom_name": "症状名称",
            "disease_id": "{disease_data['疾病ID']}",
            "specificity": true/false
        }}
    ],
    "疾病解释": "疾病的详细医学描�?
}}

要求�?
- 症状描述要简洁准确，不要过于复杂
- specificity判断要准确：true表示该症状具有特异性，false表示该症状不具有特异�?
- 补全的症状要全面，包括该疾病的主要症�?
- 保持原有的症状ID不变
- 只返回JSON，不要其他文字说�?
"""
        return prompt
    
    def parse_deepseek_response(self, response: str) -> Dict[str, Any]:
        """
        解析DeepSeek的响应，提取JSON数据
        """
        try:
            # 尝试直接解析JSON
            if response.strip().startswith('{'):
                return json.loads(response)
            
            # 如果响应包含其他文本，尝试提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return None
        except Exception as e:
            print(f"解析响应失败: {e}")
            return None
    
    def process_single_disease(self, disease_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个疾病
        """
        disease_name = disease_data['疾病名称']
        print(f"开始处�? {disease_name}")
        
        # 调用DeepSeek API
        response = self.call_deepseek_api(disease_data)
        
        if response:
            # 解析响应
            processed_disease = self.parse_deepseek_response(response)
            
            if processed_disease:
                print(f"成功处理: {disease_name}")
                return processed_disease
            else:
                print(f"解析失败: {disease_name}")
                return disease_data
        else:
            print(f"API调用失败: {disease_name}")
            return disease_data
    
    def load_existing_results(self, output_file: str) -> List[Dict[str, Any]]:
        """
        加载已存在的处理结果
        """
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('疾病�?, [])
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"加载已有结果失败: {e}")
            return []
    
    def save_results(self, processed_diseases: List[Dict[str, Any]], output_file: str):
        """
        保存处理结果
        """
        output_data = {
            "疾病�?: processed_diseases
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    def get_processed_disease_ids(self, processed_diseases: List[Dict[str, Any]]) -> set:
        """
        获取已处理的疾病ID集合
        """
        return {disease['疾病ID'] for disease in processed_diseases}
    
    def get_all_processed_disease_ids(self) -> set:
        """
        从所有批次文件中获取已处理的疾病ID集合
        """
        import glob
        processed_ids = set()
        
        # 查找所有批次文�?
        batch_files = glob.glob("batch_*_processed_diseases.json")
        
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    batch_diseases = data.get('疾病�?, [])
                    for disease in batch_diseases:
                        processed_ids.add(disease['疾病ID'])
            except Exception as e:
                print(f"读取批次文件失败 {batch_file}: {e}")
        
        return processed_ids
    
    def process_diseases_parallel(self, input_file: str, output_file: str, batch_size: int = 50):
        """
        分批并行处理疾病数据，每批次单独保存文件
        """
        # 读取原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_diseases = data['疾病�?]
        
        # 获取已处理的疾病ID（从所有批次文件中�?
        processed_ids = self.get_all_processed_disease_ids()
        
        # 过滤出未处理的疾�?
        remaining_diseases = [d for d in all_diseases if d['疾病ID'] not in processed_ids]
        
        if not remaining_diseases:
            print("所有疾病都已处理完成！")
            return
        
        print(f"总疾病数: {len(all_diseases)}")
        print(f"已处�? {len(processed_ids)}")
        print(f"剩余待处�? {len(remaining_diseases)}")
        print(f"使用 {self.max_workers} 个线�?)
        print(f"开始时�? {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 分批处理
        total_processed = len(processed_ids)
        start_time = time.time()
        
        for i in range(0, len(remaining_diseases), batch_size):
            batch_diseases = remaining_diseases[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(remaining_diseases) + batch_size - 1) // batch_size
            
            print(f"\n处理�?{batch_num}/{total_batches} �?({len(batch_diseases)} 个疾�?...")
            
            # 处理当前批次
            batch_results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_disease = {
                    executor.submit(self.process_single_disease, disease): disease 
                    for disease in batch_diseases
                }
                
                for future in as_completed(future_to_disease):
                    try:
                        result = future.result()
                        batch_results.append(result)
                        total_processed += 1
                        
                        print(f"批次进度: {len(batch_results)}/{len(batch_diseases)} "
                              f"总进�? {total_processed}/{len(all_diseases)}")
                        
                    except Exception as e:
                        disease = future_to_disease[future]
                        print(f"处理失败: {disease['疾病名称']} - {e}")
                        batch_results.append(disease)
            
            # 保存当前批次到单独文�?
            batch_output_file = f"batch_{batch_num:03d}_processed_diseases.json"
            self.save_results(batch_results, batch_output_file)
            
            print(f"�?{batch_num} 批处理完成，已保存到: {batch_output_file}")
            
            # 显示统计信息
            elapsed_time = time.time() - start_time
            avg_time_per_disease = elapsed_time / total_processed if total_processed > 0 else 0
            remaining_diseases_count = len(all_diseases) - total_processed
            estimated_remaining_time = remaining_diseases_count * avg_time_per_disease
            
            print(f"已用�? {elapsed_time/60:.1f}分钟")
            print(f"平均每个疾病: {avg_time_per_disease:.2f}�?)
            print(f"预计剩余: {estimated_remaining_time/60:.1f}分钟")
            
            # 如果不是最后一批，等待一下避免API限制
            if i + batch_size < len(remaining_diseases):
                print("等待5秒后继续下一�?..")
                time.sleep(5)
        
        total_time = time.time() - start_time
        print(f"\n所有批次处理完成！")
        print(f"总用�? {total_time/60:.1f}分钟")
        print(f"平均每个疾病: {total_time/len(all_diseases):.2f}�?)
        print(f"各批次文件已保存，请运行合并脚本生成最终结�?)

def main():
    # 配置API
    api_key = 'xxx'
    api_base = "https://api.deepseek.com/v1/chat/completions"
    
    # 创建并行处理�?(可以根据需要调整线程数)
    max_workers = 5  # 建议5-10个线程，避免API限制
    processor = ParallelDiseaseSymptomProcessor(api_key, api_base, max_workers)
    
    # 处理数据
    input_file = "disease_symptom_database.json"
    output_file = "processed_disease_symptom_database_parallel.json"
    
    # 分批处理，每�?0个疾�?
    batch_size = 50
    processor.process_diseases_parallel(input_file, output_file, batch_size)

if __name__ == "__main__":
    main()
