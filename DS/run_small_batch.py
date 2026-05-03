#!/usr/bin/env python3
"""
运行小批次处理，每批10个疾病
"""

from disease_symptom_processor_parallel import ParallelDiseaseSymptomProcessor

def main():
    # 配置API - 华为云ModelArts
    api_key = 'WEBaX4AAhPUQyTHbQIYhWq8vURSJnU8WNFN1zpbrCZ1qFl87XTpH9M-UGQJkO_UHfGIXqaUJ3CqgI3M2DBuSng'
    api_base = "https://api.modelarts-maas.com/v1/chat/completions"
    
    # 创建并行处理器
    max_workers = 3  # 使用3个线程
    processor = ParallelDiseaseSymptomProcessor(api_key, api_base, max_workers)
    
    # 处理数据
    input_file = "disease_symptom_database.json"
    output_file = "processed_disease_symptom_database_parallel.json"
    
    # 分批处理，每批10个疾病 (更小的批次)
    batch_size = 10
    print(f"开始分批处理，每批 {batch_size} 个疾病...")
    print(f"结果将保存到: {output_file}")
    
    processor.process_diseases_parallel(input_file, output_file, batch_size)

if __name__ == "__main__":
    main()
