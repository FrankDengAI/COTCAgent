import json
import glob
import os
from typing import List, Dict, Any

def merge_batch_files(output_file: str = "merged_processed_diseases.json"):
    """
    合并所有批次文件为最终结果
    """
    print("开始合并批次文件...")
    
    # 查找所有批次文件
    batch_files = glob.glob("batch_*_processed_diseases.json")
    batch_files.sort()  # 按文件名排序
    
    if not batch_files:
        print("未找到任何批次文件！")
        return
    
    print(f"找到 {len(batch_files)} 个批次文件")
    
    all_diseases = []
    processed_count = 0
    
    for batch_file in batch_files:
        try:
            print(f"读取批次文件: {batch_file}")
            with open(batch_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                batch_diseases = data.get('疾病库', [])
                all_diseases.extend(batch_diseases)
                processed_count += len(batch_diseases)
                print(f"  - 包含 {len(batch_diseases)} 个疾病")
        except Exception as e:
            print(f"读取批次文件失败 {batch_file}: {e}")
    
    if not all_diseases:
        print("没有找到任何疾病数据！")
        return
    
    # 去重（按疾病ID）
    unique_diseases = []
    seen_ids = set()
    
    for disease in all_diseases:
        disease_id = disease.get('疾病ID')
        if disease_id and disease_id not in seen_ids:
            unique_diseases.append(disease)
            seen_ids.add(disease_id)
    
    print(f"\n合并统计:")
    print(f"总疾病数: {len(all_diseases)}")
    print(f"去重后: {len(unique_diseases)}")
    print(f"重复数: {len(all_diseases) - len(unique_diseases)}")
    
    # 保存合并结果
    merged_data = {
        "疾病库": unique_diseases
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n合并完成！")
    print(f"最终结果已保存到: {output_file}")
    print(f"包含 {len(unique_diseases)} 个疾病")

def list_batch_files():
    """
    列出所有批次文件
    """
    batch_files = glob.glob("batch_*_processed_diseases.json")
    batch_files.sort()
    
    if not batch_files:
        print("未找到任何批次文件")
        return
    
    print(f"找到 {len(batch_files)} 个批次文件:")
    total_diseases = 0
    
    for batch_file in batch_files:
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                count = len(data.get('疾病库', []))
                total_diseases += count
                print(f"  {batch_file}: {count} 个疾病")
        except Exception as e:
            print(f"  {batch_file}: 读取失败 - {e}")
    
    print(f"\n总计: {total_diseases} 个疾病")

def main():
    print("批次文件合并工具")
    print("=" * 50)
    
    # 列出所有批次文件
    list_batch_files()
    
    print("\n" + "=" * 50)
    
    # 合并所有批次文件
    merge_batch_files()

if __name__ == "__main__":
    main()
