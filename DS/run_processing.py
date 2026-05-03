#!/usr/bin/env python3
"""
疾病症状数据库处理工具 - 主运行脚本
支持串行和并行两种处理方式
"""

import os
import sys
import time
from datetime import datetime

def show_menu():
    """
    显示主菜单
    """
    print("=" * 60)
    print("疾病症状数据库处理工具")
    print("=" * 60)
    print("请选择处理方式：")
    print()
    print("1. 快速测试 (并行) - 处理前6个疾病")
    print("2. 分批处理 (并行) - 分批处理所有疾病 (推荐)")
    print("3. 查看帮助信息")
    print("0. 退出")
    print("=" * 60)

def run_parallel_test():
    """
    运行并行测试
    """
    print("\n开始并行测试...")
    print("处理前6个疾病，预计用时: 30-60秒")
    
    try:
        from test_parallel_processor import test_parallel_processing
        test_parallel_processing()
        print("\n并行测试完成！结果保存在: test_parallel_processed_diseases.json")
    except Exception as e:
        print(f"并行测试失败: {e}")

def run_batch_processing():
    """
    运行分批处理
    """
    print("\n开始分批处理...")
    print("分批处理所有疾病，支持断点续传")
    
    # 询问参数
    try:
        max_workers = int(input("请输入线程数 (建议5-10，默认5): ") or "5")
        if max_workers < 1 or max_workers > 20:
            print("线程数无效，使用默认值5")
            max_workers = 5
    except ValueError:
        max_workers = 5
    
    try:
        batch_size = int(input("请输入每批处理数量 (建议20-100，默认50): ") or "50")
        if batch_size < 1 or batch_size > 200:
            print("批次大小无效，使用默认值50")
            batch_size = 50
    except ValueError:
        batch_size = 50
    
    print(f"使用 {max_workers} 个线程，每批处理 {batch_size} 个疾病")
    
    try:
        from disease_symptom_processor_parallel import ParallelDiseaseSymptomProcessor
        
        # 配置API - 华为云ModelArts
        api_key = 'WEBaX4AAhPUQyTHbQIYhWq8vURSJnU8WNFN1zpbrCZ1qFl87XTpH9M-UGQJkO_UHfGIXqaUJ3CqgI3M2DBuSng'
        api_base = "https://api.modelarts-maas.com/v1/chat/completions"
        
        # 创建处理器
        processor = ParallelDiseaseSymptomProcessor(api_key, api_base, max_workers)
        
        # 处理数据
        input_file = "disease_symptom_database.json"
        output_file = "processed_disease_symptom_database_parallel.json"
        
        processor.process_diseases_parallel(input_file, output_file, batch_size)
        print("\n分批处理完成！")
        print("\n" + "="*50)
        print("批次处理完成！")
        print("请运行以下命令合并所有批次文件：")
        print("python merge_batches.py")
    except Exception as e:
        print(f"分批处理失败: {e}")

def show_help():
    """
    显示帮助信息
    """
    print("\n帮助信息")
    print("=" * 40)
    print("1. 快速测试：处理前6个疾病验证功能")
    print("2. 分批处理：分批处理所有疾病，支持断点续传")
    print()
    print("分批处理特点：")
    print("- 每处理一批就保存结果")
    print("- 支持断点续传，中断后可继续")
    print("- 可自定义批次大小和线程数")
    print("- 避免长时间运行的风险")
    print()
    print("输出文件：")
    print("- test_parallel_processed_diseases.json (测试结果)")
    print("- processed_disease_symptom_database_parallel.json (分批处理结果)")
    print()
    print("参数建议：")
    print("- 线程数：5-10个 (根据网络情况)")
    print("- 批次大小：20-100个 (根据内存情况)")
    print("- 需要稳定的网络连接")
    print()
    print("故障排除：")
    print("- API调用失败：检查网络连接")
    print("- 内存不足：减少批次大小")
    print("- 处理中断：重新运行会自动续传")

def main():
    """
    主函数
    """
    while True:
        show_menu()
        
        try:
            choice = input("\n请选择 (0-3): ").strip()
            
            if choice == '0':
                print("\n再见！")
                break
            elif choice == '1':
                run_parallel_test()
            elif choice == '2':
                run_batch_processing()
            elif choice == '3':
                show_help()
            else:
                print("\n无效选择，请重新输入")
            
            if choice in ['1', '2']:
                input("\n按回车键继续...")
                
        except KeyboardInterrupt:
            print("\n\n用户中断，再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            input("按回车键继续...")

if __name__ == "__main__":
    main()
