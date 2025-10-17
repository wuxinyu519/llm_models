#!/usr/bin/env python3
import pickle
import os
import json
from collections import Counter
import datetime

def analyze_pkl_files_summary(data_directory, save_results=True, output_file=None):
    """
    统计数据目录中所有pkl文件的生成标签分布，只输出摘要
    
    Args:
        data_directory: 包含pkl文件的目录路径
        save_results: 是否保存结果到文件
        output_file: 输出文件路径，如果为None则自动生成
    """
    # 递归查找所有pkl文件
    pkl_files = []
    for root, dirs, files in os.walk(data_directory):
        for f in files:
            if f.endswith('.pkl') and '_results.pkl' in f:
                pkl_files.append(os.path.join(root, f))
    
    if not pkl_files:
        print(f"在目录 {data_directory} 中没有找到结果pkl文件")
        return
    
    # 准备输出内容
    output_lines = []
    output_lines.append(f"PKL文件生成标签统计分析报告")
    output_lines.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"数据目录: {data_directory}")
    output_lines.append(f"找到 {len(pkl_files)} 个pkl结果文件\n")
    
    print(f"找到 {len(pkl_files)} 个pkl结果文件\n")
    
    # 标签类型名称
    tag_types = ['Domain', 'Task Type', 'Difficulty Level', 'Language']
    
    # 为每个pkl文件统计
    for i, pkl_file in enumerate(pkl_files, 1):
        try:
            # 加载pkl文件
            print(f"正在加载: {pkl_file}")
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, list):
                print(f"{pkl_file} 格式错误")
                continue
            
            print(f"{'='*80}")
            print(f"文件 {i}/{len(pkl_files)}: {os.path.basename(pkl_file)}")
            print(f"完整路径: {pkl_file}")
            print(f"{'='*80}")
            
            # 添加到输出
            output_lines.append(f"{'='*80}")
            output_lines.append(f"文件 {i}/{len(pkl_files)}: {os.path.basename(pkl_file)}")
            output_lines.append(f"完整路径: {pkl_file}")
            output_lines.append(f"{'='*80}")
            
            # 提取所有generated_tags
            all_generated_tags = []
            all_ground_truth_tags = []
            
            for item in data:
                if isinstance(item, dict):
                    # 提取生成的标签
                    if 'generated_tags' in item:
                        generated_tags = item['generated_tags']
                        if isinstance(generated_tags, list):
                            all_generated_tags.append(generated_tags)
                    
                    # 提取ground truth标签（用于对比）
                    if 'parsed_tags' in item:
                        gt_tags = item['parsed_tags']
                        if isinstance(gt_tags, list):
                            all_ground_truth_tags.append(gt_tags)
            
            if not all_generated_tags:
                msg = f"警告: 文件 {os.path.basename(pkl_file)} 中没有找到有效的generated_tags"
                print(msg)
                output_lines.append(msg)
                output_lines.append("")
                continue
            
            record_count_msg = f"总记录数: {len(all_generated_tags)}"
            gt_count_msg = f"Ground Truth记录数: {len(all_ground_truth_tags)}\n"
            print(record_count_msg)
            print(gt_count_msg)
            output_lines.append(record_count_msg)
            output_lines.append(gt_count_msg)
            
            # ========== 分析生成的标签 ==========
            output_lines.append("【模型生成的标签统计】")
            print("【模型生成的标签统计】")
            
            # 为每种标签类型统计（前4个位置）
            for tag_index in range(4):
                tag_type = tag_types[tag_index]
                
                # 提取对应位置的标签
                position_tags = []
                for record_tags in all_generated_tags:
                    if isinstance(record_tags, list) and len(record_tags) > tag_index:
                        tag_item = record_tags[tag_index]
                        if isinstance(tag_item, dict) and 'tag' in tag_item:
                            position_tags.append(tag_item['tag'])
                        elif isinstance(tag_item, str):
                            position_tags.append(tag_item)
                        else:
                            position_tags.append(str(tag_item))
                
                if not position_tags:
                    msg = f"{tag_type}: 没有找到数据"
                    print(msg)
                    output_lines.append(msg)
                    continue
                
                # 统计频次
                tag_counts = Counter(position_tags)
                unique_count = len(tag_counts)
                total_count = len(position_tags)
                
                # 输出到控制台和文件
                header = f"{tag_type}:"
                unique_msg = f"  唯一值数量: {unique_count}"
                total_msg = f"  总标签数量: {total_count}"
                dist_msg = f"  分布情况:"
                
                print(header)
                print(unique_msg)
                print(total_msg)
                print(dist_msg)
                
                output_lines.append(header)
                output_lines.append(unique_msg)
                output_lines.append(total_msg)
                output_lines.append(dist_msg)
                
                # 按频次排序显示
                for tag, count in tag_counts.most_common():
                    percentage = (count / total_count) * 100
                    item_msg = f"    {tag}: {count} ({percentage:.1f}%)"
                    print(item_msg)
                    output_lines.append(item_msg)
                
                print()
                output_lines.append("")
            
            # 如果有第5个及以后的位置，统计Topic(s)
            topic_tags = []
            for record_tags in all_generated_tags:
                if isinstance(record_tags, list) and len(record_tags) > 4:
                    for topic_index in range(4, len(record_tags)):
                        tag_item = record_tags[topic_index]
                        if isinstance(tag_item, dict) and 'tag' in tag_item:
                            topic_tags.append(tag_item['tag'])
                        elif isinstance(tag_item, str):
                            topic_tags.append(tag_item)
                        else:
                            topic_tags.append(str(tag_item))
            
            if topic_tags:
                # 统计Topic(s)频次
                topic_counts = Counter(topic_tags)
                unique_count = len(topic_counts)
                total_count = len(topic_tags)
                
                header = f"Topic(s):"
                unique_msg = f"  唯一值数量: {unique_count}"
                total_msg = f"  总标签数量: {total_count}"
                dist_msg = f"  分布情况 (Top 30):"
                
                print(header)
                print(unique_msg)
                print(total_msg)
                print(dist_msg)
                
                output_lines.append(header)
                output_lines.append(unique_msg)
                output_lines.append(total_msg)
                output_lines.append(dist_msg)
                
                # 显示前30个最常见的Topic
                for tag, count in topic_counts.most_common(30):
                    percentage = (count / total_count) * 100
                    item_msg = f"    {tag}: {count} ({percentage:.1f}%)"
                    print(item_msg)
                    output_lines.append(item_msg)
                
                print()
                output_lines.append("")
            
            # ========== 对比 Ground Truth 标签（如果存在）==========
            if all_ground_truth_tags:
                output_lines.append("\n【Ground Truth标签统计（用于对比）】")
                print("\n【Ground Truth标签统计（用于对比）】")
                
                for tag_index in range(4):
                    tag_type = tag_types[tag_index]
                    
                    # 提取对应位置的GT标签
                    gt_position_tags = []
                    for record_tags in all_ground_truth_tags:
                        if isinstance(record_tags, list) and len(record_tags) > tag_index:
                            tag_item = record_tags[tag_index]
                            if isinstance(tag_item, dict) and 'tag' in tag_item:
                                gt_position_tags.append(tag_item['tag'])
                            elif isinstance(tag_item, str):
                                gt_position_tags.append(tag_item)
                            else:
                                gt_position_tags.append(str(tag_item))
                    
                    if not gt_position_tags:
                        continue
                    
                    # 统计频次
                    gt_tag_counts = Counter(gt_position_tags)
                    gt_unique_count = len(gt_tag_counts)
                    gt_total_count = len(gt_position_tags)
                    
                    header = f"{tag_type} (GT):"
                    unique_msg = f"  唯一值数量: {gt_unique_count}"
                    total_msg = f"  总标签数量: {gt_total_count}"
                    dist_msg = f"  分布情况:"
                    
                    print(header)
                    print(unique_msg)
                    print(total_msg)
                    print(dist_msg)
                    
                    output_lines.append(header)
                    output_lines.append(unique_msg)
                    output_lines.append(total_msg)
                    output_lines.append(dist_msg)
                    
                    # 显示GT标签分布
                    for tag, count in gt_tag_counts.most_common():
                        percentage = (count / gt_total_count) * 100
                        item_msg = f"    {tag}: {count} ({percentage:.1f}%)"
                        print(item_msg)
                        output_lines.append(item_msg)
                    
                    print()
                    output_lines.append("")
            
            # ========== 统计生成失败的情况 ==========
            failed_count = 0
            empty_count = 0
            for record_tags in all_generated_tags:
                if not record_tags:
                    empty_count += 1
                elif len(record_tags) < 5:
                    failed_count += 1
            
            if failed_count > 0 or empty_count > 0:
                output_lines.append("\n【生成质量统计】")
                print("\n【生成质量统计】")
                
                fail_msg = f"  生成失败（空标签）: {empty_count}"
                incomplete_msg = f"  生成不完整（少于5个标签）: {failed_count}"
                success_rate = ((len(all_generated_tags) - empty_count - failed_count) / len(all_generated_tags)) * 100
                success_msg = f"  完整生成率: {success_rate:.1f}%"
                
                print(fail_msg)
                print(incomplete_msg)
                print(success_msg)
                
                output_lines.append(fail_msg)
                output_lines.append(incomplete_msg)
                output_lines.append(success_msg)
            
        except Exception as e:
            error_msg = f"处理文件 {pkl_file} 时出错: {str(e)}"
            print(error_msg)
            output_lines.append(error_msg)
            import traceback
            traceback.print_exc()
        
        print()  # 文件之间的分隔
        output_lines.append("\n")
    
    # ========== 汇总统计（所有文件） ==========
    output_lines.append("="*80)
    output_lines.append("【全局汇总统计】")
    output_lines.append("="*80)
    print("="*80)
    print("【全局汇总统计】")
    print("="*80)
    
    global_summary = f"分析了 {len(pkl_files)} 个PKL文件"
    print(global_summary)
    output_lines.append(global_summary)
    
    # 保存结果到文件
    if save_results:
        if output_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"pkl_generated_tags_analysis_{timestamp}.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            print(f"\n结果已保存到: {os.path.abspath(output_file)}")
        except Exception as e:
            print(f"保存文件时出错: {str(e)}")
    
    return output_lines

def compare_generated_vs_gt(pkl_file):
    """
    对比单个PKL文件中生成标签和GT标签的差异
    
    Args:
        pkl_file: PKL文件路径
    """
    print(f"\n{'='*80}")
    print(f"对比分析: {os.path.basename(pkl_file)}")
    print(f"{'='*80}\n")
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, list):
            print("错误: PKL文件格式不正确")
            return
        
        # 统计标签匹配情况
        total_samples = len(data)
        exact_matches = 0
        partial_matches = 0
        no_matches = 0
        
        tag_types = ['Domain', 'Task Type', 'Difficulty', 'Language', 'Topics']
        position_match_counts = [0] * 5
        
        for item in data:
            if not isinstance(item, dict):
                continue
            
            gen_tags = item.get('generated_tags', [])
            gt_tags = item.get('parsed_tags', [])
            
            if not gen_tags or not gt_tags:
                continue
            
            # 提取标签文本
            gen_tag_texts = []
            for tag_item in gen_tags:
                if isinstance(tag_item, dict) and 'tag' in tag_item:
                    gen_tag_texts.append(tag_item['tag'].lower().strip())
                elif isinstance(tag_item, str):
                    gen_tag_texts.append(tag_item.lower().strip())
            
            gt_tag_texts = []
            for tag_item in gt_tags:
                if isinstance(tag_item, dict) and 'tag' in tag_item:
                    gt_tag_texts.append(tag_item['tag'].lower().strip())
                elif isinstance(tag_item, str):
                    gt_tag_texts.append(tag_item.lower().strip())
            
            # 计算匹配
            matches = 0
            for i in range(min(len(gen_tag_texts), len(gt_tag_texts), 5)):
                if gen_tag_texts[i] == gt_tag_texts[i]:
                    matches += 1
                    position_match_counts[i] += 1
            
            if matches == len(gt_tag_texts):
                exact_matches += 1
            elif matches > 0:
                partial_matches += 1
            else:
                no_matches += 1
        
        # 输出统计结果
        print(f"总样本数: {total_samples}")
        print(f"完全匹配: {exact_matches} ({exact_matches/total_samples*100:.1f}%)")
        print(f"部分匹配: {partial_matches} ({partial_matches/total_samples*100:.1f}%)")
        print(f"完全不匹配: {no_matches} ({no_matches/total_samples*100:.1f}%)")
        print()
        
        print("各位置标签匹配率:")
        for i, tag_type in enumerate(tag_types[:5]):
            if i < len(position_match_counts):
                match_rate = (position_match_counts[i] / total_samples) * 100
                print(f"  {tag_type}: {position_match_counts[i]}/{total_samples} ({match_rate:.1f}%)")
        
    except Exception as e:
        print(f"分析出错: {str(e)}")
        import traceback
        traceback.print_exc()

# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分析PKL文件中生成的标签统计")
    parser.add_argument("--data_dir", type=str, default="./", 
                        help="包含PKL结果文件的目录路径")
    parser.add_argument("--output_file", type=str, default="gemma3_1b_pkl_tags_analysis.txt",
                        help="输出文件路径（默认自动生成）")
    parser.add_argument("--no_save", action="store_true",
                        help="不保存结果到文件")
    parser.add_argument("--compare", type=str, default=None,
                        help="对比指定PKL文件中生成标签和GT标签")
    
    args = parser.parse_args()
    
    print("PKL文件生成标签统计分析工具")
    print("=" * 80)
    
    if args.compare:
        # 详细对比模式
        compare_generated_vs_gt(args.compare)
    else:
        # 统计分析模式
        analyze_pkl_files_summary(
            args.data_dir, 
            save_results=not args.no_save,
            output_file=args.output_file
        )
    
    print("\n使用方法:")
    print("1. 统计分析所有PKL文件:")
    print("   python analyze_pkl_tags.py --data_dir ./results_gemma3_1b_unified")
    print()
    print("2. 不保存到文件:")
    print("   python analyze_pkl_tags.py --data_dir ./results --no_save")
    print()
    print("3. 自定义输出文件名:")
    print("   python analyze_pkl_tags.py --data_dir ./results --output_file my_analysis.txt")
    print()
    print("4. 详细对比生成标签vs GT标签:")
    print("   python analyze_pkl_tags.py --compare ./results/file1_results.pkl")
    print()
    print("输出内容:")
    print("  - 每个PKL文件的生成标签分布统计")
    print("  - Domain, Task Type, Difficulty, Language 各类别分布")
    print("  - Topics 标签分布（Top 30）")
    print("  - Ground Truth 标签分布（用于对比）")
    print("  - 生成质量统计（成功率、失败率）")