"""
过滤低频 Tag 的数据清洗脚本（灵活版）

功能：
1. 统计每个 response 前4个 tag 的出现频率
2. 可灵活指定检查哪些列（位置）
3. 删除指定列中包含低频 tag (出现次数 < 阈值) 的样本

Usage:
    # 只检查第1列（Domain）
    python filter_tags.py --input data.json --output cleaned.json --check-positions 0 --min-freq 2
    
    # 检查第1和第3列（Domain和Difficulty）
    python filter_tags.py --input data.json --output cleaned.json --check-positions 0 2 --min-freq 2
    
    # 检查所有4列（默认）
    python filter_tags.py --input data.json --output cleaned.json --check-positions 0 1 2 3 --min-freq 2
"""

import json
import argparse
from collections import Counter
from typing import List, Dict, Any, Set


class TagFrequencyFilter:
    def __init__(self, min_frequency: int = 2, check_positions: List[int] = None):
        self.min_frequency = min_frequency
        # 默认检查所有4个位置
        self.check_positions = check_positions if check_positions else [0, 1, 2, 3]
        
    def load_json(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_first_4_tags(self, response: Dict) -> List[str]:
        """提取 response 的前4个 tag"""
        text = response.get('text', [])
        
        if isinstance(text, list):
            # 取前4个 tag
            tags = []
            for item in text[:4]:
                if isinstance(item, dict) and 'tag' in item:
                    tags.append(item['tag'])
            return tags
        
        return []
    
    def collect_tag_frequencies(self, data: List[Dict]) -> Dict[int, Counter]:
        """
        统计每个位置的 tag 频率
        Returns: {position: Counter({tag: count})}
        """
        position_counters = {0: Counter(), 1: Counter(), 2: Counter(), 3: Counter()}
        
        for item in data:
            responses = item.get('responses', [])
            
            for response in responses:
                tags = self.extract_first_4_tags(response)
                
                # 统计每个位置的 tag
                for pos, tag in enumerate(tags):
                    if pos < 4:
                        position_counters[pos][tag] += 1
        
        return position_counters
    
    def is_valid_sample(self, item: Dict, position_counters: Dict[int, Counter]) -> bool:
        """
        检查样本是否有效
        只检查指定位置的 tag，如果有低频tag则无效
        """
        responses = item.get('responses', [])
        
        for response in responses:
            tags = self.extract_first_4_tags(response)
            
            # 只检查指定位置的 tag
            for pos in self.check_positions:
                if pos < len(tags):
                    tag = tags[pos]
                    count = position_counters[pos].get(tag, 0)
                    if count < self.min_frequency:
                        return False  
        
        return True  
    
    def filter_data(self, data: List[Dict]) -> tuple[List[Dict], Dict]:
        """
        过滤数据
        Returns: (filtered_data, stats)
        """
        position_names = ["Domain", "Task Type", "Difficulty", "Language"]
        
        print(f"原始样本数: {len(data)}")
        print(f"检查的位置: {[position_names[p] for p in self.check_positions]}")
        print(f"最小频率阈值: {self.min_frequency}")
        print("=" * 60)
        
        # Step 1: 统计频率
        print("\n统计 Tag 频率...")
        position_counters = self.collect_tag_frequencies(data)
        
        # 打印频率统计
        for pos in range(4):
            check_marker = "[CHECK]" if pos in self.check_positions else "PASS]"
            print(f"\n{check_marker} 位置 {pos + 1} ({position_names[pos]}) 的 Tag 分布:")
            
            # 显示所有 tag，按频率排序
            for tag, count in position_counters[pos].most_common():
                if pos in self.check_positions:
                    status = "TRUE" if count >= self.min_frequency else "FALSE"
                    print(f"  {status} {tag}: {count} 次")
                else:
                    # 不检查的位置，只显示前10个
                    if position_counters[pos].most_common().index((tag, count)) < 10:
                        print(f" {tag}: {count} 次")
        
        # Step 2: 过滤低频样本
        print(f"\n过滤低频 tag (仅检查指定位置)...")
        
        filtered_data = []
        removed_indices = []
        removed_reasons = {}  # 记录删除原因
        
        for item in data:
            if self.is_valid_sample(item, position_counters):
                filtered_data.append(item)
            else:
                idx = item.get('index', -1)
                removed_indices.append(idx)
                
                # 记录删除原因
                reasons = []
                for response in item.get('responses', []):
                    tags = self.extract_first_4_tags(response)
                    for pos in self.check_positions:
                        if pos < len(tags):
                            tag = tags[pos]
                            count = position_counters[pos].get(tag, 0)
                            if count < self.min_frequency:
                                reasons.append(f"位置{pos+1}({position_names[pos]}): '{tag}' 仅{count}次")
                
                removed_reasons[idx] = reasons[:3]  # 只记录前3个原因
        
        # 统计信息
        stats = {
            'original_count': len(data),
            'kept_count': len(filtered_data),
            'removed_count': len(removed_indices),
            'check_positions': self.check_positions,
            'check_position_names': [position_names[p] for p in self.check_positions],
            'min_frequency': self.min_frequency,
            'removed_indices': removed_indices[:50],  # 保存前50个
            'removed_reasons': dict(list(removed_reasons.items())[:20]),  # 前20个原因
            'position_counters': {
                pos: dict(counter.most_common()) 
                for pos, counter in position_counters.items()
            }
        }
        
        return filtered_data, stats
    
    def save_results(self, data: List[Dict], output_path: str):
        """保存结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"已保存到: {output_path}")
    
    def print_stats(self, stats: Dict):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("过滤结果统计")
        print("=" * 60)
        print(f"检查的位置: {', '.join(stats['check_position_names'])}")
        print(f"最小频率阈值: {stats['min_frequency']}")
        print(f"\n原始样本数: {stats['original_count']}")
        print(f"保留样本数: {stats['kept_count']}")
        print(f"删除样本数: {stats['removed_count']}")
        print(f"保留率: {stats['kept_count'] / stats['original_count'] * 100:.1f}%")
        
        if stats['removed_indices']:
            print(f"\n删除的样本索引 (前50个): {stats['removed_indices']}")
        
        if stats['removed_reasons']:
            print(f"\n删除原因示例:")
            for idx, reasons in list(stats['removed_reasons'].items())[:5]:
                print(f"  样本 {idx}:")
                for reason in reasons:
                    print(f"    - {reason}")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='过滤低频 Tag 的样本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 只检查第1列（Domain）
  python filter_tags.py --input data.json --output cleaned.json --check-positions 0 --min-freq 2
  
  # 检查第1和第3列（Domain和Difficulty）
  python filter_tags.py --input data.json --output cleaned.json --check-positions 0 2 --min-freq 5
  
  # 检查所有4列（默认）
  python filter_tags.py --input data.json --output cleaned.json --min-freq 2
  
位置索引:
  0 = Domain (领域)
  1 = Task Type (任务类型)
  2 = Difficulty (难度)
  3 = Language (语言)
        """
    )
    parser.add_argument('--input', required=True, help='输入 JSON 文件')
    parser.add_argument('--output', required=True, help='输出 JSON 文件')
    parser.add_argument('--min-freq', type=int, default=2, 
                       help='最小频率阈值（默认2，即至少出现2次）')
    parser.add_argument('--check-positions', type=int, nargs='+', 
                       default=[0, 1, 2, 3],
                       help='要检查的位置索引（0-3），默认检查所有4个位置')
    parser.add_argument('--stats-output', default=None, 
                       help='保存统计信息的 JSON 文件（可选）')
    
    args = parser.parse_args()
    
    # 验证位置索引
    for pos in args.check_positions:
        if pos < 0 or pos > 3:
            print(f"位置索引必须在 0-3 之间，收到: {pos}")
            return
    
    # 初始化过滤器
    filter_tool = TagFrequencyFilter(
        min_frequency=args.min_freq,
        check_positions=args.check_positions
    )
    
    # 加载数据
    print(f"加载数据: {args.input}")
    data = filter_tool.load_json(args.input)
    
    # 过滤
    filtered_data, stats = filter_tool.filter_data(data)
    
    # 保存结果
    filter_tool.save_results(filtered_data, args.output)
    
    # 打印统计
    filter_tool.print_stats(stats)
    
    # 保存统计信息（可选）
    if args.stats_output:
        with open(args.stats_output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\n统计信息已保存到: {args.stats_output}")


if __name__ == "__main__":
    main()