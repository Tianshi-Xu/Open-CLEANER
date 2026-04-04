#!/usr/bin/env python3
"""
根据rollout统计信息过滤数据集并生成不同版本的数据集
"""
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path


# 数据源分类映射
DATA_SOURCE_CATEGORY = {
    # Code 类
    'leetcode': 'code',
    'codeforces': 'code',
    'apps': 'code',
    'mbpp': 'code',
    'humaneval': 'code',
    'codecontests': 'code',
    'taco': 'code',
    'code': 'code',
    'programming': 'code',
    'bigcodebench': 'code',
    'livecodebench': 'code',
    
    # Math 类
    'math': 'math',
    'gsm8k': 'math',
    'mathqa': 'math',
    'aime': 'math',
    'olympiad': 'math',
    'competition_math': 'math',
    'algebra': 'math',
    'geometry': 'math',
    'number_theory': 'math',
    'counting': 'math',
    'prealgebra': 'math',
    'precalculus': 'math',
    'intermediate_algebra': 'math',
    'amc': 'math',
    'numina': 'math',
    'deepscaler': 'math',
    'open-r1-math': 'math',
    
    # Science 类
    'scibench': 'science',
    'scienceqa': 'science',
    'physics': 'science',
    'chemistry': 'science',
    'biology': 'science',
    'science': 'science',
    'gpqa': 'science',
}


def get_data_category(data_source):
    """
    根据数据源名称获取分类（code, math, science）
    
    Args:
        data_source: 数据源名称
        
    Returns:
        str: 分类名称 (code, math, science, other)
    """
    if pd.isna(data_source) or data_source == 'unknown':
        return 'other'
    
    data_source_lower = str(data_source).lower()
    
    # 精确匹配
    if data_source_lower in DATA_SOURCE_CATEGORY:
        return DATA_SOURCE_CATEGORY[data_source_lower]
    
    # 模糊匹配
    for key, category in DATA_SOURCE_CATEGORY.items():
        if key in data_source_lower or data_source_lower in key:
            return category
    
    return 'other'


def load_statistics(jsonl_path):
    """
    加载JSONL格式的统计信息
    
    Returns:
        dict: {uid: {successes, attempts, success_rate, avg_score, keep, data_source}}
    """
    stats_dict = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading statistics"):
            if line.strip():
                data = json.loads(line)
                uid = str(data['uid'])  # 确保uid是字符串
                stats_dict[uid] = {
                    'successes': data['successes'],
                    'attempts': data['attempts'],
                    'success_rate': data['success_rate'],
                    'avg_score': data['avg_score'],
                    'keep': data['keep'],
                    'data_source': data.get('data_source', 'unknown')
                }
    print(f"✅ 加载了 {len(stats_dict)} 条统计信息")
    return stats_dict


def load_dataset(parquet_path):
    """
    加载Parquet格式的数据集
    """
    print(f"正在加载数据集: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"✅ 数据集加载成功，共 {len(df)} 条数据")
    print(f"列名: {list(df.columns)}")
    
    # 确保有uid列（如果没有，使用索引作为uid）
    if 'uid' not in df.columns:
        print("⚠️  数据集中没有'uid'列，将使用索引作为uid")
        df['uid'] = df.index.astype(str)
    else:
        df['uid'] = df['uid'].astype(str)
    
    return df


def filter_dataset(df, stats_dict, filter_config, retain_code=False):
    """
    根据统计信息和过滤配置过滤数据集
    
    Args:
        df: 原始数据集DataFrame
        stats_dict: 统计信息字典
        filter_config: 过滤配置，例如：
            {
                'success_rate_ratios': {0.0: 0.25, 0.125: 0.5, ...}  # 每个成功率的保留比例
            }
        retain_code: 是否保留code类型任务（只过滤成功率为0的code任务）
    
    Returns:
        tuple: (过滤后的数据集, 过滤前有统计信息的数据集)
    """
    # 添加统计信息到数据集
    print("正在合并统计信息...")
    
    # 初始化统计列
    df['successes'] = -1
    df['attempts'] = -1
    df['success_rate'] = -1.0
    df['avg_score'] = -1.0
    df['keep'] = False
    df['data_source_stat'] = 'unknown'
    
    # 填充统计信息
    matched_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Matching statistics"):
        uid = str(row['uid'])
        if uid in stats_dict:
            stat = stats_dict[uid]
            df.at[idx, 'successes'] = stat['successes']
            df.at[idx, 'attempts'] = stat['attempts']
            df.at[idx, 'success_rate'] = stat['success_rate']
            df.at[idx, 'avg_score'] = stat['avg_score']
            df.at[idx, 'keep'] = stat['keep']
            df.at[idx, 'data_source_stat'] = stat['data_source']
            matched_count += 1
    
    print(f"✅ 成功匹配 {matched_count}/{len(df)} 条数据")
    
    # 只保留有统计信息的数据
    df_filtered = df[df['successes'] >= 0].copy()
    print(f"移除未匹配数据后剩余: {len(df_filtered)} 条")
    
    # 保存过滤前的数据集（用于对比）
    df_before_filter = df_filtered.copy()
    
    # 按成功率分层采样
    success_rate_ratios = filter_config.get('success_rate_ratios', {})
    
    # 为每个样本计算数据类别
    df_filtered['_data_category'] = df_filtered['data_source_stat'].apply(get_data_category)
    
    print(f"\n应用按成功率分层采样:")
    if retain_code:
        print(f"⚠️  注意: retain_code=True, code类型任务只过滤成功率为0的，其余全部保留")
    kept_indices_list = []
    
    # 定义9种成功率 (0/8 到 8/8)
    success_rates = [i / 8.0 for i in range(9)]
    epsilon = 0.001  # 浮点误差容忍度
    
    for sr in success_rates:
        # 找到该成功率的所有样本
        sr_mask = (df_filtered['success_rate'] >= sr - epsilon) & (df_filtered['success_rate'] <= sr + epsilon)
        sr_count = sr_mask.sum()
        
        if sr_count > 0:
            # 分离code类型和非code类型
            code_mask = sr_mask & (df_filtered['_data_category'] == 'code')
            non_code_mask = sr_mask & (df_filtered['_data_category'] != 'code')
            
            code_count = code_mask.sum()
            non_code_count = non_code_mask.sum()
            
            # 获取该成功率的保留比例（默认为1.0，即全部保留）
            keep_ratio = success_rate_ratios.get(sr, 1.0)
            
            # 处理code类型：根据retain_code参数决定是否特殊处理
            if retain_code:
                # retain_code=True: 只有成功率为0时才应用过滤，其他成功率全部保留
                if sr < epsilon:  # 成功率为0
                    code_keep_ratio = keep_ratio
                else:
                    code_keep_ratio = 1.0  # 非0成功率的code任务全部保留
            else:
                # retain_code=False: code类型和其他类型一样处理
                code_keep_ratio = keep_ratio
            
            code_keep_count = int(code_count * code_keep_ratio)
            non_code_keep_count = int(non_code_count * keep_ratio)
            
            # 处理code类型样本
            if code_count > 0:
                code_indices = df_filtered[code_mask].index.tolist()
                if code_keep_count >= code_count:
                    kept_indices_list.extend(code_indices)
                elif code_keep_count > 0:
                    np.random.seed(42 + int(sr * 1000))
                    kept_indices = np.random.choice(code_indices, size=code_keep_count, replace=False)
                    kept_indices_list.extend(kept_indices.tolist())
            
            # 处理非code类型样本
            if non_code_count > 0:
                non_code_indices = df_filtered[non_code_mask].index.tolist()
                if non_code_keep_count >= non_code_count:
                    kept_indices_list.extend(non_code_indices)
                elif non_code_keep_count > 0:
                    np.random.seed(43 + int(sr * 1000))  # 使用不同的随机种子
                    kept_indices = np.random.choice(non_code_indices, size=non_code_keep_count, replace=False)
                    kept_indices_list.extend(kept_indices.tolist())
            
            total_keep = code_keep_count + non_code_keep_count
            print(f"  成功率 {sr:.3f} ({int(sr*8)}/8): {sr_count} 条 (code:{code_count}, 其他:{non_code_count})")
            print(f"    -> 保留 {total_keep} 条 (code:{code_keep_count}[{code_keep_ratio*100:.1f}%], 其他:{non_code_keep_count}[{keep_ratio*100:.1f}%])")
        else:
            print(f"  成功率 {sr:.3f} ({int(sr*8)}/8): 0 条")
    
    # 清理临时列
    df_filtered.drop('_data_category', axis=1, inplace=True)
    
    # 创建最终的mask
    mask = pd.Series(False, index=df_filtered.index)
    mask.loc[kept_indices_list] = True
    
    result = df_filtered[mask].copy()
    print(f"✅ 过滤后剩余: {len(result)} 条数据")
    
    return result, df_before_filter


def print_statistics(df, name="数据集", df_before=None):
    """打印数据集统计信息
    
    Args:
        df: 当前数据集
        name: 数据集名称
        df_before: 过滤前的数据集（可选，用于对比显示）
    """
    print(f"\n{'='*60}")
    print(f"{name} 统计信息:")
    print(f"{'='*60}")
    print(f"总样本数: {len(df)}")
    
    if 'success_rate' in df.columns:
        print(f"\n成功率统计:")
        print(f"  平均成功率: {df['success_rate'].mean():.4f}")
        print(f"  中位数成功率: {df['success_rate'].median():.4f}")
        print(f"  最小成功率: {df['success_rate'].min():.4f}")
        print(f"  最大成功率: {df['success_rate'].max():.4f}")
        
        # 成功率分布
        print(f"\n成功率分布:")
        print(f"  [0.0]: {(df['success_rate'] == 0.0).sum()}")
        print(f"  (0.0, 0.2): {((df['success_rate'] > 0.0) & (df['success_rate'] < 0.2)).sum()}")
        print(f"  [0.2, 0.4): {((df['success_rate'] >= 0.2) & (df['success_rate'] < 0.4)).sum()}")
        print(f"  [0.4, 0.6): {((df['success_rate'] >= 0.4) & (df['success_rate'] < 0.6)).sum()}")
        print(f"  [0.6, 0.8): {((df['success_rate'] >= 0.6) & (df['success_rate'] < 0.8)).sum()}")
        print(f"  [0.8, 1.0): {((df['success_rate'] >= 0.8) & (df['success_rate'] < 1.0)).sum()}")
        print(f"  [1.0]: {(df['success_rate'] == 1.0).sum()}")
    
    if 'data_source_stat' in df.columns:
        print(f"\n数据源分布:")
        source_counts = df['data_source_stat'].value_counts()
        
        if df_before is not None and 'data_source_stat' in df_before.columns:
            # 对比显示过滤前后的分布
            source_counts_before = df_before['data_source_stat'].value_counts()
            all_sources = set(source_counts.index) | set(source_counts_before.index)
            
            print(f"  {'数据源':<25} {'过滤前':>10} {'过滤后':>10} {'保留比例':>10}")
            print(f"  {'-'*55}")
            for source in sorted(all_sources):
                before = source_counts_before.get(source, 0)
                after = source_counts.get(source, 0)
                ratio = f"{after/before*100:.1f}%" if before > 0 else "N/A"
                print(f"  {source:<25} {before:>10} {after:>10} {ratio:>10}")
            
            # 总计
            total_before = len(df_before)
            total_after = len(df)
            total_ratio = f"{total_after/total_before*100:.1f}%" if total_before > 0 else "N/A"
            print(f"  {'-'*55}")
            print(f"  {'总计':<25} {total_before:>10} {total_after:>10} {total_ratio:>10}")
        else:
            for source, count in source_counts.items():
                print(f"  {source}: {count}")
        
        # 按类别统计（code, math, science）
        print(f"\n数据类别分布 (code/math/science):")
        df['_category'] = df['data_source_stat'].apply(get_data_category)
        category_counts = df['_category'].value_counts()
        
        if df_before is not None and 'data_source_stat' in df_before.columns:
            df_before['_category'] = df_before['data_source_stat'].apply(get_data_category)
            category_counts_before = df_before['_category'].value_counts()
            
            total_before = len(df_before)
            total_after = len(df)
            
            print(f"  {'类别':<15} {'过滤前':>10} {'占比':>8} {'过滤后':>10} {'占比':>8} {'保留比例':>10}")
            print(f"  {'-'*65}")
            
            for category in ['code', 'math', 'science', 'other']:
                before = category_counts_before.get(category, 0)
                after = category_counts.get(category, 0)
                before_pct = f"{before/total_before*100:.1f}%" if total_before > 0 else "N/A"
                after_pct = f"{after/total_after*100:.1f}%" if total_after > 0 else "N/A"
                keep_ratio = f"{after/before*100:.1f}%" if before > 0 else "N/A"
                print(f"  {category:<15} {before:>10} {before_pct:>8} {after:>10} {after_pct:>8} {keep_ratio:>10}")
            
            # 总计
            total_ratio = f"{total_after/total_before*100:.1f}%" if total_before > 0 else "N/A"
            print(f"  {'-'*65}")
            print(f"  {'总计':<15} {total_before:>10} {'100.0%':>8} {total_after:>10} {'100.0%':>8} {total_ratio:>10}")
            
            # 清理临时列
            df_before.drop('_category', axis=1, inplace=True)
        else:
            total = len(df)
            print(f"  {'类别':<15} {'数量':>10} {'占比':>10}")
            print(f"  {'-'*35}")
            for category in ['code', 'math', 'science', 'other']:
                count = category_counts.get(category, 0)
                pct = f"{count/total*100:.1f}%" if total > 0 else "N/A"
                print(f"  {category:<15} {count:>10} {pct:>10}")
            print(f"  {'-'*35}")
            print(f"  {'总计':<15} {total:>10} {'100.0%':>10}")
        
        # 清理临时列
        df.drop('_category', axis=1, inplace=True)
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='根据统计信息过滤数据集')
    parser.add_argument('--stats', type=str, 
                        default='dataset/Open-AgentRL-30K/Open-AgentRL-30K.filtered.parquet.success.jsonl',
                        help='统计信息JSONL文件路径')
    parser.add_argument('--dataset', type=str,
                        default='dataset/Open-AgentRL-30K/Open-AgentRL-30K.parquet',
                        help='原始数据集Parquet文件路径')
    parser.add_argument('--output_dir', type=str,
                        default='dataset/Open-AgentRL-30K/filtered',
                        help='输出目录')
    parser.add_argument('--success_rate_ratios', type=str,
                        default=None,
                        help='每个成功率的保留比例，格式: "0.0:0.25,0.125:0.5,0.25:1.0,..." 或简化格式: "0.25,0.5,1.0,..." (按0/8到8/8顺序)')
    parser.add_argument('--retain_code', action='store_true',
                        help='是否保留code类型任务（启用后只过滤成功率为0的code任务，其余全部保留）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    stats_dict = load_statistics(args.stats)
    df_original = load_dataset(args.dataset)
    
    # 打印原始数据集统计
    print_statistics(df_original, "原始数据集")
    
    # 解析success_rate_ratios参数
    success_rate_ratios = {}
    if args.success_rate_ratios:
        # 定义9种成功率
        success_rates = [i / 8.0 for i in range(9)]
        
        if ':' in args.success_rate_ratios:
            # 格式: "0.0:0.25,0.125:0.5,..."
            for pair in args.success_rate_ratios.split(','):
                sr_str, ratio_str = pair.split(':')
                success_rate_ratios[float(sr_str)] = float(ratio_str)
        else:
            # 简化格式: "0.25,0.5,1.0,..." (按0/8到8/8顺序)
            ratios = [float(r) for r in args.success_rate_ratios.split(',')]
            if len(ratios) != 9:
                raise ValueError(f"简化格式需要恰好9个比例值，实际提供了{len(ratios)}个")
            for sr, ratio in zip(success_rates, ratios):
                success_rate_ratios[sr] = ratio
    else:
        # 默认全部保留
        success_rate_ratios = {i / 8.0: 1.0 for i in range(9)}
    
    # 定义过滤配置
    filter_config = {'success_rate_ratios': success_rate_ratios}
    
    print(f"\n{'='*60}")
    print(f"过滤配置: 按成功率分层采样")
    print(f"retain_code: {args.retain_code}")
    for sr in sorted(success_rate_ratios.keys()):
        ratio = success_rate_ratios[sr]
        print(f"  成功率 {sr:.3f} ({int(sr*8)}/8): 保留比例 {ratio*100:.1f}%")
    print(f"{'='*60}")
    
    # 应用过滤
    filtered_df, df_before_filter = filter_dataset(df_original, stats_dict, filter_config, retain_code=args.retain_code)
    
    # 打印统计信息（包含过滤前后对比）
    print_statistics(filtered_df, "过滤后的数据集", df_before=df_before_filter)
    
    # 保存过滤后的数据集
    output_filename = f"filtered_sr.parquet"
    output_path = output_dir / output_filename
    filtered_df.to_parquet(output_path, index=False)
    
    print(f"\n✅ 过滤完成！")
    print(f"输出文件: {output_path}")


if __name__ == '__main__':
    main()
