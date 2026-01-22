#!/usr/bin/env python3
"""
延迟测试结果可视化工具

使用方法:
    python visualize_benchmark.py latency_benchmark_results.json
"""

import json
import argparse
import sys


def plot_results(results_file: str):
    """绘制测试结果图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
    except ImportError:
        print("❌ 需要安装 matplotlib: pip install matplotlib")
        sys.exit(1)
    
    # 读取结果
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    
    if not results:
        print("❌ 没有找到测试数据")
        sys.exit(1)
    
    # 提取数据
    concurrencies = [r['concurrency'] for r in results]
    avg_latencies = [r['avg_latency_ms'] for r in results]
    median_latencies = [r['median_latency_ms'] for r in results]
    p95_latencies = [r['p95_latency_ms'] for r in results]
    p99_latencies = [r['p99_latency_ms'] for r in results]
    qps = [r['requests_per_second'] for r in results]
    success_rates = [r['successful_requests'] / r['total_requests'] * 100 for r in results]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('沙箱执行环境高并发延迟测试结果', fontsize=16, fontweight='bold')
    
    # 1. 延迟随并发数变化
    ax1 = axes[0, 0]
    ax1.plot(concurrencies, avg_latencies, 'o-', label='平均延迟', linewidth=2)
    ax1.plot(concurrencies, median_latencies, 's-', label='中位数延迟', linewidth=2)
    ax1.plot(concurrencies, p95_latencies, '^-', label='P95延迟', linewidth=2)
    ax1.plot(concurrencies, p99_latencies, 'd-', label='P99延迟', linewidth=2)
    ax1.set_xlabel('并发数', fontsize=12)
    ax1.set_ylabel('延迟 (ms)', fontsize=12)
    ax1.set_title('延迟 vs 并发数', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. QPS随并发数变化
    ax2 = axes[0, 1]
    ax2.plot(concurrencies, qps, 'o-', color='green', linewidth=2, markersize=6)
    ax2.set_xlabel('并发数', fontsize=12)
    ax2.set_ylabel('QPS (请求/秒)', fontsize=12)
    ax2.set_title('吞吐量 vs 并发数', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 标记最大QPS点
    max_qps_idx = qps.index(max(qps))
    ax2.annotate(f'最大QPS: {qps[max_qps_idx]:.1f}\n并发数: {concurrencies[max_qps_idx]}',
                xy=(concurrencies[max_qps_idx], qps[max_qps_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 3. 成功率随并发数变化
    ax3 = axes[1, 0]
    ax3.plot(concurrencies, success_rates, 'o-', color='blue', linewidth=2, markersize=6)
    ax3.set_xlabel('并发数', fontsize=12)
    ax3.set_ylabel('成功率 (%)', fontsize=12)
    ax3.set_title('成功率 vs 并发数', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 105])
    ax3.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='100%')
    ax3.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90%')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 延迟增长率 (相对于基准)
    ax4 = axes[1, 1]
    baseline_latency = avg_latencies[0]
    latency_increase = [(lat - baseline_latency) / baseline_latency * 100 for lat in avg_latencies]
    ax4.plot(concurrencies, latency_increase, 'o-', color='red', linewidth=2, markersize=6)
    ax4.set_xlabel('并发数', fontsize=12)
    ax4.set_ylabel('延迟增长率 (%)', fontsize=12)
    ax4.set_title('延迟增长率 vs 并发数 (相对于最低并发)', fontsize=14, fontweight='bold')
    ax4.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50%增长阈值')
    ax4.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100%增长阈值')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 标记显著性能下降点
    for i, increase in enumerate(latency_increase):
        if i > 0 and increase > 50 and latency_increase[i-1] <= 50:
            ax4.annotate(f'性能下降点\n并发数: {concurrencies[i]}\n增长: {increase:.1f}%',
                        xy=(concurrencies[i], increase),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                        color='white', fontweight='bold')
            break
    
    plt.tight_layout()
    
    # 保存图表
    output_file = results_file.replace('.json', '.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存到: {output_file}")
    
    # 尝试显示图表
    try:
        plt.show()
    except:
        pass


def print_text_summary(results_file: str):
    """打印文本摘要（不需要matplotlib）"""
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    
    print("\n" + "=" * 100)
    print("📊 测试结果摘要")
    print("=" * 100)
    
    print(f"\n配置信息:")
    print(f"  - 服务地址: {data['config']['host']}")
    print(f"  - 并发范围: {data['config']['min_concurrency']} ~ {data['config']['max_concurrency']}")
    print(f"  - 每级请求数: {data['config']['requests_per_level']}")
    
    print(f"\n{'并发数':<10} {'平均延迟':<12} {'P95延迟':<12} {'P99延迟':<12} {'QPS':<10} {'成功率':<10}")
    print("-" * 100)
    
    for r in results:
        success_rate = r['successful_requests'] / r['total_requests'] * 100
        print(f"{r['concurrency']:<10} "
              f"{r['avg_latency_ms']:<12.2f} "
              f"{r['p95_latency_ms']:<12.2f} "
              f"{r['p99_latency_ms']:<12.2f} "
              f"{r['requests_per_second']:<10.1f} "
              f"{success_rate:<10.1f}%")
    
    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="延迟测试结果可视化工具"
    )
    
    parser.add_argument(
        "results_file",
        help="测试结果JSON文件路径"
    )
    
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="仅显示文本摘要，不生成图表"
    )
    
    args = parser.parse_args()
    
    try:
        if args.text_only:
            print_text_summary(args.results_file)
        else:
            plot_results(args.results_file)
            print_text_summary(args.results_file)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {args.results_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
