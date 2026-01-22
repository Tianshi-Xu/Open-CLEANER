#!/usr/bin/env python3
"""
高并发沙箱执行环境延迟测试脚本

该脚本用于测量 code-judge 沙箱在不同并发级别下的平均延迟情况，
帮助识别性能瓶颈和阻塞点。

使用方法:
    python benchmark_latency.py --host http://localhost:8000 --max-concurrency 100
"""

import argparse
import asyncio
import aiohttp
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict
import json
from collections import defaultdict
import sys


@dataclass
class LatencyStats:
    """延迟统计信息"""
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    requests_per_second: float
    total_duration: float
    
    def __str__(self):
        return (
            f"{self.concurrency:<10} "
            f"{self.total_requests:<12} "
            f"{self.successful_requests}/{self.total_requests:<15} "
            f"{self.avg_latency:>7.2f}ms    "
            f"{self.median_latency:>7.2f}ms    "
            f"{self.p95_latency:>7.2f}ms    "
            f"{self.p99_latency:>7.2f}ms    "
            f"{self.requests_per_second:>6.1f}"
        )


@dataclass
class BenchmarkConfig:
    """压力测试配置"""
    host: str = "http://localhost:8000"
    endpoint: str = "/judge"
    min_concurrency: int = 1
    max_concurrency: int = 100
    concurrency_step: int = 5
    batches_per_level: int = 10  # 每个并发级别执行的批次数（总请求数 = 并发数 × 批次数）
    timeout: int = 30
    warmup_requests: int = 10
    
    # 测试用例配置
    test_cases: List[Dict] = field(default_factory=lambda: [
        # 轻量级测试
        {
            "type": "python",
            "solution": "print(input())",
            "input": "hello",
            "expected_output": "hello"
        },
        # 中等计算量 - 0.5秒延迟 + 计算
        {
            "type": "python",
            "solution": "import time\ntime.sleep(0.5)\nprint(sum([int(x) for x in input().split()]))",
            "input": "1 2 3 4 5",
            "expected_output": "15"
        },
        # 较重的计算 - 1秒延迟 + 复杂计算
        {
            "type": "python",
            "solution": """import time
time.sleep(1.0)
n = int(input())
result = sum(i*i for i in range(n))
print(result)""",
            "input": "1000",
            "expected_output": "332833500"
        },
        # CPU密集型 - 斐波那契计算
        {
            "type": "python",
            "solution": """def fib(n):
    if n <= 1: return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b
n = int(input())
print(fib(n))""",
            "input": "10000",
            "expected_output": "33644764876431783266621612005107543310302148460680063906564769974680081442166662368155595513633734025582065332680836159373734790483865268263040892463056431887354544369559827491606602099884183933864652731300088830269235673613135117579297437854413752130520504347701602264758318906527890855154366159582987279682987510631200575428783453215515103870818298969791613127856265033195487140214287532698187962046936097879900350962302291026368131493195275630227837628441540360584402572114334961180023091208287046088923962328835461505776583271252546093591128203925285393434620904245248929403901706233888991085841065183173360437470737908552631764325733993712871937587746897479926305837065742830161637408969178426378624212835258112820516370298089332099905707920064367426202389783111470054074998459250360633560933883831923386783056136435351892133279732908133732642652633989763922723407882928177953580570993691049175470808931841056146322338217465637321248226383092103297701648054726243842374862411453093812206564914032751086643394517512161526545361333111314042436854805106765843493523836959653428071768775328348234345557366719731392746273629108210679280784718035329131176778924659089938635459327894523777674406192240337638674004021330343297496902028328145933418826817683893072003634795623117103101291953169794607632737589253530772552375943788434504067715555779056450443016640119462580972216729758615026968443146952034614932291105970676243268515992834709891284706740862008587135016260312071903172086094081298321581077282076353186624611278245537208532365305775956430072517744315051539600905168603220349163222640885248852433158051534849622434848299380905070483482449327453732624567755879089187190803662058009594743150052402532709746995318770724376825907419939632265984147498193609285223945039707165443156421328157688908058783183404917434556270520223564846495196112460268313970975069382648706613264507665074611512677522748621598642530711298441182622661057163515069260029861704945425047491378115154139941550671256271197133252763631939606902895650288268608362241082050562430701794976171121233066073310059947366875"
        },
        # IO密集型 - 文件操作模拟
        {
            "type": "python",
            "solution": """import time
# 模拟IO操作
time.sleep(0.3)
data = input().split(',')
result = ','.join(sorted(data))
print(result)""",
            "input": "z,a,m,b,x,c",
            "expected_output": "a,b,c,m,x,z"
        },
    ])


class LatencyBenchmark:
    """延迟压力测试类"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[LatencyStats] = []
        
    async def send_request(self, session: aiohttp.ClientSession, test_case: Dict) -> tuple[float, bool]:
        """
        发送单个请求并测量延迟
        
        返回: (延迟(毫秒), 是否成功)
        """
        url = f"{self.config.host}{self.config.endpoint}"
        start_time = time.perf_counter()
        
        try:
            async with session.post(
                url,
                json=test_case,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                await response.read()
                latency = (time.perf_counter() - start_time) * 1000  # 转换为毫秒
                success = response.status == 200
                return latency, success
        except asyncio.TimeoutError:
            latency = (time.perf_counter() - start_time) * 1000
            print(f"⚠️  请求超时 (>{self.config.timeout}s)", file=sys.stderr)
            return latency, False
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            print(f"❌ 请求失败: {e}", file=sys.stderr)
            return latency, False
    
    async def run_concurrent_requests(self, concurrency: int, num_requests: int) -> List[tuple[float, bool]]:
        """
        运行指定并发级别的请求
        
        Args:
            concurrency: 并发数
            num_requests: 总请求数
            
        Returns:
            包含所有请求结果的列表: [(延迟, 是否成功), ...]
        """
        results = []
        
        # 创建连接池
        connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # 创建任务队列
            tasks = []
            test_case_index = 0
            
            for i in range(num_requests):
                test_case = self.config.test_cases[test_case_index % len(self.config.test_cases)]
                task = self.send_request(session, test_case)
                tasks.append(task)
                test_case_index += 1
                
                # 控制并发数
                if len(tasks) >= concurrency:
                    batch_results = await asyncio.gather(*tasks)
                    results.extend(batch_results)
                    tasks = []
            
            # 处理剩余任务
            if tasks:
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
        
        return results
    
    def calculate_stats(self, concurrency: int, results: List[tuple[float, bool]], duration: float) -> LatencyStats:
        """
        计算统计信息
        
        Args:
            concurrency: 并发数
            results: 请求结果列表
            duration: 总耗时(秒)
            
        Returns:
            LatencyStats 对象
        """
        latencies = [lat for lat, _ in results]
        successes = [success for _, success in results]
        
        successful_requests = sum(successes)
        failed_requests = len(successes) - successful_requests
        
        if not latencies:
            return LatencyStats(
                concurrency=concurrency,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_latency=0,
                median_latency=0,
                p95_latency=0,
                p99_latency=0,
                min_latency=0,
                max_latency=0,
                requests_per_second=0,
                total_duration=duration
            )
        
        sorted_latencies = sorted(latencies)
        
        return LatencyStats(
            concurrency=concurrency,
            total_requests=len(results),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_latency=statistics.mean(latencies),
            median_latency=statistics.median(latencies),
            p95_latency=sorted_latencies[int(len(sorted_latencies) * 0.95)],
            p99_latency=sorted_latencies[int(len(sorted_latencies) * 0.99)],
            min_latency=min(latencies),
            max_latency=max(latencies),
            requests_per_second=len(results) / duration if duration > 0 else 0,
            total_duration=duration
        )
    
    async def warmup(self):
        """预热服务"""
        print(f"🔥 预热中... (发送 {self.config.warmup_requests} 个请求)")
        connector = aiohttp.TCPConnector(limit=5)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for i in range(self.config.warmup_requests):
                test_case = self.config.test_cases[i % len(self.config.test_cases)]
                tasks.append(self.send_request(session, test_case))
            await asyncio.gather(*tasks)
        print("✅ 预热完成\n")
    
    async def run_benchmark(self):
        """运行完整的压力测试"""
        print("=" * 120)
        print("🚀 开始高并发延迟压力测试")
        print(f"📍 目标服务: {self.config.host}{self.config.endpoint}")
        print(f"📊 测试范围: 并发数 {self.config.min_concurrency} ~ {self.config.max_concurrency} (步长: {self.config.concurrency_step})")
        print(f"📦 每级批次数: {self.config.batches_per_level} (总请求数 = 并发数 × {self.config.batches_per_level})")
        print("=" * 120)
        print()
        
        # 预热
        await self.warmup()
        
        # 测试不同并发级别
        concurrency_levels = range(
            self.config.min_concurrency,
            self.config.max_concurrency + 1,
            self.config.concurrency_step
        )
        
        print(f"{'并发数':<10} {'总请求':<12} {'成功/总数':<15} {'平均延迟':<12} {'中位数':<12} {'P95':<12} {'P99':<12} {'QPS':<10}")
        print("-" * 120)
        
        for concurrency in concurrency_levels:
            # 计算该并发级别的总请求数（并发数 × 批次数）
            num_requests = concurrency * self.config.batches_per_level
            
            start_time = time.perf_counter()
            results = await self.run_concurrent_requests(concurrency, num_requests)
            duration = time.perf_counter() - start_time
            
            stats = self.calculate_stats(concurrency, results, duration)
            self.results.append(stats)
            
            print(stats)
            
            # 如果失败率过高，停止测试
            if stats.failed_requests / stats.total_requests > 0.5:
                print(f"\n⚠️  警告: 失败率超过50%，停止测试")
                break
            
            # 短暂延迟，避免过度压力
            await asyncio.sleep(0.5)
        
        print("\n" + "=" * 120)
        self.print_summary()
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n📊 测试摘要")
        print("=" * 120)
        
        if not self.results:
            print("没有收集到测试数据")
            return
        
        # 找出性能拐点
        print("\n🔍 性能分析:")
        
        baseline_latency = self.results[0].avg_latency
        significant_degradation_found = False
        
        for i, stats in enumerate(self.results):
            if i == 0:
                continue
            
            latency_increase = (stats.avg_latency - baseline_latency) / baseline_latency * 100
            
            # 如果延迟增加超过50%，认为出现显著阻塞
            if latency_increase > 50 and not significant_degradation_found:
                print(f"\n⚠️  显著性能下降点: 并发数 {stats.concurrency}")
                print(f"   - 平均延迟从 {baseline_latency:.2f}ms 增加到 {stats.avg_latency:.2f}ms (+{latency_increase:.1f}%)")
                print(f"   - P99延迟: {stats.p99_latency:.2f}ms")
                print(f"   - 建议最大并发数: {self.results[i-1].concurrency}")
                significant_degradation_found = True
        
        if not significant_degradation_found:
            best_qps_stats = max(self.results, key=lambda s: s.requests_per_second)
            print(f"\n✅ 在测试范围内未发现显著阻塞")
            print(f"   - 最佳性能点: 并发数 {best_qps_stats.concurrency}, QPS: {best_qps_stats.requests_per_second:.1f}")
        
        # 最佳性能点
        print(f"\n📈 性能指标:")
        best_qps = max(self.results, key=lambda s: s.requests_per_second)
        print(f"   - 最高QPS: {best_qps.requests_per_second:.1f} (并发数: {best_qps.concurrency})")
        
        lowest_latency = min(self.results, key=lambda s: s.avg_latency)
        print(f"   - 最低平均延迟: {lowest_latency.avg_latency:.2f}ms (并发数: {lowest_latency.concurrency})")
        
        highest_latency = max(self.results, key=lambda s: s.avg_latency)
        print(f"   - 最高平均延迟: {highest_latency.avg_latency:.2f}ms (并发数: {highest_latency.concurrency})")
        
        print("\n" + "=" * 120)
    
    def export_results(self, filename: str = "latency_benchmark_results.json"):
        """导出结果到JSON文件"""
        data = {
            "config": {
                "host": self.config.host,
                "min_concurrency": self.config.min_concurrency,
                "max_concurrency": self.config.max_concurrency,
                "batches_per_level": self.config.batches_per_level,
            },
            "results": [
                {
                    "concurrency": s.concurrency,
                    "total_requests": s.total_requests,
                    "successful_requests": s.successful_requests,
                    "failed_requests": s.failed_requests,
                    "avg_latency_ms": s.avg_latency,
                    "median_latency_ms": s.median_latency,
                    "p95_latency_ms": s.p95_latency,
                    "p99_latency_ms": s.p99_latency,
                    "min_latency_ms": s.min_latency,
                    "max_latency_ms": s.max_latency,
                    "requests_per_second": s.requests_per_second,
                    "total_duration_s": s.total_duration,
                }
                for s in self.results
            ]
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已导出到: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="高并发沙箱执行环境延迟测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本测试 (并发1-100，步长5，每级批次数10)
  # 例如并发10时发送10×10=100个请求，并发50时发送50×10=500个请求
  python benchmark_latency.py
  
  # 自定义测试范围和批次数
  python benchmark_latency.py --min-concurrency 10 --max-concurrency 200 --step 10 --batches-per-level 20
  
  # 增加批次数以获得更准确的结果（但会增加测试时间）
  python benchmark_latency.py --batches-per-level 50
  
  # 测试远程服务器
  python benchmark_latency.py --host http://192.168.1.100:8000
  
注意: 总请求数 = 并发数 × 批次数
      例如: 并发50，批次10，总共会发送 50×10=500 个请求
        """
    )
    
    parser.add_argument(
        "--host",
        default="http://localhost:8088",
        help="judge 服务地址 (默认: http://localhost:8088)"
    )
    
    parser.add_argument(
        "--endpoint",
        default="/judge",
        help="API端点 (默认: /judge)"
    )
    
    parser.add_argument(
        "--min-concurrency",
        type=int,
        default=1,
        help="最小并发数 (默认: 1)"
    )
    
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=100,
        help="最大并发数 (默认: 100)"
    )
    
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="并发数步长 (默认: 5)"
    )
    
    parser.add_argument(
        "--batches-per-level",
        type=int,
        default=10,
        help="每个并发级别的批次数，总请求数=并发数×批次数 (默认: 10)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="单个请求超时时间(秒) (默认: 30)"
    )
    
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="预热请求数 (默认: 10)"
    )
    
    parser.add_argument(
        "--output",
        default="latency_benchmark_results.json",
        help="结果输出文件 (默认: latency_benchmark_results.json)"
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = BenchmarkConfig(
        host=args.host,
        endpoint=args.endpoint,
        min_concurrency=args.min_concurrency,
        max_concurrency=args.max_concurrency,
        concurrency_step=args.step,
        batches_per_level=args.batches_per_level,
        timeout=args.timeout,
        warmup_requests=args.warmup,
    )
    
    # 运行测试
    benchmark = LatencyBenchmark(config)
    
    try:
        asyncio.run(benchmark.run_benchmark())
        benchmark.export_results(args.output)
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        if benchmark.results:
            benchmark.print_summary()
            benchmark.export_results(args.output)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
