#!/usr/bin/env python3
"""检查当前环境：Python、PyTorch、CUDA、flash-attn 等是否可用。"""
import sys
import subprocess

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def main():
    print("=== Python ===")
    print(f"  executable: {sys.executable}")
    print(f"  version:    {sys.version.split()[0]}")

    print("\n=== PyTorch ===")
    try:
        import torch
        print(f"  version:    {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  torch CUDA:  {torch.version.cuda}")
            print(f"  device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  (CUDA 不可用，检查 LD_LIBRARY_PATH 或 PyTorch 是否为 CPU 版)")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n=== flash-attn ===")
    try:
        import flash_attn
        v = getattr(flash_attn, "__version__", "?")
        print(f"  version:    {v}")
        print("  import:    OK")
    except ImportError as e:
        print(f"  import:    FAIL - {e}")
    except Exception as e:
        print(f"  ERROR:     {e}")

    print("\n=== 其他常用包 ===")
    for name in ("transformers", "ray", "hydra_core", "omegaconf", "sglang"):
        try:
            mod = __import__(name.replace("-", "_").split("_")[0])
            v = getattr(mod, "__version__", "?")
            print(f"  {name}: {v}")
        except ImportError:
            print(f"  {name}: 未安装")

    print("\n=== 建议 ===")
    try:
        import torch
        if not torch.cuda.is_available():
            print("  - PyTorch 未检测到 CUDA，需安装带 CUDA 的 torch 并保证 libcudart 在 LD_LIBRARY_PATH")
        # 项目 sglang 依赖里写的是 torch==2.8.0
        major_minor = torch.__version__.split(".")[:2]
        if len(major_minor) >= 2:
            try:
                v = int(major_minor[0]) * 10 + int(major_minor[1])
                if v < 28:
                    print("  - 当前 torch 版本偏低，本仓库 sglang 推荐 torch==2.8.0；若被降级可重装。")
            except ValueError:
                pass
        if torch.cuda.is_available() and torch.version.cuda:
            print("  - 本机若为 CUDA 13：PyTorch 官方多为 cu124/cu126，cu124 自带运行时不依赖系统 CUDA；详见 https://pytorch.org")
    except Exception:
        pass
    print()

if __name__ == "__main__":
    main()
