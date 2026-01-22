from contextlib import contextmanager
import tempfile
import io
import shlex
import ast
from typing import Set

from .executor import ScriptExecutor, ProcessExecuteResult, TIMEOUT_EXIT_CODE

SCRIPT_ENDING_MARK = "@@E"
DURATION_MARK = "@@D"

# 可用的库映射：检测到的名称 -> 导入语句
AVAILABLE_IMPORTS = {
    'math': 'import math',
    'itertools': 'import itertools',
    'collections': 'import collections',
    'np': 'import numpy as np',
    'numpy': 'import numpy',
    'sp': 'import sympy as sp',
    'sympy': 'import sympy',
    'symbols': 'from sympy import symbols',
}

# typing 总是导入（轻量级）
# from __future__ import annotations 必须在文件最开头，以避免前向引用问题
ALWAYS_IMPORT = ['from __future__ import annotations', 'from typing import *']


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to detect which names are used in the code"""
    
    def __init__(self):
        self.used_names: Set[str] = set()
    
    def visit_Name(self, node):
        self.used_names.add(node.id)
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            self.used_names.add(node.value.id)
        self.generic_visit(node)


def analyze_required_imports(script: str) -> Set[str]:
    """分析用户代码，返回需要导入的语句集合"""
    required_imports = set()
    
    try:
        tree = ast.parse(script)
        analyzer = ImportAnalyzer()
        analyzer.visit(tree)
        
        for name in analyzer.used_names:
            if name in AVAILABLE_IMPORTS:
                required_imports.add(AVAILABLE_IMPORTS[name])
            
    except SyntaxError:
        pass
    
    return required_imports


# 提前计算 TIMEOUT_EXIT_CODE 的值，避免在模板字符串中混用
_TIMEOUT_EXIT_CODE_STR = str(TIMEOUT_EXIT_CODE)

PRE_TEMPLATE_BASE = f"""
{{imports}}

def _exec_prepare():
    import signal
    import resource
    import os
    import time

    # preventing multi-threading for numpy
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    def _exec_set_alarm_timeout(timeout):
        signal.signal(signal.SIGALRM, _exec_time_exceeded)
        signal.alarm(timeout)

    # checking time limit exceed
    def _exec_time_exceeded(*_):
        print('Suicide from timeout.', flush=True)
        try:
            os.killpg(0, signal.SIGKILL)  # sometime this can still fail
        except Exception:
            pass
        try:
            os.kill(0, signal.SIGKILL)  # sometime this can still fail
        except Exception:
            pass
        os._exit({_TIMEOUT_EXIT_CODE_STR})  # may not run here.

    def _exec_set_max_runtime(seconds):
        # setting up the resource limit
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, hard))
        # Just use its default behavior to terminate the process.
        # signal.signal(signal.SIGXCPU, _exec_time_exceeded)

    def _exec_limit_memory(maxsize):
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if {{{{timeout}}}}:
        _exec_set_alarm_timeout({{{{timeout}}}})
        _exec_set_max_runtime({{{{timeout}}}})

    if {{{{memory_limit}}}}:
        _exec_limit_memory({{{{memory_limit}}}})

    return time.perf_counter()

_exec_time_start = _exec_prepare()

""".strip()

POST_TEMPLATE = f"""

def _exec_end():
    import time
    _exec_time_end = time.perf_counter()
    _exec_duration = _exec_time_end - _exec_time_start
    print("{SCRIPT_ENDING_MARK}")
    print(f"{DURATION_MARK}{{_exec_duration}}", flush=True)

_exec_end()

""".strip()

class PythonExecutor(ScriptExecutor):
    def __init__(self, run_cl: str, timeout: int = None, memory_limit: int = None):
        self.timeout = timeout
        self.memory_limit = (
            memory_limit + 256 * 1024 * 1024  # extra 256MB for python overhead
            if memory_limit
            else None
        )
        self.run_cl = run_cl

    def setup_command(self, tmp_path: str, script: str):
        source_path = f"{tmp_path}/source.py"
        
        # 分析用户代码，获取需要的导入语句
        required_imports = analyze_required_imports(script)
        all_imports = ALWAYS_IMPORT + sorted(required_imports)
        imports_str = '\n'.join(all_imports)
        
        # 生成完整的前置模板
        pre_template = PRE_TEMPLATE_BASE.format(imports=imports_str)
        
        with open(source_path, mode='w') as f:
            f.write(pre_template.format(timeout=self.timeout, memory_limit=self.memory_limit))
            f.write("\n")
            f.write(script)
            f.write("\n")
            f.write(POST_TEMPLATE)
            f.flush()
        yield shlex.split(self.run_cl.format(
            source=shlex.quote(source_path),
            workdir=shlex.quote(str(tmp_path))
        ))

    def process_result(self, result):
        if SCRIPT_ENDING_MARK in result.stdout:
            result.stdout, meta_info = result.stdout.split(SCRIPT_ENDING_MARK, 2)
            for line in io.StringIO(meta_info):
                if line.startswith(DURATION_MARK):
                    result.cost = float(line[len(DURATION_MARK):])
                    break
        return result
