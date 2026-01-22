# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import re
import traceback
import os, pickle
from math_verify import parse, verify
import tempfile
import subprocess
from contextlib import contextmanager
import signal
import ast
import numpy as np
from verl.utils.reward_score.code_judge_client import CodeJudgeClient
from typing import Optional

IMPORT_PROMPT='''from typing import *

from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *
import math
import datetime
inf = float('inf')

'''
code_judge_url = "http://localhost:8088"
code_judge_client = CodeJudgeClient(base_url=code_judge_url, max_workers=10)
logger = logging.getLogger(__name__)
livecodebench_dir = os.environ.get("LIVECODEBENCH_DATA_PATH", None)
# if livecodebench_dir is None:
#     raise ValueError("LIVECODEBENCH_DATA_PATH is not set")

def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    # while i < len(string):
    for i in range(idx+4, len(string)):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx+1] if right_brace_idx is not None else None


@contextmanager
def timeout_run(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("代码执行超时")
    
    # 注册信号处理器
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

def convert_function_to_class_method(raw_code: str, function_name: str) -> Optional[str]:
    # 解析原始代码为 AST
    try:
        tree = ast.parse(raw_code)
    except SyntaxError as exc:
        logger.warning("Failed to parse function before conversion", exc_info=exc)
        return None
    target_func = None
    new_body = []
    # 遍历顶层节点，保留非目标函数的代码
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            target_func = node
        else:
            new_body.append(node)
    
    if target_func is None:
        return None

    if not (target_func.args.args and target_func.args.args[0].arg == "self"):
        self_arg = ast.arg(arg="self", annotation=None)
        target_func.args.args.insert(0, self_arg)    
    class_def = ast.ClassDef(
        name="Solution",
        bases=[],
        keywords=[],
        body=[target_func],
        decorator_list=[]
    )
    
    new_body.append(class_def)
    tree.body = new_body
    
    # 使用 ast.unparse 将 AST 转换为代码字符串（Python 3.9+支持）
    try:
        new_code = ast.unparse(tree)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to unparse converted Solution class", exc_info=exc)
        return None
    return new_code


def math_verify_reward_function(solution_str, ground_truth):

    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    
    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return_dict ={
        "score": -1.0,
        "acc": False,
        "pred": None,
        }
        return return_dict
    
    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return_dict ={
        "score": -1.0,
        "acc": False,
        "pred": None,
        }
        return return_dict
    
    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return_dict ={
        "score": 1.0,
        "acc": True,
        "pred": math_verify_parsed[1],
        }
        return return_dict
    
    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return_dict ={
                "score": 1.0,
                "acc": True,
                "pred": math_verify_parsed[1],
                }
                return return_dict
        except Exception:
            continue

    
    # Very unlikely to be correct after the above matches
    return  {"score": -1.0, "acc": False, "pred": math_verify_parsed[1]}

def compute_score(completion, test_cases, task=None, timeout=30, is_long_penalty=False, is_binary_reward=False, is_power4_reward=False):
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    # solution = completion.split('```python')[-1].split('```')[0]

    if "</think>" in completion:
        solution_str = completion.split("</think>")[1]
    else:
        solution_str = completion

    test_cases_dict = None
    test_cases_str = None
    if isinstance(test_cases, dict):
        test_cases_dict = test_cases
        try:
            test_cases_str = json.dumps(test_cases)
        except TypeError:  # pragma: no cover - defensive fallback
            test_cases_str = str(test_cases)
    elif isinstance(test_cases, str):
        test_cases_str = test_cases
        try:
            test_cases_dict = json.loads(test_cases)
        except json.JSONDecodeError:
            try:
                test_cases_dict = ast.literal_eval(test_cases)
            except (ValueError, SyntaxError):
                test_cases_dict = None
    else:
        try:
            test_cases_str = json.dumps(test_cases)
        except TypeError:  # pragma: no cover - defensive fallback
            test_cases_str = str(test_cases)

    has_import_prefix = isinstance(test_cases_dict, dict) and "import_prefix" in test_cases_dict
    has_inputs = isinstance(test_cases_dict, dict) and "inputs" in test_cases_dict
    if not has_import_prefix and test_cases_str:
        has_import_prefix = "import_prefix" in test_cases_str
    if not has_inputs and test_cases_str:
        has_inputs = "inputs" in test_cases_str

    if has_import_prefix:
        if not isinstance(test_cases_dict, dict):
            logger.warning("Unable to parse test cases containing import_prefix")
            return {"score": -1.0, "acc": False, "pred": None}
        solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
        if len(solutions) == 0:
            return {"score": -1.0, "acc": False, "pred": None}
        try:
            solution = solutions[-1]
            try:
                ast.parse(solution)
            except SyntaxError as exc:
                logger.info("Syntax error detected before executing sandbox tests", exc_info=exc)
                return {"score": -1.0, "acc": False, "pred": None, "error": "syntax_error"}
            solution = test_cases_dict["import_prefix"] + solution
            test_code = [x for x in test_cases_dict['test_code'].split("\n") if x != ""]
            unit_test_result = []
            unit_test_metadata = []
            summary = None
            for i in range(1, len(test_code)):
                cur_solution = solution
                cur_solution += "\n" + test_code[0] + test_code[i]
                cur_solution += "\ncheck({})".format(test_cases_dict['entry_point'])
                try:
                    # 执行代码的逻辑
                    ## Use CodeJudgeClient
                    result = code_judge_client.run_single(
                        code=cur_solution,
                        stdin=None,
                        language="python",
                        timeout=timeout,
                    )
                    if result.get('run_success', False) and result.get('success', False):
                         unit_test_result.append(True)
                         unit_test_metadata.append(f"成功")
                    else:
                         unit_test_result.append(False)
                         stderr = result.get('stderr', '')
                         reason = result.get('reason', 'unknown')
                         unit_test_metadata.append(f"执行错误: {reason} - {stderr}")
                except TimeoutError:
                    print("代码执行超时")
                    traceback.print_exc(10)
                    unit_test_result.append(False)
                    unit_test_metadata.append("代码执行超时")
                except Exception as e:
                    print(f"执行异常: {str(e)}")
                    unit_test_result.append(False)
                    unit_test_metadata.append("执行异常")
                    
            sandbox_stats = summary
            if is_binary_reward:
                payload = {"score": 1.0 if all(unit_test_result) else -1.0, "acc": all(unit_test_result), "pred": solution}
            else:
                if is_power4_reward:
                    payload = {
                        "score": (sum(unit_test_result)/len(unit_test_result))**4,
                        "acc": all(unit_test_result),
                        "pred": solution,
                    }
                else:
                    payload = {
                        "score": sum(unit_test_result)/len(unit_test_result),
                        "acc": all(unit_test_result),
                        "pred": solution,
                    }
            if sandbox_stats:
                payload["sandbox_stats"] = sandbox_stats
            return payload

        except Exception as e:
            traceback.print_exc(10)
            return {"score": -1.0, "acc": False, "pred": None}

    elif has_inputs:
        if not isinstance(test_cases_dict, dict):
            logger.warning("Unable to parse test cases containing inputs")
            return {"score": -1.0, "acc": False, "pred": None}
        try:
            solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
            if len(solutions) == 0:
                return {"score": -1.0, "acc": False, "pred": None}
            else:
                solution = solutions[-1]
                try:
                    ast.parse(solution)
                except SyntaxError as exc:
                    logger.info("Syntax error detected before sandbox correctness check", exc_info=exc)
                    return {"score": -1.0, "acc": False, "pred": None, "error": "syntax_error"}

            input_output = test_cases_dict
            if "fn_name" in input_output and "class Solution" not in solution:
                converted_solution = convert_function_to_class_method(solution, input_output["fn_name"])
                if isinstance(converted_solution, str):
                    solution = converted_solution
                
             ## Use CodeJudgeClient
            # Build test cases from input_output
            test_cases = []
            inputs = input_output.get('inputs', [])
            outputs = input_output.get('outputs', [])
            for inp, out in zip(inputs, outputs):
                test_cases.append({
                    'stdin': str(inp) if inp is not None else None,
                    'expected_output': str(out) if out is not None else None,
                })
            
            result = code_judge_client.check_correctness(
                code=solution,
                test_cases=test_cases,
                language="python",
                timeout=timeout,
            )
            
            # Convert results to metrics format
            metrics = [[r.get('success', False) for r in result['results']], result['results']]
            fixed = metrics[0]

            if is_binary_reward:
                payload = {
                    "score": 1.0 if sum(metrics[0]) == len(metrics[0]) else -1.0,
                    "acc": sum(metrics[0]) == len(metrics[0]),
                    "pred": solution,
                }
            else:
                if is_power4_reward:
                    payload = {
                        "score": (sum((x if x in [False, True] else False) for x in metrics[0]) / len(metrics[0]))**4,
                        "acc": sum(metrics[0]) == len(metrics[0]),
                        "pred": solution,
                    }
                else:
                    payload = {
                        "score": sum((x if x in [False, True] else False) for x in metrics[0]) / len(metrics[0]),
                        "acc": sum(metrics[0]) == len(metrics[0]),
                        "pred": solution,
                    }
            return payload

        except Exception as e:
            traceback.print_exc(10)
            return {"score": -1.0, "acc": False, "pred": solution}
    else:
        try:
            last_boxed = last_boxed_only_string(solution_str)
            return math_verify_reward_function(last_boxed, test_cases)
        except:
            traceback.print_exc(10)
            return {"score": -1.0, "acc": False, "pred": None}    

