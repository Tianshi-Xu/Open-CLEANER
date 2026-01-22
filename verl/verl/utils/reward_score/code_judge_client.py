# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""
Client for calling code-judge service API for reward calculation.
This module provides synchronous interface to call code-judge service.
"""

import json
import logging
import requests
from typing import Optional, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class CodeJudgeClient:
    """Client for code-judge service API."""
    
    def __init__(self, base_url: str = "http://localhost:8088", max_workers: int = 10):
        """
        Initialize CodeJudgeClient.
        
        Args:
            base_url: Base URL of the code-judge service
            max_workers: Maximum number of concurrent workers for batch execution
        """
        self.base_url = base_url.rstrip('/')
        self.max_workers = max_workers
    
    def run_single(
        self,
        code: str,
        stdin: Optional[str] = None,
        expected_output: Optional[str] = None,
        language: str = "python",
        timeout: int = 30,
    ) -> dict[str, Any]:
        """
        Execute a single code submission.
        
        Args:
            code: The code to execute
            stdin: Standard input for the code
            expected_output: Expected output for verification
            language: Programming language (default: "python")
            timeout: Request timeout in seconds
            
        Returns:
            Response dict with keys:
                - success: bool - Whether output matches expected_output
                - run_success: bool - Whether code ran without errors
                - stdout: str - Standard output
                - stderr: str - Standard error
                - cost: float - Execution time
                - reason: str - Error reason if any
        """
        url = f"{self.base_url}/run"
        payload = {
            "type": language,
            "solution": code,
            "input": stdin if stdin else "",
        }
        
        if expected_output is not None:
            payload["expected_output"] = expected_output
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Code-judge API request failed: {e}")
            return {
                "success": False,
                "run_success": False,
                "stdout": None,
                "stderr": None,
                "cost": 0.0,
                "reason": f"api_error: {str(e)}",
            }
    
    def run_batch(
        self,
        submissions: List[dict[str, Any]],
        language: str = "python",
        timeout: int = 60,
    ) -> List[dict[str, Any]]:
        """
        Execute multiple code submissions in batch.
        
        Args:
            submissions: List of submission dicts, each containing:
                - code: str - The code to execute
                - stdin: Optional[str] - Standard input
                - expected_output: Optional[str] - Expected output
            language: Programming language (default: "python")
            timeout: Request timeout in seconds
            
        Returns:
            List of response dicts (same format as run_single)
        """
        url = f"{self.base_url}/run/batch"
        
        batch_submissions = []
        for sub in submissions:
            payload = {
                "type": language,
                "solution": sub["code"],
                "input": sub.get("stdin", "") if sub.get("stdin") else "",
            }
            if sub.get("expected_output") is not None:
                payload["expected_output"] = sub["expected_output"]
            batch_submissions.append(payload)
        
        batch_payload = {
            "type": "batch",
            "submissions": batch_submissions,
        }
        
        try:
            response = requests.post(
                url,
                json=batch_payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("results", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Code-judge batch API request failed: {e}")
            # Return error results for all submissions
            error_result = {
                "success": False,
                "run_success": False,
                "stdout": None,
                "stderr": None,
                "cost": 0.0,
                "reason": f"api_error: {str(e)}",
            }
            return [error_result] * len(submissions)
    
    def check_correctness(
        self,
        code: str,
        test_cases: List[dict[str, Any]],
        language: str = "python",
        timeout: int = 30,
    ) -> dict[str, Any]:
        """
        Check code correctness against multiple test cases.
        
        Args:
            code: The code to test
            test_cases: List of test case dicts, each containing:
                - stdin: Optional[str] - Standard input
                - expected_output: Optional[str] - Expected output
            language: Programming language (default: "python")
            timeout: Timeout for each test case execution
            
        Returns:
            Dict with:
                - passed: int - Number of test cases passed
                - total: int - Total number of test cases
                - results: List[dict] - Individual test case results
                - all_passed: bool - Whether all test cases passed
        """
        submissions = []
        for test_case in test_cases:
            submissions.append({
                "code": code,
                "stdin": test_case.get("stdin"),
                "expected_output": test_case.get("expected_output"),
            })
        
        # Use ThreadPoolExecutor for concurrent execution
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.run_single,
                    code=sub["code"],
                    stdin=sub.get("stdin"),
                    expected_output=sub.get("expected_output"),
                    language=language,
                    timeout=timeout,
                ): i for i, sub in enumerate(submissions)
            }
            
            # Collect results in order
            result_dict = {}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result_dict[idx] = future.result()
                except Exception as e:
                    logger.error(f"Test case {idx} execution failed: {e}")
                    result_dict[idx] = {
                        "success": False,
                        "run_success": False,
                        "stdout": None,
                        "stderr": None,
                        "cost": 0.0,
                        "reason": f"execution_error: {str(e)}",
                    }
            
            # Sort by index
            results = [result_dict[i] for i in range(len(submissions))]
        
        passed = sum(1 for r in results if r.get("success", False))
        total = len(results)
        
        return {
            "passed": passed,
            "total": total,
            "results": results,
            "all_passed": passed == total,
        }
