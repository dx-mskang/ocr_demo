#!/usr/bin/env python3
"""
OCR API Server Benchmark Runner (Worker Model - 标准生产者-消费者模式)
运行 API 基准测试并生成 Markdown 报告

架构设计 (Producer → N Workers → Collector):
- Producer: 构建任务队列，放入 sentinel 通知 worker 退出
- Workers (N个): 从队列取任务，执行请求，将结果放入结果队列
- Collector: 从结果队列收集结果，统计聚合

优点:
- 退出条件可靠：task_queue.join() + sentinel 保证所有任务被处理
- 不会卡死：即便请求异常也会产出 RequestResult
- 并发硬约束：worker 数量就是最大并发
- 指标口径清晰：按请求粒度统计，再按 filename 聚合
"""

import sys
import json
import argparse
import time
import base64
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime
from statistics import mean, stdev
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import difflib
import traceback

# 默认测试图片 (1x1 红色 PNG)
DEFAULT_TEST_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="


# ==================== 数据结构定义 ====================

class RequestStatus(Enum):
    """请求状态枚举"""
    PENDING = "pending"      # 已发送，等待响应
    COMPLETED = "completed"  # 成功完成
    TIMEOUT = "timeout"      # 超时
    ERROR = "error"          # 错误


@dataclass
class TaskInfo:
    """任务信息"""
    request_id: int
    filename: str
    base64_data: str
    run_idx: int
    send_time: float = 0.0


@dataclass
class RequestResult:
    """请求结果"""
    request_id: int
    filename: str
    run_idx: int
    status: RequestStatus
    latency_ms: float = 0.0
    http_code: int = 0
    error_msg: str = ""
    text: str = ""
    char_count: int = 0
    ocr_results: List[Any] = field(default_factory=list)


# ==================== 结果收集器 ====================

class ResultsCollector:
    """
    结果收集器
    
    职责:
    - 收集所有请求的结果
    - 按 filename 聚合多次运行的结果
    - 提供统计分析
    """
    
    def __init__(self):
        self._results: Dict[int, RequestResult] = {}  # request_id -> result
        self._lock = asyncio.Lock()
        
        # 统计
        self.success_count = 0
        self.error_count = 0
        self.timeout_count = 0
    
    async def add_result(self, result: RequestResult):
        """添加结果（通用接口）"""
        async with self._lock:
            self._results[result.request_id] = result
            if result.status == RequestStatus.COMPLETED:
                self.success_count += 1
            elif result.status == RequestStatus.TIMEOUT:
                self.timeout_count += 1
            else:
                self.error_count += 1
    
    async def add_success(self, request_id: int, filename: str, run_idx: int,
                          latency_ms: float, http_code: int, text: str,
                          char_count: int, ocr_results: List[Any]):
        """添加成功结果"""
        async with self._lock:
            self._results[request_id] = RequestResult(
                request_id=request_id,
                filename=filename,
                run_idx=run_idx,
                status=RequestStatus.COMPLETED,
                latency_ms=latency_ms,
                http_code=http_code,
                text=text,
                char_count=char_count,
                ocr_results=ocr_results
            )
            self.success_count += 1
    
    async def add_error(self, request_id: int, filename: str, run_idx: int,
                        error_msg: str, http_code: int = 0, latency_ms: float = 0):
        """添加错误结果"""
        async with self._lock:
            self._results[request_id] = RequestResult(
                request_id=request_id,
                filename=filename,
                run_idx=run_idx,
                status=RequestStatus.ERROR,
                latency_ms=latency_ms,
                http_code=http_code,
                error_msg=error_msg
            )
            self.error_count += 1
    
    async def add_timeout(self, request_id: int, filename: str, run_idx: int,
                          latency_ms: float = 0):
        """添加超时结果"""
        async with self._lock:
            self._results[request_id] = RequestResult(
                request_id=request_id,
                filename=filename,
                run_idx=run_idx,
                status=RequestStatus.TIMEOUT,
                latency_ms=latency_ms,
                error_msg="Request timeout"
            )
            self.timeout_count += 1
    
    def get_all_results(self) -> List[RequestResult]:
        """获取所有结果"""
        return list(self._results.values())
    
    def get_results_by_filename(self) -> Dict[str, List[RequestResult]]:
        """按 filename 分组结果"""
        grouped = {}
        for result in self._results.values():
            if result.filename not in grouped:
                grouped[result.filename] = []
            grouped[result.filename].append(result)
        return grouped


# ==================== 辅助函数 ====================

def load_images_from_directory(images_dir: str) -> list:
    """从目录加载图片为 Base64"""
    images = []
    path = Path(images_dir)
    
    if not path.exists():
        print(f"Warning: Directory not found: {images_dir}")
        return images
    
    for image_file in sorted(path.glob("*.png")) + sorted(path.glob("*.jpg")) + sorted(path.glob("*.jpeg")):
        try:
            with open(image_file, "rb") as f:
                image_data = f.read()
                base64_data = base64.b64encode(image_data).decode("utf-8")
                images.append({
                    "filename": image_file.name,
                    "base64": base64_data
                })
                print(f"Loaded: {image_file.name}")
        except Exception as e:
            print(f"Error loading {image_file}: {e}")
    
    return images


def load_ground_truth(images_dir: str) -> dict:
    """加载 ground truth 标签"""
    labels_path = Path(images_dir) / "labels.json"
    if not labels_path.exists():
        return {}
    
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            raw_labels = json.load(f)
        
        ground_truth = {}
        for filename, boxes in raw_labels.items():
            if isinstance(boxes, list):
                texts = [box.get("text", "") for box in boxes if isinstance(box, dict)]
                ground_truth[filename] = "".join(texts)
            elif isinstance(boxes, str):
                ground_truth[filename] = boxes
        
        return ground_truth
    except Exception as e:
        print(f"Warning: Failed to load labels.json: {e}")
        return {}


def calculate_char_accuracy(predicted: str, ground_truth: str) -> float:
    """计算字符级准确率"""
    if not ground_truth:
        return None
    if not predicted:
        return 0.0
    
    matcher = difflib.SequenceMatcher(None, predicted, ground_truth)
    return matcher.ratio() * 100


# ==================== HTTP 请求函数 ====================

async def send_single_request(session: aiohttp.ClientSession, url: str, token: str,
                               task_info: TaskInfo, params: dict = None) -> RequestResult:
    """
    发送单个 OCR 请求并返回结果
    
    注意：该函数内部处理所有异常，保证一定返回 RequestResult
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"token {token}"
    }
    
    payload = {
        "file": task_info.base64_data,
        "fileType": 1,
        "useDocOrientationClassify": params.get("useDocOrientationClassify", False) if params else False,
        "useDocUnwarping": params.get("useDocUnwarping", False) if params else False,
        "textDetThresh": params.get("textDetThresh", 0.3) if params else 0.3,
        "textDetBoxThresh": params.get("textDetBoxThresh", 0.6) if params else 0.6,
        "textDetUnclipRatio": params.get("textDetUnclipRatio", 1.5) if params else 1.5,
        "textRecScoreThresh": params.get("textRecScoreThresh", 0.0) if params else 0.0,
        "visualize": params.get("visualize", False) if params else False,
    }
    
    start_time = time.time()
    
    try:
        async with session.post(url, headers=headers, json=payload,
                                timeout=aiohttp.ClientTimeout(total=60)) as response:
            latency = (time.time() - start_time) * 1000
            
            if response.status == 200:
                try:
                    json_response = await response.json()
                    if json_response.get("errorCode", -1) != 0:
                        return RequestResult(
                            request_id=task_info.request_id,
                            filename=task_info.filename,
                            run_idx=task_info.run_idx,
                            status=RequestStatus.ERROR,
                            latency_ms=latency,
                            http_code=response.status,
                            error_msg=json_response.get("errorMsg", "Unknown error")
                        )
                    
                    ocr_results = json_response.get("result", {}).get("ocrResults", [])
                    texts = [r.get("prunedResult", "") for r in ocr_results]
                    text = "".join(texts)
                    
                    return RequestResult(
                        request_id=task_info.request_id,
                        filename=task_info.filename,
                        run_idx=task_info.run_idx,
                        status=RequestStatus.COMPLETED,
                        latency_ms=latency,
                        http_code=response.status,
                        text=text,
                        char_count=len(text),
                        ocr_results=ocr_results
                    )
                    
                except json.JSONDecodeError:
                    return RequestResult(
                        request_id=task_info.request_id,
                        filename=task_info.filename,
                        run_idx=task_info.run_idx,
                        status=RequestStatus.ERROR,
                        latency_ms=latency,
                        http_code=response.status,
                        error_msg="Invalid JSON response"
                    )
            else:
                return RequestResult(
                    request_id=task_info.request_id,
                    filename=task_info.filename,
                    run_idx=task_info.run_idx,
                    status=RequestStatus.ERROR,
                    latency_ms=latency,
                    http_code=response.status,
                    error_msg=f"HTTP {response.status}"
                )
                
    except asyncio.TimeoutError:
        return RequestResult(
            request_id=task_info.request_id,
            filename=task_info.filename,
            run_idx=task_info.run_idx,
            status=RequestStatus.TIMEOUT,
            latency_ms=(time.time() - start_time) * 1000,
            error_msg="Request timeout"
        )
    except asyncio.CancelledError:
        return RequestResult(
            request_id=task_info.request_id,
            filename=task_info.filename,
            run_idx=task_info.run_idx,
            status=RequestStatus.TIMEOUT,
            latency_ms=(time.time() - start_time) * 1000,
            error_msg="Request cancelled"
        )
    except aiohttp.ClientError as e:
        return RequestResult(
            request_id=task_info.request_id,
            filename=task_info.filename,
            run_idx=task_info.run_idx,
            status=RequestStatus.ERROR,
            latency_ms=(time.time() - start_time) * 1000,
            error_msg=str(e)
        )
    except Exception as e:
        return RequestResult(
            request_id=task_info.request_id,
            filename=task_info.filename,
            run_idx=task_info.run_idx,
            status=RequestStatus.ERROR,
            latency_ms=(time.time() - start_time) * 1000,
            error_msg=f"Unexpected error: {e}"
        )


# ==================== Worker 模型核心协程 ====================

async def worker_loop(
    worker_id: int,
    task_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    url: str,
    token: str,
    params: dict,
):
    """
    Worker 协程
    
    职责:
    - 从 task_queue 取任务
    - 执行 HTTP 请求
    - 将结果放入 result_queue
    - 遇到 sentinel (None) 时退出
    """
    processed = 0
    while True:
        task_info = await task_queue.get()
        try:
            if task_info is None:  # sentinel，退出信号
                return processed
            
            # 执行请求（send_single_request 内部处理所有异常）
            result = await send_single_request(session, url, token, task_info, params)
            await result_queue.put(result)
            processed += 1
            
        except Exception as e:
            # 即使有意外异常，也要产出一个错误结果
            print(f"[Worker-{worker_id}] Unexpected error: {e}")
            traceback.print_exc()
            if task_info is not None:
                await result_queue.put(RequestResult(
                    request_id=task_info.request_id,
                    filename=task_info.filename,
                    run_idx=task_info.run_idx,
                    status=RequestStatus.ERROR,
                    error_msg=f"Worker error: {e}"
                ))
                processed += 1
        finally:
            task_queue.task_done()
    
    return processed


async def collector_loop(
    result_queue: asyncio.Queue,
    results_collector: ResultsCollector,
    expected_results: int,
    progress_every: int = 10,
):
    """
    Collector 协程
    
    职责:
    - 从 result_queue 取结果
    - 写入 ResultsCollector
    - 进度打印
    - 收齐预期数量后退出
    """
    received = 0
    while received < expected_results:
        result = await result_queue.get()
        try:
            await results_collector.add_result(result)
            received += 1
            
            if progress_every and received % progress_every == 0:
                print(f"\r[Progress] Received: {received}/{expected_results}", end="", flush=True)
                
        finally:
            result_queue.task_done()
    
    print()  # 换行
    return received


# ==================== 统计聚合函数 ====================

def build_stats_from_results(
    results_collector: ResultsCollector,
    images: list,
    ground_truth: dict,
    wall_ms: float,
    runs_per_image: int,
    concurrency: int,
) -> dict:
    """
    从 ResultsCollector 构建统计结果
    
    统一统计口径:
    - per-request 指标：latency 分位数、real_qps
    - per-image 指标：avg_latency（每图多次运行的均值）、accuracy
    """
    all_results = results_collector.get_all_results()
    results_by_filename = results_collector.get_results_by_filename()
    total_tasks = len(images) * runs_per_image
    
    # 按图片聚合结果
    image_results = []
    total_chars = 0
    successful_images = 0
    
    for image in images:
        filename = image["filename"] if isinstance(image, dict) else "unknown"
        img_results = results_by_filename.get(filename, [])
        
        # 筛选成功的结果
        success_results = [r for r in img_results if r.status == RequestStatus.COMPLETED]
        
        if success_results:
            latencies = [r.latency_ms for r in success_results]
            avg_latency = mean(latencies)
            
            # 取第一个成功结果的数据
            first_success = success_results[0]
            char_count = first_success.char_count
            text = first_success.text
            
            fps = 1000.0 / avg_latency if avg_latency > 0 else 0
            cps = char_count * 1000.0 / avg_latency if avg_latency > 0 else 0
            
            # 计算准确率
            accuracy = None
            if filename in ground_truth:
                accuracy = calculate_char_accuracy(text, ground_truth[filename])
            
            image_results.append({
                "filename": filename,
                "latency_ms": avg_latency,
                "fps": fps,
                "cps": cps,
                "char_count": char_count,
                "accuracy": accuracy,
                "text": text,
                "success_runs": len(success_results),
                "total_runs": len(img_results)
            })
            
            total_chars += char_count
            successful_images += 1
            
            acc_str = f"{accuracy:.2f}%" if accuracy is not None else "N/A"
            print(f"  {filename}: {avg_latency:.2f}ms, {char_count} chars, CPS={cps:.2f}, Acc={acc_str}")
        else:
            print(f"  {filename}: FAILED (no successful runs)")
            image_results.append({
                "filename": filename,
                "latency_ms": 0,
                "fps": 0,
                "cps": 0,
                "char_count": 0,
                "accuracy": None,
                "text": "",
                "success_runs": 0,
                "total_runs": len(img_results)
            })
    
    # 计算汇总统计（per-image 口径）
    valid_results = [r for r in image_results if r["latency_ms"] > 0]
    
    if valid_results:
        avg_latency = mean([r["latency_ms"] for r in valid_results])
        avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0
        avg_cps = total_chars * 1000.0 / wall_ms if wall_ms > 0 else 0
        
        accuracies = [r["accuracy"] for r in valid_results if r["accuracy"] is not None]
        avg_accuracy = mean(accuracies) if accuracies else None
    else:
        avg_latency = avg_fps = avg_cps = 0
        avg_accuracy = None
    
    # 真实 QPS（吞吐量，per-request 口径）
    real_qps = results_collector.success_count * 1000.0 / wall_ms if wall_ms > 0 else 0
    
    # 延迟分布（per-request 口径）
    success_latencies = sorted([r.latency_ms for r in all_results if r.status == RequestStatus.COMPLETED])
    
    stats = {
        "total_images": len(images),
        "successful_images": successful_images,
        "failed_images": len(images) - successful_images,
        "success_rate": successful_images * 100.0 / len(images) if images else 0,
        "total_time_ms": wall_ms,
        "wall_clock_time_ms": wall_ms,
        "total_chars": total_chars,
        "avg_latency_ms": avg_latency,
        "avg_fps": avg_fps,
        "avg_cps": avg_cps,
        "real_qps": real_qps,
        "avg_accuracy": avg_accuracy,
        "image_results": image_results,
        "runs_per_image": runs_per_image,
        "max_in_flight": concurrency,
        # 详细统计
        "total_requests": total_tasks,
        "successful_requests": results_collector.success_count,
        "failed_requests": results_collector.error_count,
        "timeout_requests": results_collector.timeout_count,
        "request_success_rate": results_collector.success_count * 100.0 / total_tasks if total_tasks > 0 else 0
    }
    
    # 延迟分布统计（per-request 口径）
    if success_latencies:
        stats["min_latency_ms"] = min(success_latencies)
        stats["max_latency_ms"] = max(success_latencies)
        n = len(success_latencies)
        stats["p50_latency_ms"] = success_latencies[n * 50 // 100] if n > 0 else 0
        stats["p90_latency_ms"] = success_latencies[n * 90 // 100] if n > 0 else 0
        stats["p99_latency_ms"] = success_latencies[min(n - 1, n * 99 // 100)] if n > 0 else 0
        if len(success_latencies) > 1:
            stats["latency_stdev_ms"] = stdev(success_latencies)
    
    return stats


# ==================== 主要 Benchmark 函数 ====================

async def run_benchmark_worker_model(
    url: str,
    token: str,
    images: list,
    ground_truth: dict,
    runs_per_image: int = 1,
    concurrency: int = 10,
    params: dict = None
) -> dict:
    """
    运行基准测试 - Worker 模型（标准生产者-消费者模式）
    
    架构:
    - Producer: 构建任务队列 + sentinel
    - Workers (concurrency 个): 从队列取任务执行
    - Collector: 收集结果
    
    Args:
        url: 服务器 URL
        token: 认证 token
        images: 图片列表
        ground_truth: 真实标签
        runs_per_image: 每张图片运行次数
        concurrency: 并发 worker 数量
        params: OCR 参数
    
    Returns:
        统计结果字典
    """
    params = params or {}
    
    print("\n" + "=" * 70)
    print("Starting API Benchmark (Worker Model - 标准生产者-消费者模式)")
    print("=" * 70)
    print(f"Server URL: {url}")
    print(f"Total Images: {len(images)}")
    print(f"Runs per Image: {runs_per_image}")
    print(f"Concurrency (Workers): {concurrency}")
    print(f"Ground Truth: {'Available' if ground_truth else 'Not available'}")
    print("=" * 70 + "\n")
    
    # 1) Producer: 构建任务队列
    task_queue = asyncio.Queue()
    result_queue = asyncio.Queue()
    total_tasks = len(images) * runs_per_image
    
    request_id = 0
    for run_idx in range(runs_per_image):
        for image in images:
            filename = image["filename"] if isinstance(image, dict) else "unknown"
            image_base64 = image["base64"] if isinstance(image, dict) else image
            await task_queue.put(TaskInfo(
                request_id=request_id,
                filename=filename,
                base64_data=image_base64,
                run_idx=run_idx
            ))
            request_id += 1
    
    # 放入 sentinel 通知 workers 退出
    for _ in range(concurrency):
        await task_queue.put(None)
    
    print(f"[Producer] Task queue created: {total_tasks} tasks + {concurrency} sentinels")
    
    # 2) 创建 ClientSession / Connector
    connector = aiohttp.TCPConnector(
        limit=concurrency * 2,
        limit_per_host=concurrency * 2,
        keepalive_timeout=30,
    )
    
    results_collector = ResultsCollector()
    
    benchmark_start = time.time()
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # 3) 启动 workers
        workers = [
            asyncio.create_task(
                worker_loop(i, task_queue, result_queue, session, url, token, params)
            )
            for i in range(concurrency)
        ]
        
        # 4) 启动 collector
        collector_task = asyncio.create_task(
            collector_loop(result_queue, results_collector, expected_results=total_tasks, progress_every=10)
        )
        
        # 5) 等待任务处理完
        await task_queue.join()       # 所有 task（含 sentinel）都被 get + task_done
        await result_queue.join()     # 所有结果都被 collector 消费
        
        # 6) 等待协程结束
        worker_results = await asyncio.gather(*workers)
        received_count = await collector_task
    
    wall_ms = (time.time() - benchmark_start) * 1000
    
    print(f"\n[Benchmark] Completed!")
    print(f"  - Workers processed: {sum(worker_results)}")
    print(f"  - Collector received: {received_count}")
    print(f"  - Duration: {wall_ms:.2f} ms")
    
    # 7) 构建统计结果
    print("\n[Per-Image Results]")
    stats = build_stats_from_results(
        results_collector, images, ground_truth, wall_ms, runs_per_image, concurrency
    )
    
    print(f"\n[Summary Statistics]")
    print(f"  - Real QPS (Throughput): {stats['real_qps']:.2f}")
    print(f"  - Avg Latency (per-image): {stats['avg_latency_ms']:.2f} ms")
    print(f"  - Request Success Rate: {stats['request_success_rate']:.2f}%")
    
    return stats


def run_benchmark_serial(url: str, token: str, images: list, ground_truth: dict,
                         runs_per_image: int = 1, params: dict = None) -> dict:
    """串行运行基准测试（用于测量单请求延迟）"""
    import requests
    
    print("\n" + "=" * 60)
    print("Starting API Benchmark (Serial Mode)")
    print("=" * 60)
    print(f"Server URL: {url}")
    print(f"Total Images: {len(images)}")
    print(f"Runs per Image: {runs_per_image}")
    print(f"Concurrency: 1 (serial mode)")
    print(f"Ground Truth: {'Available' if ground_truth else 'Not available'}")
    print("=" * 60 + "\n")
    
    def send_request_sync(image_base64: str) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"token {token}"
        }
        
        payload = {
            "file": image_base64,
            "fileType": 1,
            "useDocOrientationClassify": params.get("useDocOrientationClassify", False) if params else False,
            "useDocUnwarping": params.get("useDocUnwarping", False) if params else False,
            "textDetThresh": params.get("textDetThresh", 0.3) if params else 0.3,
            "textDetBoxThresh": params.get("textDetBoxThresh", 0.6) if params else 0.6,
            "textDetUnclipRatio": params.get("textDetUnclipRatio", 1.5) if params else 1.5,
            "textRecScoreThresh": params.get("textRecScoreThresh", 0.0) if params else 0.0,
            "visualize": params.get("visualize", False) if params else False,
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            latency = (time.time() - start_time) * 1000
            
            result = {
                "success": response.status_code == 200,
                "http_code": response.status_code,
                "latency_ms": latency,
                "error_msg": "",
                "text": "",
                "char_count": 0
            }
            
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    if json_response.get("errorCode", -1) != 0:
                        result["success"] = False
                        result["error_msg"] = json_response.get("errorMsg", "Unknown error")
                    else:
                        ocr_results = json_response.get("result", {}).get("ocrResults", [])
                        texts = [r.get("prunedResult", "") for r in ocr_results]
                        result["text"] = "".join(texts)
                        result["char_count"] = len(result["text"])
                except json.JSONDecodeError:
                    result["success"] = False
                    result["error_msg"] = "Invalid JSON response"
            else:
                result["error_msg"] = f"HTTP {response.status_code}"
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "http_code": 0,
                "latency_ms": (time.time() - start_time) * 1000,
                "error_msg": str(e),
                "text": "",
                "char_count": 0
            }
    
    image_results = []
    total_chars = 0
    total_time = 0
    successful = 0
    
    for idx, image in enumerate(images):
        filename = image["filename"] if isinstance(image, dict) else f"image_{idx}.png"
        image_base64 = image["base64"] if isinstance(image, dict) else image
        
        latencies = []
        char_count = 0
        text = ""
        
        for run in range(runs_per_image):
            result = send_request_sync(image_base64)
            if result["success"]:
                latencies.append(result["latency_ms"])
                char_count = result["char_count"]
                text = result["text"]
        
        if latencies:
            avg_latency = mean(latencies)
            fps = 1000.0 / avg_latency if avg_latency > 0 else 0
            cps = char_count * 1000.0 / avg_latency if avg_latency > 0 else 0
            
            accuracy = None
            if filename in ground_truth:
                accuracy = calculate_char_accuracy(text, ground_truth[filename])
            
            image_results.append({
                "filename": filename,
                "latency_ms": avg_latency,
                "fps": fps,
                "cps": cps,
                "char_count": char_count,
                "accuracy": accuracy,
                "text": text
            })
            
            total_chars += char_count
            total_time += avg_latency
            successful += 1
            
            acc_str = f"{accuracy:.2f}%" if accuracy is not None else "N/A"
            print(f"[{idx+1}/{len(images)}] {filename}: {avg_latency:.2f}ms, {char_count} chars, CPS={cps:.2f}, Acc={acc_str}")
        else:
            print(f"[{idx+1}/{len(images)}] {filename}: FAILED")
            image_results.append({
                "filename": filename,
                "latency_ms": 0,
                "fps": 0,
                "cps": 0,
                "char_count": 0,
                "accuracy": None,
                "text": ""
            })
    
    # 计算汇总统计
    valid_results = [r for r in image_results if r["latency_ms"] > 0]
    
    if valid_results:
        avg_latency = mean([r["latency_ms"] for r in valid_results])
        avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0
        avg_cps = total_chars * 1000.0 / total_time if total_time > 0 else 0
        
        accuracies = [r["accuracy"] for r in valid_results if r["accuracy"] is not None]
        avg_accuracy = mean(accuracies) if accuracies else None
    else:
        avg_latency = avg_fps = avg_cps = 0
        avg_accuracy = None
    
    real_qps = 1000.0 / avg_latency if avg_latency > 0 else 0
    
    stats = {
        "total_images": len(images),
        "successful_images": successful,
        "failed_images": len(images) - successful,
        "success_rate": successful * 100.0 / len(images) if images else 0,
        "total_time_ms": total_time,
        "wall_clock_time_ms": total_time,
        "total_chars": total_chars,
        "avg_latency_ms": avg_latency,
        "avg_fps": avg_fps,
        "avg_cps": avg_cps,
        "real_qps": real_qps,
        "avg_accuracy": avg_accuracy,
        "image_results": image_results,
        "runs_per_image": runs_per_image,
        "max_in_flight": 1
    }
    
    # 计算延迟分布
    if valid_results:
        latencies = sorted([r["latency_ms"] for r in valid_results])
        stats["min_latency_ms"] = min(latencies)
        stats["max_latency_ms"] = max(latencies)
        stats["p50_latency_ms"] = latencies[len(latencies) * 50 // 100] if latencies else 0
        stats["p90_latency_ms"] = latencies[len(latencies) * 90 // 100] if latencies else 0
        stats["p99_latency_ms"] = latencies[min(len(latencies) - 1, len(latencies) * 99 // 100)] if latencies else 0
    
    return stats


def run_benchmark(url: str, token: str, images: list, ground_truth: dict,
                  runs_per_image: int = 1, max_in_flight: int = 1, params: dict = None) -> dict:
    """
    运行基准测试的入口函数
    
    Args:
        max_in_flight: 最大并发数
            - 1: 串行模式，逐张图片测试（测量单请求延迟）
            - >1: Worker 模式，使用 N 个 worker 并发处理（测量吞吐量）
    """
    if max_in_flight > 1:
        return asyncio.run(run_benchmark_worker_model(
            url, token, images, ground_truth, runs_per_image, max_in_flight, params
        ))
    else:
        return run_benchmark_serial(url, token, images, ground_truth, runs_per_image, params)


# ==================== 报告生成 ====================

def print_results(stats: dict):
    """打印结果"""
    print("\n" + "=" * 70)
    print("Benchmark Results Summary")
    print("=" * 70)
    print(f"Total Images:      {stats['total_images']}")
    print(f"Successful:        {stats['successful_images']}")
    print(f"Failed:            {stats['failed_images']}")
    print(f"Success Rate:      {stats['success_rate']:.2f}%")
    print("-" * 50)
    print(f"Total Time:        {stats['total_time_ms']:.2f} ms")
    print(f"Total Characters:  {stats['total_chars']}")
    print("-" * 50)
    print(f"Avg Latency:       {stats['avg_latency_ms']:.2f} ms")
    print(f"Avg FPS:           {stats['avg_fps']:.2f}")
    print(f"Avg CPS:           {stats['avg_cps']:.2f} chars/s")
    
    if stats.get('max_in_flight', 1) > 1:
        print("-" * 50)
        print(f"Real QPS:          {stats.get('real_qps', 0):.2f} (throughput)")
        print(f"Max In-Flight:     {stats['max_in_flight']}")
        if 'total_requests' in stats:
            print(f"Total Requests:    {stats['total_requests']}")
            print(f"Successful Reqs:   {stats['successful_requests']}")
            print(f"Failed Reqs:       {stats['failed_requests']}")
            print(f"Timeout Reqs:      {stats['timeout_requests']}")
    
    if stats.get('avg_accuracy') is not None:
        print("-" * 50)
        print(f"Avg Accuracy:      {stats['avg_accuracy']:.2f}%")
    
    if 'p50_latency_ms' in stats:
        print("-" * 50)
        print(f"Latency P50:       {stats['p50_latency_ms']:.2f} ms")
        print(f"Latency P90:       {stats['p90_latency_ms']:.2f} ms")
        print(f"Latency P99:       {stats['p99_latency_ms']:.2f} ms")
        print(f"Latency Min:       {stats['min_latency_ms']:.2f} ms")
        print(f"Latency Max:       {stats['max_latency_ms']:.2f} ms")
    
    print("=" * 70)


def generate_markdown_report(stats: dict, output_dir: str):
    """生成 Markdown 报告"""
    report_path = Path(output_dir) / "API_benchmark_report.md"
    
    image_results = stats.get("image_results", [])
    
    lines = []
    lines.append("# DXNN-OCR API Server Benchmark Report\n")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Test Configuration
    lines.append("## Test Configuration\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append("| Model | PP-OCR v5 (DEEPX NPU acceleration) |")
    lines.append(f"| Total Images | {stats['total_images']} |")
    lines.append(f"| Runs per Image | {stats.get('runs_per_image', 1)} |")
    lines.append(f"| Max In-Flight | {stats.get('max_in_flight', 1)} |")
    lines.append(f"| Mode | {'Worker Model (生产者-消费者)' if stats.get('max_in_flight', 1) > 1 else 'Serial'} |")
    lines.append(f"| Success Rate | {stats['success_rate']:.1f}% |\n")
    
    # Performance Summary
    lines.append("## Performance Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **Real QPS (Throughput)** | **{stats.get('real_qps', 0):.2f}** |")
    lines.append(f"| Avg Latency | {stats['avg_latency_ms']:.2f} ms |")
    lines.append(f"| Avg FPS | {stats['avg_fps']:.2f} |")
    lines.append(f"| Avg CPS | {stats['avg_cps']:.2f} chars/s |")
    lines.append(f"| Total Time | {stats['total_time_ms']:.2f} ms |")
    lines.append(f"| Total Characters | {stats['total_chars']} |")
    if stats.get('avg_accuracy') is not None:
        lines.append(f"| Avg Accuracy | {stats['avg_accuracy']:.2f}% |")
    lines.append("")
    
    # Latency Distribution
    if 'p50_latency_ms' in stats:
        lines.append("## Latency Distribution\n")
        lines.append("| Percentile | Latency (ms) |")
        lines.append("|------------|--------------|")
        lines.append(f"| Min | {stats['min_latency_ms']:.2f} |")
        lines.append(f"| P50 | {stats['p50_latency_ms']:.2f} |")
        lines.append(f"| P90 | {stats['p90_latency_ms']:.2f} |")
        lines.append(f"| P99 | {stats['p99_latency_ms']:.2f} |")
        lines.append(f"| Max | {stats['max_latency_ms']:.2f} |")
        if 'latency_stdev_ms' in stats:
            lines.append(f"| Std Dev | {stats['latency_stdev_ms']:.2f} |")
        lines.append("")
    
    # Request Statistics (Worker mode only)
    if stats.get('max_in_flight', 1) > 1 and 'total_requests' in stats:
        lines.append("## Request Statistics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Requests | {stats['total_requests']} |")
        lines.append(f"| Successful | {stats['successful_requests']} |")
        lines.append(f"| Failed | {stats['failed_requests']} |")
        lines.append(f"| Timeout | {stats['timeout_requests']} |")
        lines.append(f"| Success Rate | {stats['request_success_rate']:.2f}% |")
        lines.append("")
    
    # Per-Image Results
    lines.append("## Per-Image Results\n")
    lines.append("| Filename | Latency (ms) | FPS | CPS | Accuracy |")
    lines.append("|----------|--------------|-----|-----|----------|")
    
    for r in image_results:
        acc_str = f"{r['accuracy']:.2f}%" if r['accuracy'] is not None else "N/A"
        lines.append(f"| `{r['filename']}` | {r['latency_ms']:.2f} | {r['fps']:.2f} | {r['cps']:.2f} | {acc_str} |")
    
    # Average row
    avg_acc_str = f"{stats['avg_accuracy']:.2f}%" if stats.get('avg_accuracy') is not None else "N/A"
    lines.append(f"| **Average** | **{stats['avg_latency_ms']:.2f}** | **{stats['avg_fps']:.2f}** | **{stats['avg_cps']:.2f}** | **{avg_acc_str}** |")
    lines.append("")
    
    report_content = "\n".join(lines)
    
    # 打印到终端
    print("\n" + "=" * 70)
    print("MARKDOWN REPORT")
    print("=" * 70)
    print(report_content)
    print("=" * 70)
    
    # 保存到文件
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\n✓ Markdown report saved to: {report_path}")


def save_json_results(stats: dict, output_file: str):
    """保存 JSON 结果"""
    stats_copy = stats.copy()
    if "image_results" in stats_copy:
        stats_copy["image_results"] = [
            {k: v for k, v in r.items() if k != "text"}
            for r in stats_copy["image_results"]
        ]
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(stats_copy, f, indent=4, ensure_ascii=False)
    print(f"✓ JSON results saved to: {output_file}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="OCR API Server Benchmark (Worker Model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 串行模式（测量单请求延迟）
  python run_api_benchmark.py -c 1 -r 3 -i /path/to/images
  
  # Worker 模式（测量吞吐量，推荐）
  python run_api_benchmark.py -c 10 -r 1 -i /path/to/images
  
  # 高并发 Worker 模式
  python run_api_benchmark.py -c 20 -r 1 -i /path/to/images

Note:
  - concurrency (-c) 建议设置为 DXRT_TASK_MAX_LOAD * 2
  - 典型值: 6-12 (对应 DXRT_TASK_MAX_LOAD=3-6)
        """
    )
    
    parser.add_argument("-u", "--url", default="http://localhost:8080/ocr",
                        help="Server URL (default: http://localhost:8080/ocr)")
    parser.add_argument("-t", "--token", default="test_token",
                        help="Authorization token (default: test_token)")
    parser.add_argument("-r", "--runs", type=int, default=1,
                        help="Number of runs per image (default: 1)")
    parser.add_argument("-c", "--concurrency", type=int, default=10,
                        help="Number of concurrent workers (default: 10). Use 1 for serial mode.")
    parser.add_argument("-i", "--images", default="",
                        help="Directory containing test images (with optional labels.json)")
    parser.add_argument("-o", "--output", default="results/api_benchmark_results.json",
                        help="Output JSON file (default: results/api_benchmark_results.json)")
    parser.add_argument("--report-dir", default="results",
                        help="Directory for Markdown report (default: results)")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip Markdown report generation")
    
    args = parser.parse_args()
    
    runs_per_image = args.runs
    
    # 加载图片
    if args.images:
        images = load_images_from_directory(args.images)
        ground_truth = load_ground_truth(args.images)
    else:
        images = [{"filename": "default.png", "base64": DEFAULT_TEST_IMAGE}]
        ground_truth = {}
    
    if not images:
        print("Error: No images loaded")
        sys.exit(1)
    
    # 运行基准测试
    stats = run_benchmark(
        args.url,
        args.token,
        images,
        ground_truth,
        runs_per_image,
        args.concurrency
    )
    
    # 输出结果
    print_results(stats)
    save_json_results(stats, args.output)
    
    if not args.no_report:
        Path(args.report_dir).mkdir(parents=True, exist_ok=True)
        generate_markdown_report(stats, args.report_dir)
    
    print("\n" + "=" * 70)
    print("✓ Benchmark completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
