#!/usr/bin/env python3
"""
PDF OCR API Server Benchmark Runner (Async Version)
运行 PDF OCR 基准测试并生成 Markdown 报告

使用 aiohttp 实现异步 HTTP 请求，支持"先发后收"模式，
充分利用服务器端 pipeline 并行处理能力

与 run_api_benchmark.py 对等设计，用于 PDF OCR 性能测试

功能:
- 从 server/pdf_file/ 目录加载 PDF 文件
- 发送 PDF OCR 请求并测量性能
- 计算每个 PDF 的推理时间、FPS、CPS
- 生成 Markdown 报告到 results/ 目录
"""

import sys
import json
import argparse
import time
import base64
import subprocess
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime
from statistics import mean


def get_pdf_page_count(pdf_path: str) -> int:
    """获取 PDF 页数（使用 pdfinfo 或估算）"""
    try:
        result = subprocess.run(
            ["pdfinfo", pdf_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("Pages:"):
                    return int(line.split(":")[1].strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # 如果 pdfinfo 不可用，根据文件大小估算
    import os
    file_size_kb = os.path.getsize(pdf_path) / 1024
    estimated_pages = max(1, int(file_size_kb / 20))
    return estimated_pages


def load_pdfs_from_directory(pdf_dir: str) -> list:
    """从目录加载 PDF 文件"""
    pdfs = []
    path = Path(pdf_dir)
    
    if not path.exists():
        print(f"Warning: Directory not found: {pdf_dir}")
        return pdfs
    
    for pdf_file in sorted(path.glob("*.pdf")):
        try:
            with open(pdf_file, "rb") as f:
                pdf_data = f.read()
                base64_data = base64.b64encode(pdf_data).decode("utf-8")
                
                file_size_mb = len(pdf_data) / (1024 * 1024)
                page_count = get_pdf_page_count(str(pdf_file))
                
                pdfs.append({
                    "filename": pdf_file.name,
                    "base64": base64_data,
                    "file_size_mb": file_size_mb,
                    "estimated_pages": page_count
                })
                print(f"Loaded: {pdf_file.name} ({file_size_mb:.2f} MB, ~{page_count} pages)")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
    
    return pdfs


async def send_pdf_ocr_request_async(session: aiohttp.ClientSession, url: str, token: str, 
                                      pdf_base64: str, dpi: int = 150, max_pages: int = 100,
                                      timeout: int = 600) -> dict:
    """异步发送 PDF OCR 请求，返回详细结果"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"token {token}"
    }
    
    payload = {
        "file": pdf_base64,
        "fileType": 0,  # PDF 类型
        "pdfDpi": dpi,
        "pdfMaxPages": max_pages,
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "textDetThresh": 0.3,
        "textDetBoxThresh": 0.6,
        "textDetUnclipRatio": 1.5,
        "textRecScoreThresh": 0.0,
        "visualize": False
    }
    
    start_time = time.time()
    
    try:
        async with session.post(url, headers=headers, json=payload, 
                                timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            latency = (time.time() - start_time) * 1000  # ms
            
            result = {
                "success": response.status == 200,
                "http_code": response.status,
                "latency_ms": latency,
                "error_msg": "",
                "total_pages": 0,
                "rendered_pages": 0,
                "total_chars": 0,
                "pages_data": []
            }
            
            if response.status == 200:
                try:
                    json_response = await response.json()
                    if json_response.get("errorCode", -1) != 0:
                        result["success"] = False
                        result["error_msg"] = json_response.get("errorMsg", "Unknown error")
                    else:
                        # 提取 PDF OCR 结果
                        pdf_result = json_response.get("result", {})
                        result["total_pages"] = pdf_result.get("totalPages", 0)
                        result["rendered_pages"] = pdf_result.get("renderedPages", 0)
                        result["warning"] = pdf_result.get("warning", "")
                        
                        # 统计所有页面的字符数
                        pages = pdf_result.get("pages", [])
                        result["pages_data"] = pages
                        total_chars = 0
                        for page in pages:
                            ocr_results = page.get("ocrResults", [])
                            for ocr in ocr_results:
                                text = ocr.get("prunedResult", "")
                                total_chars += len(text)
                        result["total_chars"] = total_chars
                        
                except json.JSONDecodeError:
                    result["success"] = False
                    result["error_msg"] = "Invalid JSON response"
            else:
                result["error_msg"] = f"HTTP {response.status}"
            
            return result
            
    except asyncio.TimeoutError:
        return {
            "success": False,
            "http_code": 0,
            "latency_ms": timeout * 1000,
            "error_msg": "Timeout",
            "total_pages": 0,
            "rendered_pages": 0,
            "total_chars": 0,
            "pages_data": []
        }
    except aiohttp.ClientError as e:
        return {
            "success": False,
            "http_code": 0,
            "latency_ms": 0,
            "error_msg": str(e),
            "total_pages": 0,
            "rendered_pages": 0,
            "total_chars": 0,
            "pages_data": []
        }


def send_pdf_ocr_request(url: str, token: str, pdf_base64: str, 
                         dpi: int = 150, max_pages: int = 100,
                         timeout: int = 600) -> dict:
    """同步发送 PDF OCR 请求（用于串行模式），返回详细结果"""
    import requests
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"token {token}"
    }
    
    payload = {
        "file": pdf_base64,
        "fileType": 0,  # PDF 类型
        "pdfDpi": dpi,
        "pdfMaxPages": max_pages,
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "textDetThresh": 0.3,
        "textDetBoxThresh": 0.6,
        "textDetUnclipRatio": 1.5,
        "textRecScoreThresh": 0.0,
        "visualize": False
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        latency = (time.time() - start_time) * 1000  # ms
        
        result = {
            "success": response.status_code == 200,
            "http_code": response.status_code,
            "latency_ms": latency,
            "error_msg": "",
            "total_pages": 0,
            "rendered_pages": 0,
            "total_chars": 0,
            "pages_data": []
        }
        
        if response.status_code == 200:
            try:
                json_response = response.json()
                if json_response.get("errorCode", -1) != 0:
                    result["success"] = False
                    result["error_msg"] = json_response.get("errorMsg", "Unknown error")
                else:
                    # 提取 PDF OCR 结果
                    pdf_result = json_response.get("result", {})
                    result["total_pages"] = pdf_result.get("totalPages", 0)
                    result["rendered_pages"] = pdf_result.get("renderedPages", 0)
                    result["warning"] = pdf_result.get("warning", "")
                    
                    # 统计所有页面的字符数
                    pages = pdf_result.get("pages", [])
                    result["pages_data"] = pages
                    total_chars = 0
                    for page in pages:
                        ocr_results = page.get("ocrResults", [])
                        for ocr in ocr_results:
                            text = ocr.get("prunedResult", "")
                            total_chars += len(text)
                    result["total_chars"] = total_chars
                    
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
            "latency_ms": 0,
            "error_msg": str(e),
            "total_pages": 0,
            "rendered_pages": 0,
            "total_chars": 0,
            "pages_data": []
        }


async def run_pdf_benchmark_async(url: str, token: str, pdfs: list, 
                                   dpi: int = 150, max_pages: int = 100,
                                   runs_per_pdf: int = 1, concurrency: int = 1,
                                   timeout: int = 600) -> dict:
    """异步运行 PDF OCR 基准测试，实现"先发后收"模式"""
    print("\n" + "=" * 60)
    print("Starting PDF OCR Benchmark (Python Async)")
    print("=" * 60)
    print(f"Server URL: {url}")
    print(f"Total PDFs: {len(pdfs)}")
    print(f"Runs per PDF: {runs_per_pdf}")
    print(f"Concurrency: {concurrency}")
    print(f"PDF DPI: {dpi}")
    print(f"Max Pages: {max_pages}")
    print(f"Timeout: {timeout}s")
    print(f"Mode: Async HTTP (先发后收)")
    print("=" * 60 + "\n")
    
    pdf_results = []
    total_chars = 0
    total_time = 0
    total_pages_processed = 0
    successful = 0
    
    print(f"[异步模式] 并发发送请求 (max_concurrent={concurrency})...\n")
    
    # 构建任务列表
    tasks_info = []
    for pdf in pdfs:
        for run_idx in range(runs_per_pdf):
            tasks_info.append({
                "filename": pdf["filename"],
                "base64": pdf["base64"],
                "file_size_mb": pdf["file_size_mb"],
                "run_idx": run_idx
            })
    
    total_tasks = len(tasks_info)
    all_results = []
    completed = [0]
    
    # 使用 Semaphore 控制并发数
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request(session, task_info):
        """带并发限制的请求"""
        async with semaphore:
            result = await send_pdf_ocr_request_async(session, url, token, task_info["base64"], 
                                                       dpi, max_pages, timeout)
            result["filename"] = task_info["filename"]
            result["run_idx"] = task_info["run_idx"]
            result["file_size_mb"] = task_info["file_size_mb"]
            
            completed[0] += 1
            print(f"\rProgress: {completed[0]}/{total_tasks} ({completed[0] * 100 // total_tasks}%)", end="", flush=True)
            
            return result
    
    # 异步执行所有请求
    benchmark_start = time.time()
    
    # 创建连接池，提高连接复用
    connector = aiohttp.TCPConnector(
        limit=concurrency * 2,  # 连接池大小
        limit_per_host=concurrency * 2,
        keepalive_timeout=30
    )
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # 先发后收：一次性创建所有任务，异步等待完成
        tasks = [bounded_request(session, info) for info in tasks_info]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    benchmark_duration = (time.time() - benchmark_start) * 1000
    print()
    
    # 处理异常结果
    processed_results = []
    for r in all_results:
        if isinstance(r, Exception):
            processed_results.append({
                "success": False,
                "filename": "unknown",
                "run_idx": -1,
                "latency_ms": 0,
                "error_msg": str(r),
                "total_pages": 0,
                "rendered_pages": 0,
                "total_chars": 0,
                "file_size_mb": 0
            })
        else:
            processed_results.append(r)
    
    all_results = processed_results
    
    # 按 PDF 聚合结果
    results_by_pdf = {}
    for r in all_results:
        fn = r["filename"]
        if fn not in results_by_pdf:
            results_by_pdf[fn] = []
        results_by_pdf[fn].append(r)
    
    for pdf in pdfs:
        filename = pdf["filename"]
        pdf_results_list = results_by_pdf.get(filename, [])
        
        latencies = [r["latency_ms"] for r in pdf_results_list if r["success"]]
        if latencies:
            avg_latency = mean(latencies)
            # 取第一个成功结果
            first_success = next((r for r in pdf_results_list if r["success"]), pdf_results_list[0] if pdf_results_list else None)
            char_count = first_success["total_chars"] if first_success else 0
            rendered_pages = first_success["rendered_pages"] if first_success else 0
            total_pages = first_success["total_pages"] if first_success else 0
            
            fps = 1000.0 / avg_latency if avg_latency > 0 else 0
            cps = char_count * 1000.0 / avg_latency if avg_latency > 0 else 0
            pps = rendered_pages * 1000.0 / avg_latency if avg_latency > 0 else 0  # Pages Per Second
            
            pdf_results.append({
                "filename": filename,
                "latency_ms": avg_latency,
                "fps": fps,
                "cps": cps,
                "pps": pps,
                "char_count": char_count,
                "total_pages": total_pages,
                "rendered_pages": rendered_pages,
                "file_size_mb": pdf["file_size_mb"]
            })
            
            total_chars += char_count
            total_time += avg_latency
            total_pages_processed += rendered_pages
            successful += 1
            
            print(f"  {filename}: {avg_latency:.2f}ms, {rendered_pages}/{total_pages} pages, {char_count} chars, CPS={cps:.2f}")
        else:
            print(f"  {filename}: FAILED")
            pdf_results.append({
                "filename": filename, "latency_ms": 0, "fps": 0, "cps": 0, "pps": 0,
                "char_count": 0, "total_pages": 0, "rendered_pages": 0,
                "file_size_mb": pdf["file_size_mb"]
            })
    
    # 计算并发 QPS (真实吞吐量)
    total_successful = sum(1 for r in all_results if r["success"])
    concurrent_qps = total_successful * 1000.0 / benchmark_duration if benchmark_duration > 0 else 0
    print(f"\n[异步统计] 总耗时: {benchmark_duration:.2f}ms, 成功: {total_successful}, 真实QPS: {concurrent_qps:.2f}")
    
    # 保存并发统计 - 使用实际墙钟时间
    total_time = benchmark_duration
    
    # 计算汇总统计
    valid_results = [r for r in pdf_results if r["latency_ms"] > 0]
    
    if valid_results:
        avg_latency = mean([r["latency_ms"] for r in valid_results])
        avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0
        avg_cps = total_chars * 1000.0 / total_time if total_time > 0 else 0
        avg_pps = total_pages_processed * 1000.0 / total_time if total_time > 0 else 0
    else:
        avg_latency = avg_fps = avg_cps = avg_pps = 0
    
    # 计算真实 QPS (使用墙钟时间)
    real_qps = successful * 1000.0 / total_time if total_time > 0 else 0
    
    stats = {
        "total_pdfs": len(pdfs),
        "successful_pdfs": successful,
        "failed_pdfs": len(pdfs) - successful,
        "success_rate": successful * 100.0 / len(pdfs) if pdfs else 0,
        "total_time_ms": total_time,
        "wall_clock_time_ms": total_time,
        "total_chars": total_chars,
        "total_pages_processed": total_pages_processed,
        "avg_latency_ms": avg_latency,
        "avg_fps": avg_fps,
        "avg_cps": avg_cps,
        "avg_pps": avg_pps,
        "real_qps": real_qps,
        "pdf_results": pdf_results,
        "runs_per_pdf": runs_per_pdf,
        "concurrency": concurrency,
        "pdf_dpi": dpi,
        "pdf_max_pages": max_pages
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


def run_pdf_benchmark_serial(url: str, token: str, pdfs: list, 
                              dpi: int = 150, max_pages: int = 100,
                              runs_per_pdf: int = 1, timeout: int = 600) -> dict:
    """串行运行 PDF OCR 基准测试（用于测量单请求延迟）"""
    print("\n" + "=" * 60)
    print("Starting PDF OCR Benchmark (Python Serial)")
    print("=" * 60)
    print(f"Server URL: {url}")
    print(f"Total PDFs: {len(pdfs)}")
    print(f"Runs per PDF: {runs_per_pdf}")
    print(f"Concurrency: 1 (serial mode)")
    print(f"PDF DPI: {dpi}")
    print(f"Max Pages: {max_pages}")
    print(f"Timeout: {timeout}s")
    print("=" * 60 + "\n")
    
    pdf_results = []
    total_chars = 0
    total_time = 0
    total_pages_processed = 0
    successful = 0
    
    for idx, pdf in enumerate(pdfs):
        filename = pdf["filename"]
        pdf_base64 = pdf["base64"]
        
        # 多次运行取平均
        latencies = []
        char_count = 0
        rendered_pages = 0
        total_pages = 0
        
        for run in range(runs_per_pdf):
            result = send_pdf_ocr_request(url, token, pdf_base64, dpi, max_pages, timeout)
            if result["success"]:
                latencies.append(result["latency_ms"])
                char_count = result["total_chars"]
                rendered_pages = result["rendered_pages"]
                total_pages = result["total_pages"]
        
        if latencies:
            avg_latency = mean(latencies)
            fps = 1000.0 / avg_latency if avg_latency > 0 else 0
            cps = char_count * 1000.0 / avg_latency if avg_latency > 0 else 0
            pps = rendered_pages * 1000.0 / avg_latency if avg_latency > 0 else 0
            
            pdf_results.append({
                "filename": filename,
                "latency_ms": avg_latency,
                "fps": fps,
                "cps": cps,
                "pps": pps,
                "char_count": char_count,
                "total_pages": total_pages,
                "rendered_pages": rendered_pages,
                "file_size_mb": pdf["file_size_mb"]
            })
            
            total_chars += char_count
            total_time += avg_latency
            total_pages_processed += rendered_pages
            successful += 1
            
            print(f"[{idx+1}/{len(pdfs)}] {filename}: {avg_latency:.2f}ms, {rendered_pages}/{total_pages} pages, {char_count} chars, CPS={cps:.2f}")
        else:
            print(f"[{idx+1}/{len(pdfs)}] {filename}: FAILED")
            pdf_results.append({
                "filename": filename, "latency_ms": 0, "fps": 0, "cps": 0, "pps": 0,
                "char_count": 0, "total_pages": 0, "rendered_pages": 0,
                "file_size_mb": pdf["file_size_mb"]
            })
    
    # 计算汇总统计
    valid_results = [r for r in pdf_results if r["latency_ms"] > 0]
    
    if valid_results:
        avg_latency = mean([r["latency_ms"] for r in valid_results])
        avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0
        avg_cps = total_chars * 1000.0 / total_time if total_time > 0 else 0
        avg_pps = total_pages_processed * 1000.0 / total_time if total_time > 0 else 0
    else:
        avg_latency = avg_fps = avg_cps = avg_pps = 0
    
    real_qps = 1000.0 / avg_latency if avg_latency > 0 else 0
    
    stats = {
        "total_pdfs": len(pdfs),
        "successful_pdfs": successful,
        "failed_pdfs": len(pdfs) - successful,
        "success_rate": successful * 100.0 / len(pdfs) if pdfs else 0,
        "total_time_ms": total_time,
        "wall_clock_time_ms": total_time,
        "total_chars": total_chars,
        "total_pages_processed": total_pages_processed,
        "avg_latency_ms": avg_latency,
        "avg_fps": avg_fps,
        "avg_cps": avg_cps,
        "avg_pps": avg_pps,
        "real_qps": real_qps,
        "pdf_results": pdf_results,
        "runs_per_pdf": runs_per_pdf,
        "concurrency": 1,
        "pdf_dpi": dpi,
        "pdf_max_pages": max_pages
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


def run_pdf_benchmark(url: str, token: str, pdfs: list, 
                      dpi: int = 150, max_pages: int = 100,
                      runs_per_pdf: int = 1, concurrency: int = 1,
                      timeout: int = 600) -> dict:
    """运行 PDF OCR 基准测试的入口函数
    
    Args:
        concurrency: 并发数
            - 1: 串行模式，逐个 PDF 测试（测量单请求延迟）
            - >1: 异步模式，使用 aiohttp 实现先发后收（测量吞吐量）
    """
    if concurrency > 1:
        return asyncio.run(run_pdf_benchmark_async(url, token, pdfs, dpi, max_pages, 
                                                    runs_per_pdf, concurrency, timeout))
    else:
        return run_pdf_benchmark_serial(url, token, pdfs, dpi, max_pages, runs_per_pdf, timeout)


def print_results(stats: dict):
    """打印结果"""
    print("\n" + "=" * 60)
    print("PDF Benchmark Results Summary")
    print("=" * 60)
    print(f"Total PDFs:        {stats['total_pdfs']}")
    print(f"Successful:        {stats['successful_pdfs']}")
    print(f"Failed:            {stats['failed_pdfs']}")
    print(f"Success Rate:      {stats['success_rate']:.2f}%")
    print("-" * 40)
    print(f"Total Time:        {stats['total_time_ms']:.2f} ms")
    print(f"Total Characters:  {stats['total_chars']}")
    print(f"Total Pages:       {stats['total_pages_processed']}")
    print("-" * 40)
    print(f"Avg Latency:       {stats['avg_latency_ms']:.2f} ms")
    print(f"Avg FPS:           {stats['avg_fps']:.2f}")
    print(f"Avg CPS:           {stats['avg_cps']:.2f} chars/s")
    print(f"Avg PPS:           {stats['avg_pps']:.2f} pages/s")
    print("=" * 60)


def generate_markdown_report(stats: dict, output_dir: str):
    """生成 Markdown 报告"""
    report_path = Path(output_dir) / "PDF_benchmark_report.md"
    
    pdf_results = stats.get("pdf_results", [])
    
    # 构建报告内容
    lines = []
    lines.append("# DXNN-OCR PDF API Server Benchmark Report\n")
    
    # Test Configuration
    lines.append("**Test Configuration**:")
    lines.append("- Model: PP-OCR v5 (DEEPX NPU acceleration)")
    lines.append(f"- Total PDFs Tested: {stats['total_pdfs']}")
    lines.append(f"- Runs per PDF: {stats.get('runs_per_pdf', 1)}")
    lines.append(f"- PDF DPI: {stats.get('pdf_dpi', 150)}")
    lines.append(f"- Max Pages per PDF: {stats.get('pdf_max_pages', 100)}")
    lines.append(f"- Success Rate: {stats['success_rate']:.1f}%\n")
    
    # Test Results Table
    lines.append("**Test Results**:")
    lines.append("| Filename | Size (MB) | Pages | Inference Time (ms) | FPS | CPS (chars/s) | PPS (pages/s) |")
    lines.append("|---|---|---|---|---|---|---|")
    
    for r in pdf_results:
        lines.append(f"| `{r['filename']}` | {r['file_size_mb']:.2f} | {r['rendered_pages']}/{r['total_pages']} | {r['latency_ms']:.2f} | {r['fps']:.2f} | **{r['cps']:.2f}** | {r['pps']:.2f} |")
    
    # Average row
    lines.append(f"| **Average** | - | {stats['total_pages_processed']} | **{stats['avg_latency_ms']:.2f}** | **{stats['avg_fps']:.2f}** | **{stats['avg_cps']:.2f}** | **{stats['avg_pps']:.2f}** |\n")
    
    # Performance Summary
    lines.append("**Performance Summary**:")
    lines.append(f"- Average Inference Time: **{stats['avg_latency_ms']:.2f} ms** (per-request latency)")
    lines.append(f"- Average FPS: **{stats['avg_fps']:.2f}** (1000/latency)")
    lines.append(f"- Average CPS: **{stats['avg_cps']:.2f} chars/s**")
    lines.append(f"- Average PPS: **{stats['avg_pps']:.2f} pages/s**")
    
    # 并发模式下显示真实 QPS
    if stats.get('concurrency', 1) > 1:
        lines.append(f"- **Real QPS (Throughput): {stats.get('real_qps', 0):.2f}** (async mode)")
        lines.append(f"- Concurrency: **{stats['concurrency']}** workers")
        lines.append(f"- Wall Clock Time: **{stats['wall_clock_time_ms']:.2f} ms**")
    
    lines.append(f"- Total Characters Detected: **{stats['total_chars']}**")
    lines.append(f"- Total Pages Processed: **{stats['total_pages_processed']}**")
    lines.append(f"- Total Processing Time: **{stats['total_time_ms']:.2f} ms**")
    lines.append(f"- Success Rate: **{stats['success_rate']:.1f}%** ({stats['successful_pdfs']}/{stats['total_pdfs']} PDFs)")
    
    report_content = "\n".join(lines)
    
    # 打印到终端
    print("\n" + "=" * 60)
    print("MARKDOWN REPORT")
    print("=" * 60)
    print(report_content)
    print("=" * 60)
    
    # 保存到文件
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\n✓ Markdown report saved to: {report_path}")


def save_json_results(stats: dict, output_file: str):
    """保存 JSON 结果"""
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    print(f"✓ JSON results saved to: {output_file}")


def main():
    # 获取脚本所在目录
    script_dir = Path(__file__).parent.absolute()
    server_dir = script_dir.parent
    default_pdf_dir = server_dir / "pdf_file"
    
    parser = argparse.ArgumentParser(description="PDF OCR API Server Benchmark (Async)")
    parser.add_argument("-u", "--url", default="http://localhost:8080/ocr",
                        help="Server URL (default: http://localhost:8080/ocr)")
    parser.add_argument("-t", "--token", default="test_token",
                        help="Authorization token (default: test_token)")
    parser.add_argument("-r", "--runs", type=int, default=1,
                        help="Number of runs per PDF (default: 1)")
    parser.add_argument("-c", "--concurrency", type=int, default=1,
                        help="Number of concurrent workers (default: 1)")
    parser.add_argument("-p", "--pdfs", default=str(default_pdf_dir),
                        help=f"Directory containing test PDFs (default: {default_pdf_dir})")
    parser.add_argument("--dpi", type=int, default=150,
                        help="PDF rendering DPI (default: 150)")
    parser.add_argument("--max-pages", type=int, default=100,
                        help="Max pages per PDF (default: 100)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Request timeout in seconds (default: 600)")
    parser.add_argument("-o", "--output", default="results/pdf_benchmark_results.json",
                        help="Output JSON file (default: results/pdf_benchmark_results.json)")
    parser.add_argument("--report-dir", default="results",
                        help="Directory for Markdown report (default: results)")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip Markdown report generation")
    
    args = parser.parse_args()
    
    # 加载 PDF 文件
    pdfs = load_pdfs_from_directory(args.pdfs)
    
    if not pdfs:
        print(f"Error: No PDF files found in {args.pdfs}")
        sys.exit(1)
    
    # 运行基准测试
    stats = run_pdf_benchmark(
        args.url, args.token, pdfs,
        dpi=args.dpi,
        max_pages=args.max_pages,
        runs_per_pdf=args.runs,
        concurrency=args.concurrency,
        timeout=args.timeout
    )
    
    # 输出结果
    print_results(stats)
    save_json_results(stats, args.output)
    
    if not args.no_report:
        Path(args.report_dir).mkdir(parents=True, exist_ok=True)
        generate_markdown_report(stats, args.report_dir)
    
    print("\n" + "=" * 60)
    print("✓ PDF Benchmark completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
