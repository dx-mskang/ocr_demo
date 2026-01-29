/**
 * @file api_benchmark.cpp
 * @brief OCR API Server 高并发压力测试工具 (C++)
 * 
 * 专注于高并发场景下的性能测试，用于测量服务器的 QPS 上限和吞吐能力
 * 
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <filesystem>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <getopt.h>

using json = nlohmann::json;
namespace fs = std::filesystem;

// ==================== 配置结构 ====================

struct BenchmarkConfig {
    std::string server_url = "http://localhost:8080/ocr";
    std::string token = "test_token";
    int concurrency = 8;              // 默认 8 并发（压力测试场景）
    int runs_per_image = 5;           // 每张图片运行次数
    std::string images_dir = "";      // 测试图片目录
    std::string output_dir = "";      // 输出目录
    std::string output_file = "stress_benchmark_results.json";
    bool verbose = false;
};

// ==================== 结果统计 ====================

struct RequestResult {
    bool success;
    int http_code;
    double latency_ms;
    std::string error_msg;
    std::string filename;
    int run_idx = 0;
    std::string text;
    int char_count = 0;
};

struct BenchmarkResults {
    int total_images = 0;
    int successful_images = 0;
    int failed_images = 0;
    double success_rate = 0;
    
    int total_requests = 0;
    int successful_requests = 0;
    double benchmark_duration_ms = 0;  // 整个测试的实际耗时
    double qps = 0;                    // 实际 QPS
    
    double total_time_ms = 0;          // = sum(avg_latency_ms per image)
    int total_chars = 0;
    double avg_latency_ms = 0;
    double avg_fps = 0;
    double avg_cps = 0;
    
    double min_latency_ms = 0;
    double max_latency_ms = 0;
    double p50_latency_ms = 0;
    double p90_latency_ms = 0;
    double p99_latency_ms = 0;
    
    json image_results = json::array();
};

// ==================== CURL 回调 ====================

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// ==================== Base64 编码 ====================

static const char base64_chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(const std::vector<unsigned char>& data) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];
    size_t in_len = data.size();
    const unsigned char* bytes_to_encode = data.data();

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; i < 4; i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (j = 0; j < i + 1; j++)
            ret += base64_chars[char_array_4[j]];

        while (i++ < 3)
            ret += '=';
    }

    return ret;
}

// ==================== 图片加载 ====================

std::string loadFileAsBase64(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return "";
    }
    
    std::vector<unsigned char> buffer(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    
    return base64_encode(buffer);
}

struct ImageInput {
    std::string filename;
    std::string base64;
};

std::vector<ImageInput> loadImagesFromDirectory(const std::string& dir_path) {
    std::vector<ImageInput> images;
    
    if (!fs::exists(dir_path)) {
        std::cerr << "Directory not found: " << dir_path << std::endl;
        return images;
    }
    
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) continue;
        
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            std::string base64 = loadFileAsBase64(entry.path().string());
            if (!base64.empty()) {
                images.push_back({entry.path().filename().string(), base64});
                std::cout << "Loaded image: " << entry.path().filename().string() << std::endl;
            }
        }
    }
    
    return images;
}

// ==================== HTTP 请求 ====================

RequestResult sendOCRRequest(
    const std::string& url,
    const std::string& token,
    const std::string& image_base64,
    bool verbose,
    const std::string& filename = "",
    int run_idx = 0) {
    
    RequestResult result;
    result.success = false;
    result.http_code = 0;
    result.latency_ms = 0;
    result.filename = filename;
    result.run_idx = run_idx;
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        result.error_msg = "Failed to initialize CURL";
        return result;
    }
    
    // 构建请求 JSON
    json request_json;
    request_json["file"] = image_base64;
    request_json["fileType"] = 1;
    request_json["useDocOrientationClassify"] = false;
    request_json["useDocUnwarping"] = false;
    request_json["textDetThresh"] = 0.3;
    request_json["textDetBoxThresh"] = 0.6;
    request_json["textDetUnclipRatio"] = 1.5;
    request_json["textRecScoreThresh"] = 0.0;
    request_json["visualize"] = false;
    
    std::string request_body = request_json.dump();
    std::string response_body;
    
    // 设置 CURL 选项
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string auth_header = "Authorization: token " + token;
    headers = curl_slist_append(headers, auth_header.c_str());
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, request_body.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
    
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    
    // 执行请求
    CURLcode res = curl_easy_perform(curl);
    
    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    result.latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // 获取 HTTP 状态码
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &result.http_code);
    
    if (res != CURLE_OK) {
        result.error_msg = curl_easy_strerror(res);
    } else {
        try {
            json response = json::parse(response_body);
            if (response.contains("errorCode") && response["errorCode"] == 0) {
                result.success = true;
                
                // 提取 OCR 结果文本
                try {
                    auto ocr_results = response["result"]["ocrResults"];
                    std::string text;
                    if (ocr_results.is_array()) {
                        for (const auto& r : ocr_results) {
                            if (r.contains("prunedResult") && r["prunedResult"].is_string()) {
                                text += r["prunedResult"].get<std::string>();
                            }
                        }
                    }
                    result.text = text;
                    result.char_count = static_cast<int>(text.size());
                } catch (...) {
                    // 忽略解析错误
                }
            } else {
                result.error_msg = response.value("errorMsg", "Unknown error");
            }
        } catch (const json::exception& e) {
            result.error_msg = std::string("JSON parse error: ") + e.what();
        }
    }
    
    if (verbose) {
        std::cout << "Request completed: HTTP " << result.http_code 
                  << ", " << result.latency_ms << " ms"
                  << (result.success ? " [OK]" : " [FAIL: " + result.error_msg + "]")
                  << std::endl;
    }
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    return result;
}

// ==================== 压力测试执行 ====================

BenchmarkResults runStressBenchmark(
    const BenchmarkConfig& config,
    const std::vector<ImageInput>& images) {
    
    BenchmarkResults stats;
    stats.total_images = static_cast<int>(images.size());
    
    if (images.empty()) {
        return stats;
    }
    
    const int total_tasks = static_cast<int>(images.size()) * std::max(1, config.runs_per_image);
    stats.total_requests = total_tasks;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting Stress Test (C++)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Server URL: " << config.server_url << std::endl;
    std::cout << "Total Images: " << images.size() << std::endl;
    std::cout << "Runs per Image: " << config.runs_per_image << std::endl;
    std::cout << "Concurrency: " << config.concurrency << std::endl;
    std::cout << "Total Requests: " << total_tasks << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::vector<RequestResult> all_results;
    all_results.reserve(total_tasks);
    std::mutex results_mutex;
    std::atomic<int> task_index{0};
    std::atomic<int> completed{0};
    
    auto worker = [&]() {
        while (true) {
            int t = task_index.fetch_add(1);
            if (t >= total_tasks) break;
            
            int img_idx = t / std::max(1, config.runs_per_image);
            int run_idx = t % std::max(1, config.runs_per_image);
            
            const auto& img = images[img_idx];
            RequestResult r = sendOCRRequest(
                config.server_url, config.token, img.base64, config.verbose, img.filename, run_idx);
            
            {
                std::lock_guard<std::mutex> lock(results_mutex);
                all_results.push_back(std::move(r));
            }
            
            int done = completed.fetch_add(1) + 1;
            if (done % 10 == 0 || done == total_tasks) {
                std::cout << "\rProgress: " << done << "/" << total_tasks
                          << " (" << (done * 100 / total_tasks) << "%)" << std::flush;
            }
        }
    };
    
    // 记录整个测试的开始时间
    auto benchmark_start = std::chrono::high_resolution_clock::now();
    
    // 并发执行所有任务
    std::vector<std::thread> threads;
    int workers = std::max(1, config.concurrency);
    for (int i = 0; i < workers; ++i) threads.emplace_back(worker);
    for (auto& th : threads) th.join();
    
    // 记录整个测试的结束时间
    auto benchmark_end = std::chrono::high_resolution_clock::now();
    stats.benchmark_duration_ms = std::chrono::duration<double, std::milli>(benchmark_end - benchmark_start).count();
    
    std::cout << std::endl;
    
    // 统计成功的请求数
    stats.successful_requests = std::count_if(all_results.begin(), all_results.end(),
        [](const RequestResult& r) { return r.success; });
    
    // 计算实际 QPS
    stats.qps = stats.successful_requests * 1000.0 / stats.benchmark_duration_ms;
    
    // 按 filename 聚合
    std::map<std::string, std::vector<RequestResult>> by_image;
    for (const auto& r : all_results) {
        by_image[r.filename].push_back(r);
    }
    
    std::vector<double> per_image_latencies;
    per_image_latencies.reserve(images.size());
    
    int total_chars = 0;
    double total_time_ms = 0;
    int successful_images = 0;
    
    for (const auto& img : images) {
        const auto& fn = img.filename;
        auto it = by_image.find(fn);
        if (it == by_image.end()) {
            json row;
            row["filename"] = fn;
            row["latency_ms"] = 0.0;
            row["fps"] = 0.0;
            row["cps"] = 0.0;
            row["char_count"] = 0;
            stats.image_results.push_back(row);
            continue;
        }
        
        const auto& results = it->second;
        std::vector<double> ok_latencies;
        ok_latencies.reserve(results.size());
        int char_count = 0;
        
        for (const auto& r : results) {
            if (r.success) {
                ok_latencies.push_back(r.latency_ms);
                if (char_count == 0 && r.char_count > 0) {
                    char_count = r.char_count;
                }
            }
        }
        
        if (!ok_latencies.empty()) {
            double avg_latency = std::accumulate(ok_latencies.begin(), ok_latencies.end(), 0.0) / ok_latencies.size();
            double fps = avg_latency > 0 ? 1000.0 / avg_latency : 0;
            double cps = avg_latency > 0 ? char_count * 1000.0 / avg_latency : 0;
            
            json row;
            row["filename"] = fn;
            row["latency_ms"] = avg_latency;
            row["fps"] = fps;
            row["cps"] = cps;
            row["char_count"] = char_count;
            stats.image_results.push_back(row);
            
            successful_images++;
            total_chars += char_count;
            total_time_ms += avg_latency;
            per_image_latencies.push_back(avg_latency);
            
            if (config.verbose) {
                std::cout << "  " << fn << ": " << std::fixed << std::setprecision(2)
                          << avg_latency << "ms, " << char_count << " chars, CPS=" << cps << std::endl;
            }
        } else {
            json row;
            row["filename"] = fn;
            row["latency_ms"] = 0.0;
            row["fps"] = 0.0;
            row["cps"] = 0.0;
            row["char_count"] = 0;
            stats.image_results.push_back(row);
        }
    }
    
    stats.successful_images = successful_images;
    stats.failed_images = stats.total_images - successful_images;
    stats.success_rate = stats.total_images > 0 ? successful_images * 100.0 / stats.total_images : 0;
    
    stats.total_chars = total_chars;
    stats.total_time_ms = total_time_ms;
    
    if (!per_image_latencies.empty()) {
        std::sort(per_image_latencies.begin(), per_image_latencies.end());
        stats.min_latency_ms = per_image_latencies.front();
        stats.max_latency_ms = per_image_latencies.back();
        stats.p50_latency_ms = per_image_latencies[per_image_latencies.size() * 50 / 100];
        stats.p90_latency_ms = per_image_latencies[per_image_latencies.size() * 90 / 100];
        stats.p99_latency_ms = per_image_latencies[std::min(per_image_latencies.size() - 1, per_image_latencies.size() * 99 / 100)];
        
        stats.avg_latency_ms = std::accumulate(per_image_latencies.begin(), per_image_latencies.end(), 0.0) / per_image_latencies.size();
        stats.avg_fps = stats.avg_latency_ms > 0 ? 1000.0 / stats.avg_latency_ms : 0;
        stats.avg_cps = total_time_ms > 0 ? total_chars * 1000.0 / total_time_ms : 0;
    }
    
    return stats;
}

// ==================== 结果输出 ====================

void printResults(const BenchmarkResults& results) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Stress Test Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total Images:      " << results.total_images << std::endl;
    std::cout << "Successful:        " << results.successful_images << std::endl;
    std::cout << "Failed:            " << results.failed_images << std::endl;
    std::cout << "Image Success:     " << results.success_rate << "%" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total Requests:    " << results.total_requests << std::endl;
    std::cout << "Successful Req:    " << results.successful_requests << std::endl;
    std::cout << "Benchmark Time:    " << results.benchmark_duration_ms << " ms" << std::endl;
    std::cout << "Actual QPS:        " << results.qps << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total Chars:       " << results.total_chars << std::endl;
    std::cout << "Avg FPS:           " << results.avg_fps << std::endl;
    std::cout << "Avg CPS:           " << results.avg_cps << " chars/s" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Latency (ms):" << std::endl;
    std::cout << "  Min:             " << results.min_latency_ms << std::endl;
    std::cout << "  Max:             " << results.max_latency_ms << std::endl;
    std::cout << "  Avg:             " << results.avg_latency_ms << std::endl;
    std::cout << "  P50:             " << results.p50_latency_ms << std::endl;
    std::cout << "  P90:             " << results.p90_latency_ms << std::endl;
    std::cout << "  P99:             " << results.p99_latency_ms << std::endl;
    std::cout << "========================================" << std::endl;
}

void saveResults(const BenchmarkResults& results, const std::string& output_file) {
    json output;
    output["total_images"] = results.total_images;
    output["successful_images"] = results.successful_images;
    output["failed_images"] = results.failed_images;
    output["success_rate"] = results.success_rate;
    output["total_requests"] = results.total_requests;
    output["successful_requests"] = results.successful_requests;
    output["benchmark_duration_ms"] = results.benchmark_duration_ms;
    output["qps"] = results.qps;
    output["total_time_ms"] = results.total_time_ms;
    output["total_chars"] = results.total_chars;
    output["avg_latency_ms"] = results.avg_latency_ms;
    output["avg_fps"] = results.avg_fps;
    output["avg_cps"] = results.avg_cps;
    output["min_latency_ms"] = results.min_latency_ms;
    output["max_latency_ms"] = results.max_latency_ms;
    output["p50_latency_ms"] = results.p50_latency_ms;
    output["p90_latency_ms"] = results.p90_latency_ms;
    output["p99_latency_ms"] = results.p99_latency_ms;
    output["image_results"] = results.image_results;
    
    // 确保输出目录存在
    fs::path out_path(output_file);
    if (out_path.has_parent_path()) {
        fs::create_directories(out_path.parent_path());
    }
    
    std::ofstream file(output_file);
    file << output.dump(4);
    file.close();
    
    std::cout << "\nResults saved to: " << output_file << std::endl;
}

void saveMarkdownReport(const BenchmarkResults& stats, const std::string& output_dir) {
    fs::create_directories(output_dir);
    fs::path report_path = fs::path(output_dir) / "stress_benchmark_report.md";
    
    auto fmt2 = [](double v) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << v;
        return oss.str();
    };
    
    std::vector<std::string> lines;
    lines.push_back("# DXNN-OCR Stress Test Report (C++)\n");
    
    lines.push_back("**Test Configuration**:");
    lines.push_back("- Concurrency: High-performance C++ implementation");
    lines.push_back("- Total Images: " + std::to_string(stats.total_images));
    lines.push_back("- Total Requests: " + std::to_string(stats.total_requests));
    lines.push_back("- Success Rate: " + fmt2(stats.success_rate) + "%\n");
    
    lines.push_back("**Throughput Metrics**:");
    lines.push_back("| Metric | Value |");
    lines.push_back("|---|---|");
    lines.push_back("| Benchmark Duration | **" + fmt2(stats.benchmark_duration_ms) + " ms** |");
    lines.push_back("| Successful Requests | " + std::to_string(stats.successful_requests) + " |");
    lines.push_back("| **Actual QPS** | **" + fmt2(stats.qps) + "** |");
    lines.push_back("| Average Latency | " + fmt2(stats.avg_latency_ms) + " ms |");
    lines.push_back("| P50 Latency | " + fmt2(stats.p50_latency_ms) + " ms |");
    lines.push_back("| P90 Latency | " + fmt2(stats.p90_latency_ms) + " ms |");
    lines.push_back("| P99 Latency | " + fmt2(stats.p99_latency_ms) + " ms |");
    lines.push_back("");
    
    lines.push_back("**Per-Image Results**:");
    lines.push_back("| Filename | Latency (ms) | FPS | CPS |");
    lines.push_back("|---|---|---|---|");
    
    for (const auto& row : stats.image_results) {
        std::string fn = row.value("filename", "unknown");
        double lat = row.value("latency_ms", 0.0);
        double fps = row.value("fps", 0.0);
        double cps = row.value("cps", 0.0);
        
        std::ostringstream oss;
        oss << "| `" << fn << "` | " << std::fixed << std::setprecision(2) << lat
            << " | " << fps << " | **" << cps << "** |";
        lines.push_back(oss.str());
    }
    
    lines.push_back("");
    lines.push_back("**Performance Summary**:");
    lines.push_back("- **QPS (Queries Per Second): " + fmt2(stats.qps) + "**");
    lines.push_back("- Average CPS: " + fmt2(stats.avg_cps) + " chars/s");
    lines.push_back("- Total Characters: " + std::to_string(stats.total_chars));
    
    std::ofstream out(report_path);
    for (const auto& l : lines) out << l << "\n";
    out.close();
    
    std::cout << "Markdown report saved to: " << report_path.string() << std::endl;
}

// ==================== 主函数 ====================

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "\nOCR API Server Stress Test (C++)" << std::endl;
    std::cout << "High-concurrency performance testing tool\n" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -u, --url <url>          Server URL (default: http://localhost:8080/ocr)" << std::endl;
    std::cout << "  -t, --token <token>      Authorization token (default: test_token)" << std::endl;
    std::cout << "  -c, --concurrency <num>  Number of concurrent workers (default: 8)" << std::endl;
    std::cout << "  -r, --runs <num>         Runs per image (default: 5)" << std::endl;
    std::cout << "  -i, --images <dir>       Directory containing test images" << std::endl;
    std::cout << "  -o, --output <file>      Output JSON filename (default: stress_benchmark_results.json)" << std::endl;
    std::cout << "  --output-dir <dir>       Output directory (default: server/benchmark/results/)" << std::endl;
    std::cout << "  -v, --verbose            Verbose output" << std::endl;
    std::cout << "  -h, --help               Show this help message" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  # High concurrency stress test:" << std::endl;
    std::cout << "  " << program_name << " -i ./images -c 16 -r 5" << std::endl;
    std::cout << "\n  # Extreme stress test:" << std::endl;
    std::cout << "  " << program_name << " -i ./images -c 32 -r 10" << std::endl;
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config;
    
    enum { OPT_OUTPUT_DIR = 1000 };
    
    static struct option long_options[] = {
        {"url",           required_argument, 0, 'u'},
        {"token",         required_argument, 0, 't'},
        {"concurrency",   required_argument, 0, 'c'},
        {"runs",          required_argument, 0, 'r'},
        {"images",        required_argument, 0, 'i'},
        {"output",        required_argument, 0, 'o'},
        {"output-dir",    required_argument, 0, OPT_OUTPUT_DIR},
        {"verbose",       no_argument,       0, 'v'},
        {"help",          no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "u:t:c:r:i:o:vh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'u':
                config.server_url = optarg;
                break;
            case 't':
                config.token = optarg;
                break;
            case 'c':
                config.concurrency = std::stoi(optarg);
                break;
            case 'r':
                config.runs_per_image = std::stoi(optarg);
                break;
            case 'i':
                config.images_dir = optarg;
                break;
            case 'o':
                config.output_file = optarg;
                break;
            case OPT_OUTPUT_DIR:
                config.output_dir = optarg;
                break;
            case 'v':
                config.verbose = true;
                break;
            case 'h':
                printUsage(argv[0]);
                return 0;
            default:
                printUsage(argv[0]);
                return 1;
        }
    }
    
    // 检查必要参数
    if (config.images_dir.empty()) {
        std::cerr << "Error: Images directory (-i) is required" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    // 初始化 CURL
    curl_global_init(CURL_GLOBAL_ALL);
    
    // 加载测试图片
    std::vector<ImageInput> images = loadImagesFromDirectory(config.images_dir);
    
    if (images.empty()) {
        std::cerr << "Error: No images found in " << config.images_dir << std::endl;
        curl_global_cleanup();
        return 1;
    }
    
    // 运行压力测试
    BenchmarkResults results = runStressBenchmark(config, images);
    
    // 输出结果
    printResults(results);
    
    // 确定输出目录
    std::string output_dir = config.output_dir;
    if (output_dir.empty()) {
        if (fs::exists("../server/benchmark/results")) {
            output_dir = "../server/benchmark/results/";
        } else if (fs::exists("server/benchmark/results")) {
            output_dir = "server/benchmark/results/";
        } else {
            output_dir = "./results/";
            fs::create_directories(output_dir);
        }
    } else {
        fs::create_directories(output_dir);
    }
    
    if (!output_dir.empty() && output_dir.back() != '/') {
        output_dir += '/';
    }
    
    std::string output_file = output_dir + config.output_file;
    saveResults(results, output_file);
    saveMarkdownReport(results, output_dir);
    
    // 清理 CURL
    curl_global_cleanup();
    
    std::cout << "\n✓ Stress test completed!" << std::endl;
    
    return 0;
}
