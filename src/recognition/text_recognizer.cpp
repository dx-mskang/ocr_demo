#include "recognition/text_recognizer.h"
#include "recognition/rec_postprocess.h"
#include "preprocessing/image_ops.h"
#include "common/logger.hpp"
#include <algorithm>

namespace DeepXOCR {

TextRecognizer::TextRecognizer(const RecognizerConfig& config)
    : config_(config) {
}

bool TextRecognizer::Initialize() {
    if (config_.modelPaths.empty()) {
        LOG_ERROR("No recognition models specified");
        return false;
    }
    
    // 加载所有ratio模型
    LOG_INFO("Loading recognition models...");
    for (const auto& [ratio, model_path] : config_.modelPaths) {
        try {
            auto model = std::make_unique<dxrt::InferenceEngine>(model_path);
            models_[ratio] = std::move(model);
            LOG_INFO("  Loaded ratio_%d model: %s", ratio, model_path.c_str());
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load ratio_%d model: %s", ratio, e.what());
            return false;
        }
    }
    
    if (models_.empty()) {
        LOG_ERROR("No recognition models loaded");
        return false;
    }
    
    // 加载字符字典
    LOG_INFO("Loading character dictionary from: %s", config_.dictPath.c_str());
    decoder_ = std::make_unique<ocr::CTCDecoder>(config_.dictPath, true);
    
    if (decoder_->getDictSize() == 0) {
        LOG_ERROR("Failed to load character dictionary");
        return false;
    }
    
    LOG_INFO("TextRecognizer initialized successfully");
    LOG_INFO("  Models: %zu ratios", models_.size());
    LOG_INFO("  Dictionary: %zu characters", decoder_->getDictSize());
    
    return true;
}

std::pair<std::string, float> TextRecognizer::Recognize(const cv::Mat& textImage) {
    if (textImage.empty()) {
        LOG_ERROR("Input image is empty");
        return {"", 0.0f};
    }
    
    // 选择合适的模型
    auto* engine = SelectModel(textImage);
    if (!engine) {
        LOG_ERROR("No suitable model for image size %dx%d", 
                  textImage.cols, textImage.rows);
        return {"", 0.0f};
    }
    
    // 获取ratio
    int ratio = CalculateRatio(textImage.cols, textImage.rows);
    
    // 预处理
    cv::Mat preprocessed = Preprocess(textImage, ratio);
    if (preprocessed.empty()) {
        LOG_ERROR("Preprocessing failed");
        return {"", 0.0f};
    }
    
    // 推理
    auto outputs = engine->Run(preprocessed.data);
    if (outputs.empty()) {
        LOG_ERROR("Inference failed: no output tensors");
        return {"", 0.0f};
    }
    
    // 后处理 (CTC解码)
    auto result = Postprocess(outputs);
    
    // 置信度过滤
    if (result.second < config_.confThreshold) {
        LOG_DEBUG("Low confidence result filtered: %.3f < %.3f", 
                  result.second, config_.confThreshold);
        return {"", result.second};
    }
    
    return result;
}

std::vector<std::pair<std::string, float>> TextRecognizer::RecognizeBatch(
    const std::vector<cv::Mat>& textImages) {
    
    std::vector<std::pair<std::string, float>> results;
    results.reserve(textImages.size());
    
    for (const auto& image : textImages) {
        results.push_back(Recognize(image));
    }
    
    return results;
}

dxrt::InferenceEngine* TextRecognizer::SelectModel(const cv::Mat& image) {
    int ratio = CalculateRatio(image.cols, image.rows);
    
    auto it = models_.find(ratio);
    if (it != models_.end()) {
        return it->second.get();
    }
    
    // 如果找不到精确匹配，使用最接近的ratio
    LOG_WARN("No exact model for ratio %d, using closest match", ratio);
    
    int closest_ratio = -1;
    int min_diff = INT_MAX;
    
    for (const auto& [r, _] : models_) {
        int diff = std::abs(r - ratio);
        if (diff < min_diff) {
            min_diff = diff;
            closest_ratio = r;
        }
    }
    
    if (closest_ratio != -1) {
        LOG_DEBUG("Using ratio_%d model instead", closest_ratio);
        return models_[closest_ratio].get();
    }
    
    return nullptr;
}

int TextRecognizer::CalculateRatio(int width, int height) {
    if (height == 0) {
        return 35;  // 默认最大ratio
    }
    
    float ratio = static_cast<float>(width) / height;
    
    // 根据Python实现的逻辑
    if (ratio <= 3.0f) return 3;
    if (ratio <= 5.0f) return 5;
    if (ratio <= 10.0f) return 10;
    if (ratio <= 15.0f) return 15;
    if (ratio <= 25.0f) return 25;
    return 35;
}

cv::Mat TextRecognizer::Preprocess(const cv::Mat& image, int ratio) {
    // Recognition预处理：
    // 1. 固定高度48
    // 2. 宽度根据ratio计算
    // 3. PPOCR方式：Pad → Resize
    
    int target_height = config_.inputHeight;  // 48
    int target_width = target_height * ratio;
    
    LOG_DEBUG("Preprocessing: %dx%d -> %dx%d (ratio_%d)",
              image.cols, image.rows, target_width, target_height, ratio);
    
    // 使用PPOCR预处理方式（与Detection一致）
    int orig_h = image.rows;
    int orig_w = image.cols;
    
    float target_ratio = static_cast<float>(target_width) / target_height;
    float orig_ratio = static_cast<float>(orig_w) / orig_h;
    
    cv::Mat padded;
    
    // Step 1: Pad到目标ratio
    if (orig_ratio < target_ratio) {
        // 图像比目标窄 -> 右侧补边
        int new_width = static_cast<int>(orig_h * target_ratio);
        int pad_w = new_width - orig_w;
        cv::copyMakeBorder(image, padded, 0, 0, 0, pad_w,
                          cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else if (orig_ratio > target_ratio) {
        // 图像比目标宽 -> 底部补边
        int new_height = static_cast<int>(orig_w / target_ratio);
        int pad_h = new_height - orig_h;
        cv::copyMakeBorder(image, padded, 0, pad_h, 0, 0,
                          cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else {
        // 已经是目标ratio
        padded = image.clone();
    }
    
    // Step 2: Resize到目标尺寸
    cv::Mat final_image;
    cv::resize(padded, final_image, cv::Size(target_width, target_height));
    
    // 确保连续内存
    if (!final_image.isContinuous()) {
        final_image = final_image.clone();
    }
    
    LOG_DEBUG("Preprocessed: padded %dx%d -> resized %dx%d",
              padded.cols, padded.rows, final_image.cols, final_image.rows);
    
    return final_image;
}

std::pair<std::string, float> TextRecognizer::Postprocess(dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) {
        LOG_ERROR("Empty output tensors");
        return {"", 0.0f};
    }
    
    // 使用CTC解码器
    return decoder_->decode(outputs[0]);
}

} // namespace DeepXOCR
