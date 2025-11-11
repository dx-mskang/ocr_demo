#include "common/visualizer.h"
#include "common/logger.hpp"
#include <random>

namespace ocr {

cv::Mat Visualizer::drawTextBoxes(const cv::Mat& image,
                                  const std::vector<TextBox>& boxes,
                                  const cv::Scalar& color,
                                  int thickness) {
    cv::Mat vis = image.clone();

    for (const auto& box : boxes) {
        // 绘制多边形
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Point> int_points;
        for (int i = 0; i < 4; i++) {
            int_points.push_back(cv::Point(static_cast<int>(box.points[i].x), 
                                          static_cast<int>(box.points[i].y)));
        }
        contours.push_back(int_points);

        cv::polylines(vis, contours, true, color, thickness);

        // 绘制置信度（如果有）
        if (box.confidence > 0) {
            std::string conf_text = cv::format("%.2f", box.confidence);
            cv::Point text_pos(static_cast<int>(box.points[0].x), 
                             static_cast<int>(box.points[0].y) - 5);
            cv::putText(vis, conf_text, text_pos, cv::FONT_HERSHEY_SIMPLEX,
                       0.5, cv::Scalar(255, 0, 0), 1);
        }
    }

    return vis;
}

cv::Mat Visualizer::drawOCRResults(const cv::Mat& image,
                                   const std::vector<TextBox>& boxes,
                                   bool draw_text,
                                   bool draw_confidence) {
    cv::Mat vis = image.clone();

    for (size_t i = 0; i < boxes.size(); i++) {
        const auto& box = boxes[i];

        // 生成随机颜色
        cv::Scalar color = getRandomColor();

        // 绘制多边形
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Point> int_points;
        for (int j = 0; j < 4; j++) {
            int_points.push_back(cv::Point(static_cast<int>(box.points[j].x), 
                                          static_cast<int>(box.points[j].y)));
        }
        contours.push_back(int_points);

        cv::polylines(vis, contours, true, color, 2);

        // 绘制文本和置信度
        if (draw_text && !box.text.empty()) {
            std::string label = box.text;
            if (draw_confidence && box.confidence > 0) {
                label += cv::format(" (%.2f)", box.confidence);
            }

            cv::Point text_pos(static_cast<int>(box.points[0].x), 
                             static_cast<int>(box.points[0].y) - 5);
            
            // 绘制文本背景
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                                 0.6, 2, &baseline);
            cv::rectangle(vis, 
                         text_pos + cv::Point(0, baseline),
                         text_pos + cv::Point(text_size.width, -text_size.height),
                         cv::Scalar(0, 0, 0), cv::FILLED);

            // 绘制文本
            cv::putText(vis, label, text_pos, cv::FONT_HERSHEY_SIMPLEX,
                       0.6, cv::Scalar(255, 255, 255), 2);
        }
    }

    return vis;
}

cv::Mat Visualizer::drawPolygon(const cv::Mat& image,
                               const std::vector<cv::Point>& points,
                               const cv::Scalar& color,
                               int thickness,
                               bool fill) {
    cv::Mat vis = image.clone();

    if (points.empty()) {
        return vis;
    }

    std::vector<std::vector<cv::Point>> contours = {points};

    if (fill) {
        cv::fillPoly(vis, contours, color);
    } else {
        cv::polylines(vis, contours, true, color, thickness);
    }

    return vis;
}

cv::Mat Visualizer::drawText(const cv::Mat& image,
                            const std::string& text,
                            const cv::Point& position,
                            double font_scale,
                            const cv::Scalar& color,
                            int thickness,
                            bool background) {
    cv::Mat vis = image.clone();

    if (background) {
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                             font_scale, thickness, &baseline);
        cv::rectangle(vis, 
                     position + cv::Point(0, baseline),
                     position + cv::Point(text_size.width, -text_size.height),
                     cv::Scalar(0, 0, 0), cv::FILLED);
    }

    cv::putText(vis, text, position, cv::FONT_HERSHEY_SIMPLEX,
               font_scale, color, thickness);

    return vis;
}

cv::Mat Visualizer::createMosaic(const std::vector<cv::Mat>& images,
                                const std::vector<std::string>& labels,
                                int cols) {
    if (images.empty()) {
        return cv::Mat();
    }

    int rows = (images.size() + cols - 1) / cols;

    // 找到最大尺寸
    int max_height = 0;
    int max_width = 0;
    for (const auto& img : images) {
        max_height = std::max(max_height, img.rows);
        max_width = std::max(max_width, img.cols);
    }

    // 创建画布
    int canvas_height = rows * (max_height + 40); // 40 for label
    int canvas_width = cols * max_width;
    cv::Mat canvas = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC3);
    canvas.setTo(cv::Scalar(255, 255, 255));

    // 放置图像
    for (size_t i = 0; i < images.size(); i++) {
        int row = i / cols;
        int col = i % cols;

        int x = col * max_width;
        int y = row * (max_height + 40);

        // 确保图像有3个通道
        cv::Mat img_to_place = images[i];
        if (img_to_place.channels() == 1) {
            cv::cvtColor(img_to_place, img_to_place, cv::COLOR_GRAY2BGR);
        }

        // 放置图像
        cv::Rect roi(x, y + 30, img_to_place.cols, img_to_place.rows);
        if (roi.x + roi.width <= canvas.cols && roi.y + roi.height <= canvas.rows) {
            img_to_place.copyTo(canvas(roi));
        }

        // 添加标签
        if (i < labels.size() && !labels[i].empty()) {
            cv::putText(canvas, labels[i], cv::Point(x + 5, y + 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        }
    }

    return canvas;
}

bool Visualizer::save(const cv::Mat& image,
                     const std::string& output_path,
                     int quality) {
    if (image.empty()) {
        LOG_ERROR("Cannot save empty image to %s", output_path.c_str());
        return false;
    }

    std::vector<int> params;
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(quality);

    bool success = cv::imwrite(output_path, image, params);
    if (!success) {
        LOG_ERROR("Failed to save image to %s", output_path.c_str());
    }

    return success;
}

void Visualizer::show(const cv::Mat& image,
                     const std::string& window_name,
                     int wait_key) {
    if (image.empty()) {
        LOG_ERROR("Cannot show empty image");
        return;
    }

    cv::Mat display_img = resizeForDisplay(image);
    cv::imshow(window_name, display_img);
    cv::waitKey(wait_key);
}

cv::Mat Visualizer::resizeForDisplay(const cv::Mat& image,
                                     int max_width,
                                     int max_height) {
    if (image.cols <= max_width && image.rows <= max_height) {
        return image;
    }

    float ratio = std::min(
        static_cast<float>(max_width) / image.cols,
        static_cast<float>(max_height) / image.rows
    );

    int new_width = static_cast<int>(image.cols * ratio);
    int new_height = static_cast<int>(image.rows * ratio);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height));

    return resized;
}

cv::Scalar Visualizer::getRandomColor() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 255);

    return cv::Scalar(dis(gen), dis(gen), dis(gen));
}

cv::Size Visualizer::getTextSize(const std::string& text, 
                                double font_scale, 
                                int thickness) {
    int baseline = 0;
    return cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 
                          font_scale, thickness, &baseline);
}

} // namespace ocr
