#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <vector>

// 结构体定义
struct Task {
    cv::Mat snapshot;
    cv::Mat xyz_map; //3D 点云图存储，保证时空对齐
    std::vector<cv::Rect> boxes;
};

struct RenderResult {
    cv::Rect display_box;
    std::string label;
    cv::Scalar color;
    bool is_in_zone;
    float similarity;
};

// 线程安全队列
template <typename T>
class SafeQueue {
    std::queue<T> q;
    std::mutex m;
public:
    void push(T val) {
        std::lock_guard<std::mutex> lock(m);
        q = std::queue<T>();
        q.push(val);
    }

    bool pop(T& val) {
        std::lock_guard<std::mutex> lock(m);
        if (q.empty()) return false;
        val = q.front();
        q.pop();
        return true;
    }
};

SafeQueue<Task> g_yolo_queue, g_reid_queue;
std::vector<std::vector<float>> g_feature_db;
std::vector<RenderResult> g_render_results;
std::mutex g_db_mutex, g_render_mutex;
bool g_keep_running = true;

// 特征提取
std::vector<float> extract_feature(const cv::Mat& person_img, cv::dnn::Net& net) {
    cv::Mat rgb, resized, f32;
    cv::cvtColor(person_img, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(128, 256));
    resized.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    cv::Scalar mean(0.485, 0.456, 0.406), std(0.229, 0.224, 0.225);
    f32 = (f32 - mean) / std;
    cv::Mat blob = cv::dnn::blobFromImage(f32, 1.0, cv::Size(128, 256), cv::Scalar(), false, false);
    net.setInput(blob);
    cv::Mat feat = net.forward();
    cv::normalize(feat, feat, 1, 0, cv::NORM_L2);
    std::vector<float> f;
    feat.row(0).copyTo(f);
    return f;
}

float calculate_similarity(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dot = 0, da = 0, db = 0;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot += v1[i] * v2[i];
        da += v1[i] * v1[i];
        db += v2[i] * v2[i];
    }
    return (da == 0 || db == 0) ? 0 : dot / (sqrt(da) * sqrt(db));
}

// YOLO 线程
void yolo_detection_thread(cv::dnn::Net net) {
    while (g_keep_running) {
        Task task;
        if (!g_yolo_queue.pop(task)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        cv::Mat blob = cv::dnn::blobFromImage(task.snapshot, 1/255.0, cv::Size(320, 320), cv::Scalar(), true, true);
        net.setInput(blob);
        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());
        std::vector<cv::Rect> boxes;
        std::vector<float> confs;
        for (auto& out : outs) {
            float* data = (float*)out.data;
            for (int i = 0; i < out.size[1]; ++i, data += out.size[2]) {
                if (data[4] > 0.5 && data[5] > 0.5) {
                    boxes.push_back(cv::Rect(data[0] - data[2]/2, data[1] - data[3]/2, data[2], data[3]));
                    confs.push_back(data[4] * data[5]);
                }
            }
        }
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confs, 0.5, 0.4, indices);
        for (int idx : indices) task.boxes.push_back(boxes[idx]);
        g_reid_queue.push(task);
    }
}

// Re-ID 线程
void reid_inference_thread(cv::dnn::Net net, cv::Rect zone) {
    while (g_keep_running) {
        Task task;
        if (!g_reid_queue.pop(task)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        std::vector<RenderResult> current_render;
        for (auto& box : task.boxes) {
            double scale_x = (double)task.snapshot.cols / 320.0; // 640 / 320 = 2.0
            double scale_y = (double)task.snapshot.rows / 320.0; // 480 / 320 = 1.5

            cv::Rect crop(
                static_cast<int>(box.x * scale_x),
                static_cast<int>(box.y * scale_y),
                static_cast<int>(box.width * scale_x),
                static_cast<int>(box.height * scale_y)
            );
         
            crop &= cv::Rect(0, 0, task.snapshot.cols, task.snapshot.rows);
            if (crop.area() <= 0) continue;

            // xyz 图尺寸是 320x240,box图是320x320。所以为了切 xyz 图，我们需要把 crop 框等比缩小回小图尺度
            cv::Rect xyz_crop(
                static_cast<int>(box.x),
                static_cast<int>(box.y * 0.75),
                static_cast<int>(box.width),
                static_cast<int>(box.height * 0.75)
            );
            xyz_crop &= cv::Rect(0, 0, task.xyz_map.cols, task.xyz_map.rows);

            RenderResult res;
            res.display_box = crop;
            res.is_in_zone = zone.contains((res.display_box.br() + res.display_box.tl()) * 0.5);

            if (xyz_crop.area() > 0) {
                cv::Mat person_xyz = task.xyz_map(xyz_crop);
                std::vector<cv::Mat> person_channels;
                cv::split(person_xyz, person_channels);
                cv::Mat z_roi = person_channels[2]; // 提取 Z 轴（深度）通道

                cv::Scalar mean_z, stddev_z;
                // 过滤掉无效点（0和5000毫米以外的噪声）
                cv::meanStdDev(z_roi, mean_z, stddev_z, z_roi > 0 & z_roi < 5000);

                // 如果 Z 轴标准差小于等于 30.0，说明物体表面完全没有立体起伏 -> 平面照片/噪声攻击
                if (stddev_z[0] <= 30.0) {
                    res.similarity = 0.0f;
                    res.label = "SUSPECT / PHOTO";
                    res.color = cv::Scalar(0, 0, 255); // 高危红色警告框
                    std::cout<<"this is a photo!"<<std::endl;
                    current_render.push_back(res);
                    continue; // 【防线拦截】直接跳过特征提取，拒绝身份识别
                }
            }

            // 活体校验通过，执行原有的 Re-ID 推理逻辑
            cv::Mat p = task.snapshot(crop).clone();
            std::vector<float> feat = extract_feature(p, net);
            float max_sim = 0;
            {
                std::lock_guard<std::mutex> lock(g_db_mutex);
                for (auto& db : g_feature_db) {
                    float s = calculate_similarity(feat, db);
                    if (s > max_sim) max_sim = s;
                }
            }

            res.similarity = max_sim;
            res.label = (max_sim > 0.8) ? ("ID:" + std::to_string((int)(max_sim * 100)) + "%") : (res.is_in_zone ? "Press 'r'" : "Unknown");
            res.color = (max_sim > 0.8) ? cv::Scalar(0, 0, 255) : (res.is_in_zone ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0));
            current_render.push_back(res);
        }
        std::lock_guard<std::mutex> lock(g_render_mutex);
        g_render_results = current_render;
    }
}

int main() {
    cv::FileStorage fs("../stereo_calib.yaml", cv::FileStorage::READ);
    if (!fs.isOpened()) return -1;

    cv::Mat K1, D1, K2, D2, R, T;
    cv::Size img_size;

    fs["K1"] >> K1; fs["D1"] >> D1;
    fs["K2"] >> K2; fs["D2"] >> D2;
    fs["R"] >> R;   fs["T"] >> T;
    fs["image_size"] >> img_size;
    fs.release();
    
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K1, D1, K1, D1, img_size, R, T, R1, R2, P1, P2, Q);
    cv::Mat map11, map12, map21, map22;
    cv::initUndistortRectifyMap(K1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

    int numberOfDisparities = 64;
    int SADWindowSize = 5;

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, numberOfDisparities, SADWindowSize,
        8 * 1 * SADWindowSize * SADWindowSize,
        32 * 1 * SADWindowSize * SADWindowSize,
        1, 63, 15, 100, 1, cv::StereoSGBM::MODE_SGBM
    );

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cv::dnn::Net yolo = cv::dnn::readNetFromONNX("../models/yolov5n.onnx");
    cv::dnn::Net reid = cv::dnn::readNetFromONNX("../models/osnet_x0_25_market1501.onnx");

    cv::Rect reg_zone(240, 120, 160, 240);
    std::thread t1(yolo_detection_thread, std::ref(yolo));
    std::thread t2(reid_inference_thread, std::ref(reid), reg_zone);

    cv::Mat frame, left_raw, left_rect, right_raw, right_rect;
    cv::Mat left_gray, right_gray;
    cv::Mat left_smooth, right_smooth;
    cv::Mat disp;
    cv::Mat disp_median;
    cv::Mat left_small, right_small;
    cv::Mat xyz;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        left_raw = frame(cv::Rect(0, 0, 640, 480));
        right_raw = frame(cv::Rect(1280 / 2, 0, 1280 / 2, 480));

        cv::remap(left_raw, left_rect, map11, map12, cv::INTER_LINEAR);
        cv::remap(right_raw, right_rect, map21, map22, cv::INTER_LINEAR);

        cv::resize(left_rect, left_small, cv::Size(320, 240));
        cv::resize(right_rect, right_small, cv::Size(320, 240));

        // 2. 双目深度计算 (SGBM) 
        cv::cvtColor(left_small, left_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(right_small, right_gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(left_gray, left_smooth, cv::Size(3, 3), 0);
        cv::GaussianBlur(right_gray, right_smooth, cv::Size(3, 3), 0);

        sgbm->compute(left_smooth, right_smooth, disp);
        cv::medianBlur(disp, disp_median, 3);
        cv::reprojectImageTo3D(disp_median, xyz, Q, true);

        // 将点云 xyz 图跟随当前帧克隆打包一起送入 YOLO 队列
        g_yolo_queue.push({left_rect.clone(), xyz.clone(), {}});

        int key = cv::waitKey(1);
        std::vector<RenderResult> to_render;
        {
            std::lock_guard<std::mutex> lock(g_render_mutex);
            to_render = g_render_results;
        }

        for (auto& r : to_render) {
            cv::rectangle(left_rect, r.display_box, r.color, 2);
            std::string text = r.label + " [" + std::to_string((int)(r.similarity * 100)) + "%]";
            cv::putText(left_rect, text, r.display_box.tl() - cv::Point(0, 5), 0, 0.5, r.color, 1);

            if (key == 'r' && r.is_in_zone) {
                cv::Rect inner = r.display_box;
                inner.x += inner.width * 0.1;
                inner.y += inner.height * 0.1;
                inner.width *= 0.8;
                inner.height *= 0.8;
                cv::Mat person_roi = left_rect(inner & cv::Rect(0, 0, left_rect.cols, left_rect.rows)).clone();
                std::vector<float> feat = extract_feature(person_roi, reid);
                std::lock_guard<std::mutex> lock(g_db_mutex);
                g_feature_db.push_back(feat);
                std::cout << ">>> 录入成功! 当前库内人数: " << g_feature_db.size() << " <<<" << std::endl;
            }
        }
        cv::imshow("Final AR View", left_rect);
        if (key == 27) {
            g_keep_running = false;
            break;
        }
    }
    t1.join();
    t2.join();
    return 0;
}