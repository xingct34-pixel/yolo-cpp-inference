#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
using namespace std;
using namespace cv;

int main() {
    // 读取类别名称
    vector<string> class_names;
    ifstream f("/home/xct/cpp_projects/coco.txt");
    string line;
    while (getline(f, line)) {
        class_names.push_back(line);
    }
    f.close();
    cout << "加载类别数量：" << class_names.size() << endl;

    // 加载模型
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo");
    Ort::SessionOptions session_options;
    const char* model_path = "/home/xct/cpp_projects/yolo11n.onnx";
    Ort::Session session(env, model_path, session_options);
    cout << "模型加载成功" << endl;

    // 打开视频
    VideoCapture cap("/home/xct/cpp_projects/test.mp4");
    if (!cap.isOpened()) {
        cout << "视频打开失败" << endl;
        return -1;
    }
    cout << "视频打开成功" << endl;

    Mat img;
    int frame_count = 0;

    while (true) {
        // 读取一帧
        cap >> img;
        if (img.empty()) break;

        int img_w = img.cols, img_h = img.rows;

        // 预处理
        Mat blob;
        resize(img, blob, Size(640, 640));
        blob.convertTo(blob, CV_32F, 1.0 / 255.0);
        cvtColor(blob, blob, COLOR_BGR2RGB);

        Mat channels[3];
        split(blob, channels);
        vector<float> input_data;
        for (int c = 0; c < 3; c++) {
            input_data.insert(input_data.end(),
                (float*)channels[c].data,
                (float*)channels[c].data + 640 * 640);
        }

        array<int64_t, 4> input_shape{1, 3, 640, 640};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        // 推理计时
        auto start = chrono::high_resolution_clock::now();

        const char* input_names[] = {"images"};
        const char* output_names[] = {"output0"};
        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        auto end = chrono::high_resolution_clock::now();
        float inference_time = chrono::duration<float, milli>(end - start).count();
        float fps = 1000.0 / inference_time;

        // 解析结果
        float* data = outputs[0].GetTensorMutableData<float>();
        float conf_threshold = 0.5;
        vector<Rect> boxes;
        vector<float> scores;
        vector<int> class_ids;

        for (int i = 0; i < 8400; i++) {
            float max_score = 0;
            int class_id = 0;
            for (int c = 0; c < 80; c++) {
                float score = data[c * 8400 + 4 * 8400 + i];
                if (score > max_score) {
                    max_score = score;
                    class_id = c;
                }
            }
            if (max_score > conf_threshold) {
                float cx = data[0 * 8400 + i] * img_w / 640;
                float cy = data[1 * 8400 + i] * img_h / 640;
                float w  = data[2 * 8400 + i] * img_w / 640;
                float h  = data[3 * 8400 + i] * img_h / 640;
                int x = (int)(cx - w / 2);
                int y = (int)(cy - h / 2);
                boxes.push_back(Rect(x, y, (int)w, (int)h));
                scores.push_back(max_score);
                class_ids.push_back(class_id);
            }
        }

        // NMS
        vector<int> indices;
        dnn::NMSBoxes(boxes, scores, conf_threshold, 0.45, indices);

        // 画框
        for (int idx : indices) {
            rectangle(img, boxes[idx], Scalar(0, 255, 0), 2);
            string label;
            if (class_ids[idx] < (int)class_names.size()) {
                label = class_names[class_ids[idx]];
            } else {
                label = "class" + to_string(class_ids[idx]);
            }
            label += " " + to_string((int)(scores[idx] * 100)) + "%";
            putText(img, label, Point(boxes[idx].x, boxes[idx].y - 5),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        }

        // 显示fps
        putText(img, "FPS: " + to_string((int)fps),
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 1,
                Scalar(0, 0, 255), 2);

        // 每10帧保存一次结果图片
        frame_count++;
        if (frame_count % 10 == 0) {
            imwrite("/home/xct/cpp_projects/yolo_inference/build/frame_" + 
                    to_string(frame_count) + ".jpg", img);
            cout << "第" << frame_count << "帧，FPS：" << (int)fps 
                 << "，检测到：" << indices.size() << "个目标" << endl;
        }

        if (frame_count >= 50) break;
    }

    cap.release();
    cout << "处理完成" << endl;
    return 0;
}
