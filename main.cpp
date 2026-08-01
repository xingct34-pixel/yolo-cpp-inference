#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
using namespace std;
using namespace cv;

int main() {
    // 读取类别名称
    vector<string> class_names;                                   //相比于string names[80];，vector可以动态
    ifstream f("/home/xct/cpp_projects/coco.txt");
    string line;                                           //读一个文件或者用户输入,内容是一行一行的文字。这行代码就是提前准备一个"空盒子"(变量 line),专门用来临时装下每一次读到的一行内容。
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

    Mat img;                           //Mat 是 OpenCV 库里的一个类,全称是 Matrix(矩阵)，因为图像在计算机里本质上就是一个像素矩阵,所以叫 Mat。声明一个空的 Mat 对象,叫 img,现在还没有任何图像数据,只是先准备好这个"盘子",等下用来装每一帧画面。
    int frame_count = 0;                //声明一个整数变量,用来给帧计数,从 0 开始(应该是后面循环里会用它记录处理了第几帧)。

    while (true) {
        // 读取一帧
        cap >> img;                   //cap 应该是一个 cv::VideoCapture 对象(视频或摄像头的读取器)。>> 是重载过的运算符,作用是:从视频源里取出下一帧,存到 img 里。这行等价于 cap.read(img);。
        if (img.empty()) break;

        int img_w = img.cols, img_h = img.rows;

        // 预处理
        Mat blob;            
        resize(img, blob, Size(640, 640));               //声明一个新的 Mat 叫 blob,把原始图像 img(尺寸可能是任意大小)缩放成 640×640。因为模型通常要求固定尺寸的输入,不能随便什么分辨率都喂进去。
        blob.convertTo(blob, CV_32F, 1.0 / 255.0);        //把图像数据类型转换成 CV_32F(32位浮点数),同时每个像素值乘以 1.0/255.0，原始图像每个像素是 0~255 的整数(uchar)，这里把它归一化到 0~1 的浮点数范围,这是神经网络输入的常见要求
        cvtColor(blob, blob, COLOR_BGR2RGB); //把颜色通道顺序从 BGR 转成 RGB。，OpenCV 读图默认是 BGR 顺序，大多数深度学习框架(PyTorch/TensorFlow 训练出来的模型)习惯用 RGB 顺序,顺序对不上模型会认错颜色,精度大幅下降

        Mat channels[3];
        split(blob, channels);
        vector<float> input_data;
        for (int c = 0; c < 3; c++) {
            input_data.insert(input_data.end(),
                (float*)channels[c].data,
                (float*)channels[c].data + 640 * 640);
        }

        array<int64_t, 4> input_shape{1, 3, 640, 640};  //NCHW 格式:N(batch)、C(channel)、H(height)、W(width)1 → Batch(批次数),表示一次只送 1 张图进去推理，3 → Channel(通道数),对应 R、G、B 三个通道，640 → Height(高度)，640 → Width(宽度)
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
