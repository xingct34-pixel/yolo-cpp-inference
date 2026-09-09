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
    ifstream f("/home/xct/cpp_projects/coco.txt");             //input‑file‑stream，文件读入流，专门用来从磁盘文件读取数据（只读，不写）
    string line;                                           //读一个文件或者用户输入,内容是一行一行的文字。这行代码就是提前准备一个"空盒子"(变量 line),专门用来临时装下每一次读到的一行内容。
    while (getline(f, line)) {
        class_names.push_back(line);
    }
    f.close();                                                   //// 关闭文件流，释放文件资源；ifstream对象析构时也会自动关闭，手动关闭更规范
    cout << "加载类别数量：" << class_names.size() << endl;

    // 加载模型
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo");                      //ONNX Runtime运行环境，先有环境才能跑模型 
    Ort::SessionOptions session_options;                                  //配置选项，选择cpu或者gpu，模型cpu
    const char* model_path = "/home/xct/cpp_projects/yolo11n.onnx";          //模型路径
    Ort::Session session(env, model_path, session_options);              //创建推理‘会话’session
    cout << "模型加载成功" << endl;

    // 打开视频
    VideoCapture cap("/home/xct/cpp_projects/test.mp4");
    if (!cap.isOpened()) {
        cout << "视频打开失败" << endl;
        return -1;
    }
    cout << "视频打开成功" << endl;

    Mat img;                           //Mat 是 OpenCV 库里的一个类,Matrix(矩阵)，图像在计算机里是像素矩阵。声明一个空的 Mat 对象,叫 img
    int frame_count = 0;                //声明一个整数变量,用来给帧计数,从 0 开始。

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

        Mat channels[3];                         //三个通道分别对应RGB
        split(blob, channels);                    
        vector<float> input_data;                           //浮点型容器，动态存储输入数据
        for (int c = 0; c < 3; c++) {
            input_data.insert(input_data.end(),
                (float*)channels[c].data,                      //起始指针
                (float*)channels[c].data + 640 * 640);          //结尾指针，单通道总像素 = 409600 个 float 元素，指针向后偏移 409600 个 float 元素位置，得到这块 float 数组末尾的下一个位置
        }
//为什么这么做？OpenCV存图片的方式（HWC）：像素1的R、G、B → 像素2的R、G、B → 像素3的R、G、B...           ONNX模型要求的方式（CHW）：所有像素的R → 所有像素的G → 所有像素的B




        array<int64_t, 4> input_shape{1, 3, 640, 640};  //定义4维张量的形状，NCHW格式，类型必须是int64_t，ONNX Runtime要求NCHW （yolo默认格式）
                                                        //格式:N( Number of samples：样本数量，翻译为批次batch、C(channel)、H(height)、W(width)
                                                        //1 → Batch(批次数)，3 → Channel(通道数),对应 R、G、B 三个通道，640 → Height(高度)，640 → Width(宽度)
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(            //创建CPU内存信息对象，描述张量数据存放在CPU内存
            OrtArenaAllocator, OrtMemTypeDefault);              // 使用ORT自带的Arena内存分配器，内存池提升分配释放效率；     // 默认CPU内存类型，普通主机内存；
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(        //创建ORT的输入张量对象input_tensor，把C++内存包装成ONNX认识的Tensor
            memory_info, input_data.data(), input_data.size(),     //分别是， 内存描述：告知ORT数据在CPU里；    源数据指针：vector<float> input_data底层连续内存首地址，CHW一维数组；  张量总元素数量：1*3*640*640 = 1228800；
        
            input_shape.data(), input_shape.size());               //维度数组首地址，传入{1,3,640,640}的数组指针；   维度的个数，这里是4维(NCHW)

        // 推理计时
        auto start = chrono::high_resolution_clock::now(); //使用高精度时钟，记录开始时间

        const char* input_names[] = {"images"};           //定义输入名
        const char* output_names[] = {"output0"};           //定义输出名
        auto outputs = session.Run(                        //运行模型推理
            Ort::RunOptions{nullptr},                    //推理运行配置，参数默认
            input_names, &input_tensor, 1,              //输入节点名称，输入张量数量
            output_names, 1);                          //输出节点名称

        auto end = chrono::high_resolution_clock::now();      //使用高精度时钟记录结束时间
        float inference_time = chrono::duration<float, milli>(end - start).count();     //计算推理耗时，end-start，然后转化为毫秒的浮点数
        float fps = 1000.0 / inference_time;              //计算fps（Frames Per Second），也就是一秒能处理多少帧

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
