# YOLO11 C++ Inference

基于ONNX Runtime和OpenCV的YOLO11目标检测C++推理系统

## 功能
- 图片目标检测
- 视频逐帧检测
- 推理速度统计（FPS）
- NMS后处理

## 依赖
- OpenCV 4.x
- ONNX Runtime 1.20.1
- CMake 3.10+

## 编译
```bash
mkdir build && cd build
cmake ..
make
```

## 运行
```bash
./yolo_inference
```

## 性能
- CPU推理：约10-15 FPS
- 模型：YOLOv11n
