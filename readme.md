# 📄 手术单识别项目

> **Unified YOLO + PaddleOCR Fusion API**  
> 融合 PaddleOCR 与自研 YOLO 模型，实现手术单表格中印刷体文字与手写数字的高精度联合识别。

---

## 项目简介

- **项目名称**：Unified YOLO + PaddleOCR Fusion API  
- **一句话介绍**：融合 PaddleOCR 与自研 YOLO 模型，基于空间坐标实现手术单表格中印刷体文字与手写数字的高精度联合识别。  
- **项目状态**：**稳定版（Production Ready）**，已部署于本地服务器，支持并发请求与结构化输出。

---

## 功能概述

### 主要功能
- 自动识别手术单中的项目编码、名称（印刷体，由 PaddleOCR 处理）
- 高精度检测手写数量（如 `"2+1"`、`"0.5"`，由 YOLOv12 模型处理）
- 基于坐标空间融合，输出结构化 JSON（含 bbox、数量、ID、名称）
- 内存级处理，无中间文件残留，支持可视化结果生成

### 使用场景
- 医院手术耗材清单自动化录入  
- 手术单据电子化归档  


### 解决的问题
| 问题 | 解决方案 |
|------|----------|
| PaddleOCR 对手写数字识别率 < 30% | 引入 YOLOv12 专用检测模型，识别率 > 90% |
| 印刷体与手写体混合表格无法结构化 | 基于几何坐标融合，实现字段对齐 |
| 人工录入低效易错 | 端到端自动化，准确率 > 90% |

---

## 技术栈

### 使用的技术和框架
- **YOLOv12**（自定义检测模型，用于手写数字定位）
- **PaddleOCR**（用于印刷体文字识别）
- **FastAPI**（主 API 服务）
- **PyTorch**（YOLO 模型训练与推理）
- **CUDA + cuDNN**（GPU 加速）

### 依赖的库和工具
- `ultralytics`（YOLO API）
- `paddlepaddle-gpu`
- `paddleocr`
- `opencv-python`, `Pillow`, `numpy`, `scikit-learn`
- `uvicorn`（ASGI 服务器）

---

## 系统要求

### 操作系统
- Linux（推荐 Ubuntu 24.04 ）

### 最低硬件要求
- **GPU**：NVIDIA GPU（显存 ≥ 8 GB，支持 CUDA 11.8+）  
  - PaddleOCR 使用 **GPU 2**  
  - YOLO 使用 **GPU 2**

### 软件环境
- Python = 3.11  
- CUDA = 12.6

---


### 环境的安装 
##### paddle环境与YOLO环境存在冲突，需要隔离。 

paddle：
https://github.com/PaddlePaddle/PaddleOCR

```bash
conda create -n paddle python=3.11
conda activate paddle
 python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
 python -m pip install paddleocr
```

YOLO：
https://github.com/sunsmarterjie/yolov12

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
conda create -n yolov12 python=3.11
conda activate yolov12
pip install -r requirements.txt
pip install -e .
```
### 常见问题及解决方案

| 问题 | 解决方案 |
|------|--------|
| **YOLO 类别索引越界** | 检查 `data.yaml` 中 `nc`（类别数）是否与标签文件中的类别索引范围一致（标签索引必须 `< nc`） |
| **GPU 显存不足** | 降低 `batch=4` 或 `imgsz=1024`；或限制并发请求数；也可在 YOLO 服务端启用 `--half`（FP16 推理） |
| **PaddleOCR GPU 占用但使用率为 0%** | 确认调用时指定 `--device gpu:0`；检查是否设置了 `CUDA_VISIBLE_DEVICES`；确保 PaddlePaddle 与 CUDA 版本兼容 |

---

## 使用说明

### 基本使用

启动服务：
先启动YOLO接口
```bash
conda activate api_yolo
cd /api/code/2api_0file
CUDA_VISIBLE_DEVICES=2 uvicorn api_yolo:app --host 0.0.0.0 --port 8001
```
再启动主接口
```bash
conda activate api_paddle
cd /api/code/2api_0file
uvicorn api_final_ff:app --host 0.0.0.0 --port 8000
```
调用接口（详见 [API 文档](api/code/2api_0file/api_doc.md)）。

```bash
curl -X POST "http://localhost:8000/unified/upload" \
  -F "unique_id=zmy" \
  -F "image=@/data/zmy/workspace/YOLO/yolov12-main/zhuzhu/data/scan/mmexport1760952142264_1.jpg"
```
---
api的准确性是通过自己手动比对，进行统计的

## 高级用法

- 自定义 YOLO 模型路径（通过 `YOLO_SERVICE_URL` 环境变量）  
- 启用可视化结果返回（通过配置开关）  
- 调整融合匹配阈值（如垂直对齐容差、水平偏移容忍度）

---

### 历史接口：
##### 代码路径：/api/code

| 路径 | 改进 |
|------|------------|
|api_final.py|输出是两个文件，subprocess调用YOLO，输入文件较多的话平均时长会更短，不用多次激活YOLO环境|
|api_final/api_final_ff.py|统一 OCR + YOLO 检测融合 API（本地paddle + subprocess yolo + 临时文件模式）终端输出|
|2api_2file/api_final_ff.py|统一 OCR + YOLO 检测融合 API（临时文件传递模式）  YOLO也用api调用|
|2api_0file/api_final_ff.py|统一 OCR + YOLO 检测融合 API（内存传递模式）|
---

## 项目流程

### 数据集的构建
基于手术单模版，手动填上可能出现的值，构造一批数据，再经过扫描，整理成数据集

将扫描的PDF转成JPG：process/pdf2jpg.py
---

### 数据集的构成
#### label
labelimg： （打标注的工具）
[https://zhuanlan.zhihu.com/p/550021453](https://zhuanlan.zhihu.com/p/550021453)

labelimg 闪退的解决方法
float error：[https://blog.csdn.net/m0_74232237/article/details/130985914](https://blog.csdn.net/m0_74232237/article/details/130985914)

YOLO全流程使用

[https://blog.csdn.net/weixin_48870215/article/details/144458659](https://blog.csdn.net/weixin_48870215/article/details/144458659)


后面需要添加新的训练数据时，可以通过训练好的一版模型跑出检测结果，再手动修改预测结果得到标签，会快一些

### 模型训练与调用

先将打好标签的数据整理成训练所需要的格式
/process/split_data.py
再根据标签的类型写好data.yaml文件

训练YOLO模型，加了一部分数据增强的参数
```bash
nohup yolo task=detect mode=train     model=yolov12.yaml     data=workspace/YOLO/yolov12-main/zhuzhu/data/data_real/datasets1/data.yaml     pretrained=./yolo12m.pt     imgsz=1280     batch=4     epochs=8000     patience=1000     device=0     fliplr=0.0     flipud=0.0     degrees=2.0     translate=0.05     scale=0.3     hsv_h=0.015     hsv_s=0.7     hsv_v=0.4     mosaic=0     mixup=0     erasing=0.2     copy_paste=0.0   > train.log 2>&1 &

# 参数imgsz是指输入YOLO的图像会先压缩成一个1280*1280尺寸的图，按比例压缩，不足的位置补零

# 基于GPU的显存，以及尽可能大的尺寸下，设置合适的batch

# 剩余的参数是一些数据增强的参数，基于任务的考虑，不会有太大的旋转，以及翻转，拼接之类的。但可能存在拍照管线的变化，所以数据增强主要在颜色通道上
```

需要统计一下模型的正确率
可以直接调用YOLO的val，会得到正确率的信息
```bash
yolo val model=/model/train9/weights/best.pt data=workspace/YOLO/yolov12-main/zhuzhu/data/data_real/datasets1/data.yaml save_txt=True save=True
# 会用到data.yaml里面的val数据
# 但可能缺少更精细的指标
```

自己手动写一个代码计算正确率，首先要拿到label
process/ACC.py

调用训练好的模型
```bash
yolo predict model=/model/train9/weights/best.pt source='手术单据实拍数据' save=True save_txt=True conf=0.3 save_conf=True agnostic_nms=True
```
调用paddle模型
```bash
paddleocr ocr   -i /手术单据实拍数据   --use_doc_orientation_classify False   --use_doc_unwarping False   --use_textline_orientation False   --save_path /api/code/api_3/runs/detect/predict/paddle_scan   --device gpu:0
```
##### 剩下的代码
| 路径 | 描述 |
|------|------------|
|data_pdf|  原始扫描的PDF数据 | 
|process/scan_img.py|将拍摄的真实图像转成扫描版|
|process/copy_txt.py|由于手术单上手写部分的位置基本不变，复制txt文件，可以更快速打上标签|
|process/json_txt_kmeans.py|   基于YOLO结果使用Kmeans融合paddle结果|
|/process/fusion_model_final.py|   基于paddle的'数量'融合YOLO结果|
---


> **最后更新**：2025-12-30
