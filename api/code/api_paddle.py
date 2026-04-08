"""
PaddleOCR 独立服务 API（PP-OCRv5 · 目录批处理模式）
==================================================

🎯 目标：
  提供一个轻量级 HTTP 接口，**直接调用 PaddleOCR 对单图或整个目录执行 OCR**，
  并自动保存可视化图像和结构化 JSON 结果，**无需额外脚本**。

✨ 核心特性：
  - **目录级批处理**：`input_path` 可为单个图像（.jpg/.png）或包含多图的目录
  - **结果自动落盘**：每张图生成：
        • `{stem}_vis.png`：带检测框的可视化图
        • `{stem}_res.json`：结构化识别结果（含文本、坐标、置信度）
  - **GPU 加速**：固定使用 GPU 2（通过 `CUDA_VISIBLE_DEVICES=2`）
  - **零中间逻辑**：直接透传 PaddleOCR 原生输出，无后处理

📥 请求格式（JSON）：
  {
    "input_path": "/path/to/image.jpg 或 /path/to/image_dir",
    "output_dir": "/path/to/output"   // 必须可写
  }

📤 响应格式：
  {
    "status": "success",
    "total": 24,          // 处理的图像数量
    "output_dir": "/path/to/output"
  }

⚠️ 重要前提：
  1. **自定义 PaddleOCR 封装**：  
     本服务依赖的 `PaddleOCR` 类**非标准开源版**，而是包含以下方法的**自定义封装**：
        • `.predict(input=...)` 支持目录输入
        • `Result` 对象提供 `.save_to_img()` 和 `.save_to_json()`
     （标准 PaddleOCR 返回原生 Python 列表，无这些方法）
  2. 输出目录需提前存在或可自动创建
  3. 服务启动前已激活 `api_paddle` 环境

📦 典型用途：
  - 作为独立 OCR 微服务
  - 为 YOLO+OCR 融合系统提供 OCR 输入源
  - 快速批量处理扫描文档

作者：Zhumengying (zmy)  
日期：2025-12-17

🚀 启动命令：
    conda activate api_paddle
    uvicorn api_paddle:app --host 0.0.0.0 --port 8000

🧪 测试命令：
    curl -X POST http://localhost:8000/ocr/ \
      -H "Content-Type: application/json" \
      -d '{
        "input_path": "/data/zmy/workspace/YOLO/yolov12-main/zhuzhu/data/scan_7",
        "output_dir": "/data/zmy/workspace/api/tmp/ocr_out"
      }'
"""
# zhumengying    2025/12/17
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# # Initialize PaddleOCR instance
# from paddleocr import PaddleOCR
# ocr = PaddleOCR(
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False)

# # Run OCR inference on a sample image 
# result = ocr.predict(
#     input="/data/zmy/workspace/YOLO/yolov12-main/zhuzhu/data/real_test/scan")

# # Visualize the results and save the JSON results
# for res in result:
#     res.print()
#     res.save_to_img("output")
#     res.save_to_json("output")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 必须在 import paddleocr 之前

from fastapi import FastAPI
from pydantic import BaseModel
from paddleocr import PaddleOCR

# ======================
# 初始化 OCR 引擎（全局单例）
# ======================
ocr_engine = PaddleOCR(
    lang="ch",
    det_limit_side_len=2048,
    det_limit_type="max",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

app = FastAPI(title="PaddleOCR PP-OCRv5 API")

# ======================
# 请求体模型
# ======================
class OCRRequest(BaseModel):
    input_path: str   # 图像文件路径 或 图像目录路径
    output_dir: str   # 输出目录（保存 vis 和 json）

# ======================
# API 路由
# ======================
@app.post("/ocr")
def run_ocr(request: OCRRequest):
    """
    Run OCR on a single image or a folder (PP-OCRv5).
    Supports directory input directly (as verified in your local script).
    """
    input_path = request.input_path
    output_dir = request.output_dir

    # 调用 PaddleOCR（支持目录，如你本地脚本所示）
    results = ocr_engine.predict(input=input_path)

    # 保存结果
    for res in results:
        res.save_to_img(output_dir)
        res.save_to_json(output_dir)

    return {
        "status": "success",
        "total": len(results),
        "output_dir": output_dir
    }

# 打开接口
# conda activate api_paddle
# uvicorn api_paddle:app --host 0.0.0.0 --port 8000
# 在另一个终端
# curl -X POST http://localhost:8000/ocr/   -H "Content-Type: application/json"   -d '{ 
#     "input_path": "/data/zmy/workspace/YOLO/yolov12-main/zhuzhu/data/scan_7",
#     "output_dir": "/data/zmy/workspace/api/tmp/ocr_out"
#   }'