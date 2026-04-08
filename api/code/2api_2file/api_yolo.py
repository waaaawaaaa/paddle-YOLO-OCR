"""
YOLOv12 手写体检测服务 API（内存传递模式）
------------------------------------------------
- 功能：接收图像路径，执行 YOLO 推理，返回归一化检测框（不保存中间文件）
- 设计目标：轻量、高效，专用于与 OCR 融合模块（fusion 函数）对接
- 数据流：调用方传入本地图像路径 → 本服务加载图像 → YOLO 推理 → 返回 JSON 格式检测结果
- 注意事项：
    1. 本服务**不保存任何预测结果或图像**，所有处理基于内存
    2. 调用方必须确保 image_path 在本服务所在机器上可访问
    3. 本服务端口（如 8001）需与主 OCR 接口中调用 YOLO 的地址保持一致
    4. 模型在服务启动时加载一次，避免重复加载开销

Author: Zhumenying
Date: 2025-12-23

运行方式（示例）：
    conda activate api_yolo
    cd /data/zmy/workspace/api/code/2api_2file
    CUDA_VISIBLE_DEVICES=2 uvicorn api_yolo:app --host 0.0.0.0 --port 8001
"""
import os

import shutil
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO


app = FastAPI(title="YOLOv12 Detection Service")

# 全局加载模型（服务启动时加载一次）
MODEL_PATH = "/data/zmy/runs/detect/train9/weights/best.pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ YOLO model loaded on device: {model.device}")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    model = None

class YOLOPredictRequest(BaseModel):
    image_path: str  # 临时图像路径（由主 API 传入）

@app.post("/predict")
async def predict(request: YOLOPredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")

    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=400, detail=f"Image not found: {request.image_path}")

    try:
        # 创建临时输出目录
        temp_out = tempfile.mkdtemp(prefix="yolo_service_")
        
        # 执行推理（自动保存 labels）
        results = model(
            source=request.image_path,
            conf=0.3,
            agnostic_nms=True,
            save=True,
            save_txt=True,
            project=temp_out,
            name=".",
            exist_ok=True
        )
        
        # 返回 labels 目录路径
        label_dir = os.path.join(temp_out, "labels")
        return {"label_dir": label_dir}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO inference error: {str(e)}")
    
# conda activate api_yolo
# cd /data/zmy/workspace/api/code/api_2
# uvicorn api_yolo:app --host 0.0.0.0 --port 8001