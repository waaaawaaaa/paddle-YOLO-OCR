"""
统一 OCR + YOLO 检测融合 API（临时文件传递模式）
=================================================

📌 注意：本版本采用**临时文件中转**方式，而非纯内存传递。
       因 `fusion_model_final` 当前依赖文件系统输入，故 YOLO 和 OCR 结果会暂存于临时目录，
       融合完成后自动清理。端到端延迟仍控制在 5 秒左右。 消耗显存6104M 第一次运行会有冷启动，多运行几次就稳定了

核心流程：
  1. 接收图像上传 + unique_id
  2. 创建隔离临时目录（/data/zmy/workspace/api/tmp/upload_api_xxx）
  3. 并行执行：
      - 调用 **远程 YOLO 服务**（localhost:8001）→ 服务返回 label_dir → 本服务复制 labels 到本地临时目录
      - 调用 **本地 PaddleOCR** → 生成文本+坐标 → 保存为临时 OCR 文件（JSON + 可视化图）
  4. 将临时目录路径传给 `run_fusion_pipeline`
  5. 返回结构化结果，并清理所有临时文件

设计特点：
  ✅ **临时隔离**：每个请求独享临时目录，避免并发冲突
  ✅ **自动清理**：无论成功/失败，`finally` 块确保无残留
  ✅ **可追溯**：响应包含 unique_id 与耗时，便于监控
  ✅ **兼容现有融合模块**：适配 `fusion_model_final` 对文件路径的依赖

⚠️ 依赖前提：
  1. YOLO 服务必须运行在 `http://localhost:8001/predict`，且返回结构含 `"label_dir"`
  2. YOLO 服务与本服务**共享同一主机文件系统**（因需复制 label_dir）
  3. 目录 `/data/zmy/workspace/api/tmp` 必须存在且可读写
  4. PaddleOCR 使用 GPU 2（通过 `CUDA_VISIBLE_DEVICES=2` 指定）

输入：
  - unique_id: 字符串，用于标识请求（如原始文件名）
  - image: 图像文件（JPG/PNG 等）

输出：
  - unique_id: 原样返回
  - items: 融合后的结构化列表（由 `run_fusion_pipeline` 生成）
  - costtime: 处理耗时（秒，3 位小数）
  - message: 状态信息

作者：Zhumengying (zmy)  
日期：2025-12-23

🚀 启动命令：
    conda activate api_paddle
    cd /data/zmy/workspace/api/code/2api_2file
    uvicorn api_final_ff:app --host 0.0.0.0 --port 8000

🧪 测试命令：
    curl -X POST "http://localhost:8000/unified/upload" \
         -F "unique_id=zmy" \
         -F "image=@/data/zmy/workspace/YOLO/yolov12-main/zhuzhu/data/scan/mmexport1760952142264_1.jpg"
"""

import os
import asyncio
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from paddleocr import PaddleOCR
import time
import httpx  # 新增依赖

from fusion_model_final import run_fusion_pipeline

# ==============================
# 环境配置（PaddleOCR 用 GPU 2）
# ==============================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ==============================
# 初始化 PaddleOCR
# ==============================
ocr_engine = PaddleOCR(
    lang="ch",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

app = FastAPI(title="Unified YOLO + PaddleOCR API")

# ==============================
# YOLO 服务配置
# ==============================
YOLO_SERVICE_URL = "http://localhost:8001/predict"  # YOLO 服务地址

# ==============================
# 异步运行 YOLO（调用 HTTP 服务）
# ==============================
async def run_yolo_async(input_path: str, temp_yolo_dir: str):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 调用 YOLO 服务
            response = await client.post(YOLO_SERVICE_URL, json={"image_path": input_path})
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"YOLO service error: {response.text}")
            
            label_dir = response.json()["label_dir"]
            
        # 复制 labels 到目标目录
        target_label_dir = os.path.join(temp_yolo_dir, "labels")
        os.makedirs(target_label_dir, exist_ok=True)
        if os.path.exists(label_dir):
            for file in os.listdir(label_dir):
                shutil.copy(os.path.join(label_dir, file), target_label_dir)
                
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"YOLO service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO error: {str(e)}")

# ==============================
# 异步运行 OCR（修复调用方式！）
# ==============================
async def run_ocr_async(input_path: str, temp_ocr_dir: str):
    try:
        # ✅ 修复：直接调用 ocr_engine，不是 .predict()
        results = await asyncio.to_thread(ocr_engine.ocr, input_path)
        await asyncio.to_thread(_save_ocr_results, results, temp_ocr_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR error: {str(e)}")

def _save_ocr_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for res in results:
        res.save_to_img(output_dir)
        res.save_to_json(output_dir)

# ==============================
# 主 API 接口
# ==============================
@app.post("/unified/upload")
async def upload_and_run(
    unique_id: str = Form(...),
    image: UploadFile = File(...)
):
    start_time = time.time()
    temp_base = tempfile.mkdtemp(prefix="upload_api_", dir="/data/zmy/workspace/api/tmp")
    temp_image_path = os.path.join(temp_base, "uploaded_img" + Path(image.filename).suffix)
    temp_yolo = os.path.join(temp_base, "yolo")
    temp_ocr = os.path.join(temp_base, "ocr")

    try:
        # 保存上传的图像
        with open(temp_image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # 并行运行 YOLO 和 OCR
        await asyncio.gather(
            run_yolo_async(temp_image_path, temp_yolo),
            run_ocr_async(temp_image_path, temp_ocr)
        )

        # 调用融合逻辑
        items = run_fusion_pipeline(
            image_dir=os.path.dirname(temp_image_path),
            yolo_dir=os.path.join(temp_yolo, "labels"),
            ocr_dir=temp_ocr,
            output_dir=None
        )

        total_time = time.time() - start_time
        return {
            "unique_id": unique_id,
            "items": items,
            "costtime": round(total_time, 3),
            "message": "success" if items else "no matches"
        }

    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail={
                "unique_id": unique_id,
                "error": str(e),
                "costtime": round(total_time, 3)
            }
        )
    finally:
        # 清理临时文件
        shutil.rmtree(temp_base, ignore_errors=True)


# conda activate api_paddle
# cd /data/zmy/workspace/api/code/api_2
# uvicorn api_final_ff:app --host 0.0.0.0 --port 8000