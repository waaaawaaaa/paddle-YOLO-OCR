"""
统一 OCR + YOLO 检测融合 API（内存传递模式）
=============================================

- 功能：接收图像上传 → 并行调用 PaddleOCR（本地）和 YOLO 服务（远程）→ 内存中融合结果 → 返回结构化识别项
- 设计原则：
    ✅ **不保存任何中间文件**（YOLO、OCR、融合结果均在内存中处理）
    ✅ **高性能**：OCR 与 YOLO 并行执行，端到端延迟控制在 2~3 秒  消耗显存6104M
    ✅ **轻耦合**：YOLO 以独立服务运行（端口 8001），本服务通过 HTTP 异步调用
    ✅ **可追溯**：每个请求携带 unique_id，便于日志追踪与结果对齐
- 输入：
    - image: 上传的图像文件（支持常见格式）
    - unique_id: 用户指定的唯一标识（如原始文件名，用于结果回传）
- 输出：
    - unique_id
    - items: 融合后的结构化结果（如项目名、单位、数量、坐标等）
    - costtime: 端到端处理耗时（秒，精确到毫秒）
    - message: 成功或失败状态说明

⚠️ 注意事项：
  1. YOLO 服务必须已启动并监听 `http://localhost:8001/predict`
  2. 本服务与 YOLO 服务需共享同一文件系统（因传递的是本地路径）
  3. 图像临时保存于 `/data/zmy/workspace/api/tmp`，处理完成后自动清理
  4. PaddleOCR 使用 GPU 2（通过 CUDA_VISIBLE_DEVICES=2 指定）

Author: Zhumenying
Date: 2025-12-24

启动命令示例：
    conda activate api_paddle
    cd /data/zmy/workspace/api/code/2api_0file
    uvicorn api_final_ff:app --host 0.0.0.0 --port 8000

运行命令示例：
curl -X POST "http://localhost:8000/unified/upload"   -F "unique_id=zmy"   -F "image=@/data/zmy/workspace/YOLO/yolov12-main/zhuzhu/data/scan/mmexport1760952142264_1.jpg"
"""


import os
import asyncio
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from paddleocr import PaddleOCR
import time
import httpx

from fusion_model_final import run_fusion_pipeline

# GPU 配置（PaddleOCR 用 GPU 2）
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 初始化 OCR 引擎
ocr_engine = PaddleOCR(
    lang="ch",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

app = FastAPI(title="Unified YOLO + PaddleOCR API")

# YOLO 服务地址
YOLO_SERVICE_URL = "http://localhost:8001/predict"

# 异步调用 YOLO 服务（返回 detections 列表）
async def run_yolo_async(image_path: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(YOLO_SERVICE_URL, json={"image_path": image_path})
        if resp.status_code != 200:
            raise HTTPException(500, detail=f"YOLO error: {resp.text}")
        return resp.json()["detections"]

# 异步运行 OCR（返回内存结果，不保存文件）
async def run_ocr_async(image_path: str):
    try:
        return await asyncio.to_thread(ocr_engine.ocr, image_path)
    except Exception as e:
        print("OCR parsing error:")
        traceback.print_exc()
        raise HTTPException(500, detail=f"OCR error: {str(e)}")

# 主 API 接口
@app.post("/unified/upload")
async def upload_and_run(unique_id: str = Form(...), image: UploadFile = File(...)):
    start = time.time()
    temp_dir = tempfile.mkdtemp(prefix="api_", dir="/data/zmy/workspace/api/tmp")
    img_path = os.path.join(temp_dir, f"img{Path(image.filename).suffix}")

    try:
        # 保存上传图像
        with open(img_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # 并发执行 YOLO 和 OCR
        yolo_task = run_yolo_async(img_path)
        ocr_task = run_ocr_async(img_path)
        yolo_dets, ocr_results = await asyncio.gather(yolo_task, ocr_task)
        # 在 await ocr_task 后
        print("YOLO结果")
        print(yolo_dets)
        # 打印 PaddleOCR 结果（只显示前3项 + 类型）
        print("\nrec_texts\n")
        print(ocr_results[0]['rec_texts'])
        # 融合（传内存数据）
        items = run_fusion_pipeline(
            image_path=img_path,
            yolo_detections=yolo_dets,
            ocr_results=ocr_results[0],
            output_dir=None
        )

        cost = time.time() - start
        return {
            "unique_id": unique_id,
            "items": items,
            "costtime": round(cost, 3),
            "message": "success" if items else "no matches"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail={"unique_id": unique_id, "error": str(e), "costtime": round(time.time() - start, 3)})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

