"""  
用于测试
统一 YOLO + PaddleOCR 预处理服务（本地双模型 · GPU 隔离模式）
============================================================

🎯 目标：
  提供一个轻量级 API，**并行执行 YOLO 手写检测与 PaddleOCR 文本识别**，可以输入文件夹
  将结果分别输出到指定目录，为后续融合模块（如 `fusion_pipeline_ocr_based`）准备输入。

⚙️ 架构特点：
  - **双模型本地执行**：
      • **PaddleOCR**：在当前进程调用（使用 GPU 2）
      • **YOLOv12**：通过 `conda run -n api_yolo` 启动子进程（使用 GPU 3）
  - **GPU 资源隔离**：
      • OCR 与 YOLO 分别绑定不同 GPU，避免显存竞争
      • `CUDA_VISIBLE_DEVICES` 在子进程中显式指定
  - **纯预处理服务**：
      • 本服务**不执行融合逻辑**，仅生成 YOLO 标签（.txt）和 OCR 结果（JSON + 图）
      • 输出目录结构兼容后续 `run_fusion_pipeline` 调用

📥 输入（JSON）：
  {
    "input_path": "/path/to/image_or_dir",   // 支持单图或目录
    "yolo_output_dir": "/path/to/yolo_out",  // YOLO 标签输出目录
    "ocr_output_dir": "/path/to/ocr_out"     // OCR 结果输出目录
  }

📤 输出：
  {
    "status": "success",
    "yolo": { "status": "success", "output_dir": "..." },
    "ocr":  { "status": "success", "total": N, "output_dir": "..." }
  }

⚠️ 依赖前提：
  1. Conda 环境 `api_yolo` 已安装 Ultralytics YOLO 并可执行 `yolo` 命令
  2. 机器至少有 **2 块 GPU**（GPU 2 给 PaddleOCR，GPU 3 给 YOLO）
  3. YOLO 模型路径有效：`/data/zmy/runs/detect/train9/weights/best.pt`
  4. 输出目录需可写（服务会自动创建）

📦 典型使用场景：
  - 作为 **批量预处理工具**：先生成 YOLO/OCR 中间结果，再离线运行融合
  - 作为 **API 流水线的第一阶段**：主服务调用本接口，再调用融合模块

作者：Zhumengying
日期：2025-12-17

🚀 启动命令：
    conda activate api_paddle
    cd /data/zmy/workspace/api/code
    uvicorn api2:app --host 0.0.0.0 --port 8000

🧪 测试命令：
    curl -X POST http://localhost:8000/unified/run \
      -H "Content-Type: application/json" \
      -d '{
        "input_path": "/data/zmy/workspace/YOLO/yolov12-main/zhuzhu/data/scan_7",
        "yolo_output_dir": "/data/zmy/workspace/api/out/yolo",
        "ocr_output_dir": "/data/zmy/workspace/api/out/ocr"
      }'
"""

import os
import asyncio
import subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from paddleocr import PaddleOCR  # 必须在 CUDA_VISIBLE_DEVICES 设置后导入

# ==============================
# 环境与 GPU 配置（必须在 import paddleocr 之前）
# ==============================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # PaddleOCR 使用 GPU 2

# ==============================
# 初始化 PaddleOCR 引擎（单例）
# ==============================
ocr_engine = PaddleOCR(
    lang="ch",
    det_limit_side_len=2048,
    det_limit_type="max",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

app = FastAPI(title="Unified YOLO + PaddleOCR API")

# ==============================
# 请求模型
# ==============================
class UnifiedRequest(BaseModel):
    input_path: str   # 图像文件 或 目录路径
    yolo_output_dir: str
    ocr_output_dir: str

# ==============================
# 异步包装 YOLO 子进程调用（避免阻塞）
# ==============================
async def run_yolo_async(input_path: str, output_dir: str):
    """在 api_yolo 环境中异步运行 YOLO"""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "conda", "run", "-n", "api_yolo",
        "yolo", "predict",
        "model=/data/zmy/runs/detect/train9/weights/best.pt",
        f"source={input_path}",
        "save=True",
        "save_txt=True",
        "conf=0.3",
        "save_conf=True",
        "agnostic_nms=True",
        f"project={output_dir}",
        "name=.",
        "exist_ok=True"
    ]
    # 使用 asyncio.create_subprocess_exec 更高效，但为简单用 run + asyncio.to_thread
    try:
        await asyncio.to_thread(
            subprocess.run,
            cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "3"},
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120
        )
        return {"status": "success", "output_dir": output_dir}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="YOLO timeout")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"YOLO failed: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO error: {str(e)}")

# ==============================
# PaddleOCR 是同步的，但可放入线程
# ==============================
async def run_ocr_async(input_path: str, output_dir: str):
    """异步运行 PaddleOCR（实际在线程中执行）"""
    try:
        results = await asyncio.to_thread(ocr_engine.predict, input=input_path)
        # 保存结果（也放在线程中）
        await asyncio.to_thread(_save_ocr_results, results, output_dir)
        return {"status": "success", "total": len(results), "output_dir": output_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR error: {str(e)}")

def _save_ocr_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for res in results:
        res.save_to_img(output_dir)
        res.save_to_json(output_dir)

# ==============================
# 统一 API 接口
# ==============================
@app.post("/unified/run")
async def run_unified(request: UnifiedRequest):
    input_path = request.input_path
    yolo_out = request.yolo_output_dir
    ocr_out = request.ocr_output_dir

    # 验证输入路径
    if not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail=f"Input path not found: {input_path}")

    # 并行执行 YOLO 和 OCR
    try:
        yolo_task = run_yolo_async(input_path, yolo_out)
        ocr_task = run_ocr_async(input_path, ocr_out)

        yolo_result, ocr_result = await asyncio.gather(yolo_task, ocr_task)

        return {
            "status": "success",
            "yolo": yolo_result,
            "ocr": ocr_result
        }
    except HTTPException:
        raise  # 透传已定义的 HTTP 错误
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    

# conda activate api_paddle
# cd /data/zmy/workspace/api/code
# uvicorn api2:app --host 0.0.0.0 --port 8000

# curl -X POST http://localhost:8000/unified/run \
#   -H "Content-Type: application/json" \
#   -d '{
#     "input_path": "/data/zmy/workspace/YOLO/yolov12-main/zhuzhu/data/scan_7",                                            
#     "yolo_output_dir": "/data/zmy/workspace/api/out/yolo",
#     "ocr_output_dir": "/data/zmy/workspace/api/out/ocr"
#   }'