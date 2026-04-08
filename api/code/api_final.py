"""
还没有改输入
输出是两个文件，subprocess调用YOLO，输入文件较多的话平均时长会更短，不用多次激活YOLO环境
端到端 YOLO + PaddleOCR 融合服务（本地双模型 · 临时中间文件模式）
==================================================================

🎯 目标：
  提供一个**完整端到端 API**：从原始图像输入 → 并行执行 YOLO 手写检测 + PaddleOCR 文本识别 →
  自动融合结果 → 输出结构化 JSON 到指定目录，**隐藏所有中间过程**。

⚙️ 核心流程：
  1. 接收用户请求：`input_path`（图像/目录） + `fusion_output_dir`（最终输出目录）
  2. 创建**隔离临时目录**（`/data/zmy/workspace/api/tmp/unified_api_xxx`）
  3. **并行执行**：
      • **YOLO**：通过 `conda run -n api_yolo` 子进程运行（GPU 3），输出标签到临时目录
      • **PaddleOCR**：在当前进程调用（GPU 2），输出 JSON 到临时目录
  4. 调用 `run_fusion_pipeline`，使用临时中间结果生成最终融合输出
  5. 返回成功状态，**临时目录暂不清理**（便于调试，生产环境可启用自动清理）

📌 关键设计：
  - **GPU 资源隔离**：OCR → GPU 2，YOLO → GPU 3，避免显存竞争
  - **临时中间文件**：YOLO 标签（.txt）和 OCR 结果（JSON）仅存在于临时目录
  - **兼容现有融合模块**：调用 `fusion_model_final.run_fusion_pipeline`，保持接口一致
  - **支持批量处理**：`input_path` 可为单图或图像目录

📥 请求格式（JSON）：
  {
    "input_path": "/path/to/image.jpg 或 /path/to/dir",
    "fusion_output_dir": "/path/to/final/output"
  }

📤 响应格式：
  {
    "status": "success",
    "fusion_output": "/path/to/final/output",
    "costtime": 4.821  // 端到端耗时（秒）
  }

⚠️ 依赖前提：
  1. Conda 环境 `api_yolo` 已安装 Ultralytics YOLO
  2. 机器至少有 **2 块 GPU**（GPU 2 给 PaddleOCR，GPU 3 给 YOLO）
  3. 目录 `/data/zmy/workspace/api/tmp` 存在且可读写
  4. YOLO 模型路径有效：`/data/zmy/runs/detect/train9/weights/best.pt`

📦 部署说明：
  - **调试友好**：默认不清理临时目录，便于排查中间结果
  - **生产就绪**：取消 `shutil.rmtree` 注释可启用自动清理
  - **后台运行**：支持 `nohup` 启动，适合长期服务

作者：Zhumengying (zmy)  
日期：2025-12-17

🚀 启动命令：
    conda activate api_paddle
    cd /data/zmy/workspace/api/code
    uvicorn api_final:app --host 0.0.0.0 --port 8000

🔁 后台运行：
    nohup uvicorn api_final:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &

🧪 测试命令：
    curl -X POST http://localhost:8000/unified/run \
      -H "Content-Type: application/json" \
      -d '{
        "input_path": "/data/zmy/workspace/YOLO/yolov12-main/zhuzhu/data/scan_7",
        "fusion_output_dir": "/data/zmy/workspace/api/out/fusion"
      }'

🛑 停止服务：
    ps -ef | grep "uvicorn api_final"  # 获取 PID
    kill -9 <PID>
"""

import os
import asyncio
import subprocess
import tempfile
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from paddleocr import PaddleOCR
import time
# from workspace.api.code.fusion_model_raw import run_fusion_pipeline
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
    det_limit_side_len=2048,
    det_limit_type="max",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

app = FastAPI(title="Unified YOLO + PaddleOCR API")

# ==============================
# 请求模型（仅需 input_path 和 fusion_output_dir）
# ==============================
class UnifiedRequest(BaseModel):
    input_path: str
    fusion_output_dir: str

# ==============================
# 异步运行 YOLO（用 GPU 3，输出到临时目录）
# ==============================
async def run_yolo_async(input_path: str, temp_yolo_dir: str):
    os.makedirs(temp_yolo_dir, exist_ok=True)
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
        f"project={temp_yolo_dir}",
        "name=.",
        "exist_ok=True"
    ]
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
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="YOLO timeout")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"YOLO failed: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO error: {str(e)}")

# ==============================
# 异步运行 OCR（输出到临时目录）
# ==============================
async def run_ocr_async(input_path: str, temp_ocr_dir: str):
    try:
        results = await asyncio.to_thread(ocr_engine.predict, input=input_path)
        await asyncio.to_thread(_save_ocr_results, results, temp_ocr_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR error: {str(e)}")

def _save_ocr_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for res in results:
        res.save_to_img(output_dir)
        res.save_to_json(output_dir)

# ==============================
# 推断图像目录（支持单图或目录）
# ==============================
def infer_image_dir(input_path: str) -> str:
    p = Path(input_path)
    return str(p.parent) if p.is_file() else input_path

# ==============================
# 主 API 接口
# ==============================
@app.post("/unified/run")
async def run_unified(request: UnifiedRequest):
    start_time = time.time()
    input_path = request.input_path
    fusion_out = request.fusion_output_dir

    if not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail=f"Input path not found: {input_path}")

    # 创建临时中间目录（可替换为固定路径如 /data/temp/...）
    temp_base = tempfile.mkdtemp(prefix="unified_api_", dir="/data/zmy/workspace/api/tmp")
    temp_yolo = os.path.join(temp_base, "yolo")
    temp_ocr = os.path.join(temp_base, "ocr")

    image_dir = infer_image_dir(input_path)
    yolo_label_dir = os.path.join(temp_yolo, "labels")  # YOLO 默认结构

    try:
        # 并行执行 YOLO 和 OCR 到临时目录
        await asyncio.gather(
            run_yolo_async(input_path, temp_yolo),
            run_ocr_async(input_path, temp_ocr)
        )

        # 执行融合（使用临时中间结果 + 用户指定的 fusion 输出）
        # run_fusion_pipeline(
        #     image_dir=image_dir,
        #     yolo_label_dir=yolo_label_dir,
        #     ocr_json_dir=temp_ocr,
        #     output_dir=fusion_out
        # )
        run_fusion_pipeline(
            image_dir=image_dir,
            yolo_dir=yolo_label_dir,
            ocr_dir=temp_ocr,
            output_dir=fusion_out
        )

        # 可选：清理临时目录（当前保留以便调试，如需自动清理可取消注释）
        # import shutil
        # shutil.rmtree(temp_base, ignore_errors=True)
        
        costtime = time.time() - start_time
        return {
            "status": "success",
            "fusion_output": fusion_out,
            "costtime": round(costtime, 3)
        }

    except HTTPException:
        raise
    except Exception as e:
        # 可选：出错时也清理临时目录
        # import shutil
        # shutil.rmtree(temp_base, ignore_errors=True)
        return {"status": "error", "detail": str(e)}
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    

# conda activate api_paddle
# cd /data/zmy/workspace/api/code
# uvicorn api_final:app --host 0.0.0.0 --port 8000
# nohup uvicorn api_final:app --host 0.0.0.0 --port 8000     # 在后台启动

# curl -X POST http://localhost:8000/unified/run \
#   -H "Content-Type: application/json" \
#   -d '{
#     "input_path": "/data/zmy/workspace/YOLO/yolov12-main/zhuzhu/data/scan_7",                                            
#     "fusion_output_dir": "/data/zmy/workspace/api/out/fusion"
#   }'

# 停止进程
# kill 358424            kill -9 358424  强制终止
# ps -fp 358424   之后查看一下