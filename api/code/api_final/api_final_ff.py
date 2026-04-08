"""
统一 OCR + YOLO 检测融合 API（本地paddle + subprocess yolo + 临时文件模式）
==========================================================

🎯 目标：
  接口写成后端要求一张一张输入的格式
  接收扫描表格图像，融合 PaddleOCR（文本）与 YOLO（手写数量）结果，
  输出结构化医疗物资清单，**端到端延迟约 12 秒**。

⚙️ 架构特点：
  - **双模型本地执行**：
      • PaddleOCR：在当前进程调用（GPU 2）
      • YOLOv12：通过 `conda run -n api_yolo yolo predict` 子进程执行（GPU 3）
  - **临时文件中转**：
      • YOLO 输出 `.txt` 标签到临时目录
      • OCR 结果保存为 JSON + 可视化图
      • 融合模块 `run_fusion_pipeline` 从文件读取输入
  - **资源隔离**：
      • OCR 与 YOLO 分别使用 **GPU 2 和 GPU 3**，避免显存竞争
      • 每个请求独享临时目录，支持并发安全

📊 性能特性：
  - 端到端处理时间：**约 12 秒**
  - 自动清理临时文件，无磁盘残留

⚠️ 依赖前提：
  1. Conda 环境 `api_yolo` 已安装 Ultralytics YOLO 且可执行 `yolo` 命令
  2. YOLO 模型路径有效：`/data/zmy/runs/detect/train9/weights/best.pt`
  3. 目录 `/data/zmy/workspace/api/tmp` 存在且可读写
  4. 机器至少有 **2 块 GPU**（GPU 2 给 PaddleOCR，GPU 3 给 YOLO）

📥 输入：
  - `unique_id`：请求唯一标识（建议用原始文件名）
  - `image`：上传的 JPG/PNG 图像

📤 输出：
  {
    "unique_id": str,
    "items": List[{
        "issue_id": str,      # 项目编码
        "issue_name": str,    # 项目名称
        "issue_number": str,  # 手写数量（来自 YOLO）
        "issue_box": [x1, y1, x2, y2]
    }],
    "costtime": float,        # 总耗时（秒，3 位小数）
    "message": "success" or error detail
  }

📦 所属项目：医疗表格智能结构化系统  
作者：Zhumengying (zmy)  
日期：2025-12-23

🚀 启动命令：
    conda activate api_paddle
    cd /data/zmy/workspace/api/code/api_final
    uvicorn api_final_ff:app --host 0.0.0.0 --port 8000

🧪 测试命令：
    curl -X POST "http://localhost:8000/unified/upload" \
         -F "unique_id=zmy" \
         -F "image=@/data/zmy/workspace/YOLO/yolov12-main/zhuzhu/data/scan/mmexport1760952142264_1.jpg"

🔁 后台运行（可选）：
    nohup uvicorn api_final_ff:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &

🛑 停止服务：
    ps -ef | grep "uvicorn api_final_ff"  # 找 PID
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
from fastapi import File, UploadFile, Form
import shutil
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
    # det_limit_side_len=2048,
    # det_limit_type="max",
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
        print('model spend time:',time.time() - start_time)

        # 调用融合逻辑（只返回 items）
        items = run_fusion_pipeline(
            image_dir=os.path.dirname(temp_image_path),
            yolo_dir=os.path.join(temp_yolo, "labels"),
            ocr_dir=temp_ocr,
            output_dir=None  # 不保存
        )

        costtime = time.time() - start_time
        return {
            "unique_id": unique_id,
            "items": items,
            "costtime": round(costtime, 3),
            "message": "success"
        }

    except HTTPException as http_exc:
        # 透传已知 HTTP 错误
        costtime = time.time() - start_time
        raise HTTPException(
            status_code=http_exc.status_code,
            detail={
                "unique_id": unique_id,
                "error": http_exc.detail,
                "costtime": round(costtime, 3)
            }
        )

    except Exception as e:
        costtime = time.time() - start_time
        # 返回错误结构（让 FastAPI 自动返回 500）
        error_response = {
            "unique_id": unique_id,
            "error": str(e),
            "costtime": round(costtime, 3)
        }
        raise HTTPException(status_code=500, detail=error_response)

    finally:
        # 清理所有临时文件
        shutil.rmtree(temp_base, ignore_errors=True)
# @app.post("/unified/upload")
# async def upload_and_run(
#     unique_id: str = Form(...),
#     image: UploadFile = File(...)
# ):
#     start_time = time.time()
#     temp_base = tempfile.mkdtemp(prefix="upload_api_", dir="/data/zmy/workspace/api/tmp")
#     temp_image_path = os.path.join(temp_base, "uploaded_img" + Path(image.filename).suffix)
#     temp_yolo = os.path.join(temp_base, "yolo")
#     temp_ocr = os.path.join(temp_base, "ocr")

#     try:
#         # 保存上传的图像
#         with open(temp_image_path, "wb") as f:
#             shutil.copyfileobj(image.file, f)

#         # === 分别运行 YOLO 和 OCR，以便获取各自耗时 ===
#         yolo_start = time.time()
#         await run_yolo_async(temp_image_path, temp_yolo)
#         yolo_time = time.time() - yolo_start

#         ocr_start = time.time()
#         await run_ocr_async(temp_image_path, temp_ocr)
#         ocr_time = time.time() - ocr_start

#         fusion_start = time.time()
#         items = run_fusion_pipeline(
#             image_dir=os.path.dirname(temp_image_path),
#             yolo_dir=os.path.join(temp_yolo, "labels"),
#             ocr_dir=temp_ocr,
#             output_dir=None
#         )
#         fusion_time = time.time() - fusion_start

#         total_time = time.time() - start_time

#         # 打印详细耗时（带 flush=True 确保输出）
#         print(f"📊 Timing for {unique_id}:")
#         print(f"   YOLO:      {yolo_time:.3f}s")
#         print(f"   OCR:       {ocr_time:.3f}s")
#         print(f"   Fusion:    {fusion_time:.3f}s")
#         print(f"   Total:     {total_time:.3f}s", flush=True)

#         return {
#             "unique_id": unique_id,
#             "items": items,
#             "costtime": round(total_time, 3),
#             "message": "success"
#         }

#     except HTTPException as http_exc:
#         costtime = time.time() - start_time
#         raise HTTPException(
#             status_code=http_exc.status_code,
#             detail={
#                 "unique_id": unique_id,
#                 "error": http_exc.detail,
#                 "costtime": round(costtime, 3)
#             }
#         )
#     except Exception as e:
#         costtime = time.time() - start_time
#         raise HTTPException(
#             status_code=500,
#             detail={
#                 "unique_id": unique_id,
#                 "error": str(e),
#                 "costtime": round(costtime, 3)
#             }
#         )
#     finally:
#         shutil.rmtree(temp_base, ignore_errors=True)

        
# conda activate api_paddle
# cd /data/zmy/workspace/api/code/api_final
# uvicorn api_final_ff:app --host 0.0.0.0 --port 8000
# nohup uvicorn api_final:app --host 0.0.0.0 --port 8000     # 在后台启动

# curl -X POST "http://localhost:8000/unified/upload" \
#   -F "unique_id=zmy" \
#   -F "image=@/data/zmy/workspace/YOLO/yolov12-main/zhuzhu/data/scan/mmexport1760952142264_1.jpg"

# 停止进程
# kill 358424            kill -9 358424  强制终止
# ps -fp 358424   之后查看一下