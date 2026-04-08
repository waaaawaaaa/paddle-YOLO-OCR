"""
YOLO 手写检测独立服务 API（Conda 环境隔离 · GPU 3 专用）
========================================================

🎯 目标：
  提供一个安全、隔离的 HTTP 接口，**通过 conda 环境 `api_yolo` 调用 YOLOv8/v12 模型**，
  对单图或整个目录执行手写数量检测，输出图像 + 标签文件，**不依赖主服务环境**。

✨ 核心特性：
  - **环境完全隔离**：通过 `conda run -n api_yolo` 启动独立 Python 环境，避免依赖冲突
  - **GPU 资源独占**：强制使用 **物理 GPU 3**（通过 `CUDA_VISIBLE_DEVICES=3`）
  - **批量处理支持**：`input_path` 可为单张图像（.jpg/.png）或包含多图的目录
  - **标准 YOLO 输出**：在 `output_dir` 生成：
        • `images/`：带检测框的可视化图
        • `labels/`：归一化坐标标签（.txt，格式：class cx cy w h）
  - **超时保护**：120 秒硬性超时，防止单次请求阻塞服务

📥 请求格式（JSON）：
  {
    "input_path": "/path/to/image.jpg 或 /path/to/image_dir",
    "output_dir": "/path/to/yolo_output"   // 必须可写
  }

📤 响应格式：
  {
    "status": "success",
    "output_dir": "/path/to/yolo_output",
    "message": "Processed ..."
  }

⚠️ 依赖前提：
  1. Conda 环境 `api_yolo` 已安装 Ultralytics YOLO 并可执行 `yolo` 命令
  2. 指定模型路径有效：`/data/zmy/runs/detect/train9/weights/best.pt`
  3. 机器至少有 4 块 GPU（GPU 3 可用）
  4. 输出目录有写权限

📦 系统定位：
  - 作为 **YOLO 微服务**，供主融合 API（如 `api_final`）或独立调用
  - 与 PaddleOCR 服务（GPU 2）并行运行，**资源无竞争**
  - 适用于预处理或端到端流水线中的检测阶段

作者：Zhumengying (zmy)  
日期：2025-12-17

🚀 启动命令：
    conda activate base   # 任意基础环境即可
    uvicorn api_yolo:app --host 0.0.0.0 --port 8000

🧪 测试命令：
    curl -X POST http://localhost:8000/yolo \
      -H "Content-Type: application/json" \
      -d '{
        "input_path": "/data/zmy/workspace/YOLO/yolov12-main/zhuzhu/data/scan_7",
        "output_dir": "/data/zmy/yolo_out/"
      }'
"""
# zhumengying 2025/12/17
import subprocess
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="YOLO Detection API (via conda run)")

class YOLORequest(BaseModel):
    input_path: str   # 单张图像路径 或 整个目录路径
    output_dir: str   # 输出目录（YOLO 会在此生成结果）

@app.post("/yolo")
def run_yolo(request: YOLORequest):
    input_path = request.input_path
    output_dir = request.output_dir

    # 验证输入路径存在
    if not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail=f"Input path does not exist: {input_path}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 构建 conda run 命令（完全隔离在 yolo 环境中执行）
    cmd = [
        "conda", "run", "-n", "api_yolo",          # 激活 yolo 环境
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

    try:
        # 执行命令，指定 GPU 3（物理 GPU 3）
        result = subprocess.run(
            cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "3"},
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120  # 防止卡死（根据你的数据量调整）
        )
        return {
            "status": "success",
            "output_dir": output_dir,
            "message": f"Processed {input_path}"
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="YOLO inference timeout")
    except subprocess.CalledProcessError as e:
        # 捕获 YOLO 命令本身的错误
        raise HTTPException(status_code=500, detail=f"YOLO failed: {e.stderr}")
    except Exception as e:
        # 其他异常（如路径权限等）
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import subprocess
# import os

# app = FastAPI(title="YOLO Detection API")

# class YOLORequest(BaseModel):
#     input_path: str   # 单张图像路径 或 整个目录路径
#     output_dir: str   # 输出目录（YOLO 会在此生成 images/ 和 labels/）

# @app.post("/yolo")
# def run_yolo(request: YOLORequest):
#     input_path = request.input_path
#     output_dir = request.output_dir

#     # ✅ 验证输入路径存在
#     if not os.path.exists(input_path):
#         raise HTTPException(400, f"Input path does not exist: {input_path}")

#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)

#     try:
#         # 调用 YOLO（直接传 input_path，支持文件/文件夹）
#         subprocess.run([
#             "conda", "run", "-n", "yolo",
#             "yolo", "predict",
#             "model=/data/zmy/runs/detect/train9/weights/best.pt",
#             f"source={input_path}",          # ← 支持文件 or 文件夹！
#             "save=True",
#             "save_txt=True",
#             "conf=0.3",
#             "save_conf=True",
#             "agnostic_nms=True",
#             f"project={output_dir}",
#             "name=.",      # 直接输出到 output_dir
#             "exist_ok=True"
#         ], env={**os.environ, "CUDA_VISIBLE_DEVICES": "3"}, check=True)

#         return {
#             "status": "success",
#             "output_dir": output_dir
#         }

#     except subprocess.CalledProcessError as e:
#         raise HTTPException(500, f"YOLO subprocess failed: {e.stderr}")
#     except Exception as e:
#         raise HTTPException(500, f"YOLO error: {str(e)}")

# uvicorn api_yolo:app --host 0.0.0.0 --port 8000   
# curl -X POST http://localhost:8000/yolo   -H "Content-Type: application/json"   -d '{ 
#     "input_path": "/data/zmy/workspace/YOLO/yolov12-main/zhuzhu/data/scan_7",
#     "output_dir": "/data/zmy/yolo_out/"
#   }'