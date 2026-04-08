"""
医疗表格结构化融合引擎：YOLO 手写数量 + PaddleOCR 文本字段对齐
==================================================================

🎯 目标：
  从扫描的医疗物资清单表格图像中，自动提取结构化记录：
    - 项目编码（如 "A102"）
    - 项目名称（如 "一次性注射器"）
    - 手写数量（如 "2+1", "0.5"）
  并输出带坐标对齐的 JSON 结果，用于后续入库或核验。

🔍 核心逻辑：
  1. **表头定位**：通过关键词（"数量", "剂量" 等）识别数量列位置，划分逻辑列区。
  2. **行对齐**：利用 YOLO 检测框的垂直中心（y）与 OCR 文本行对齐。
  3. **字段提取**：
        - 在每行中，从 YOLO 检测框左侧开始，向左搜索最近的“编码格式”文本（纯数字/字母）作为 `issue_id`
        - 其后连续文本拼接为 `issue_name`
        - YOLO 识别值作为 `issue_number`
  4. **去重过滤**：避免 OCR 文本与 YOLO 框因空间重叠被重复匹配。
  5. **坐标融合**：输出框左上来自 OCR 编码，右下来自 YOLO 框，实现视觉对齐。

📂 输入要求（由上游 API 保证）：
  - `image_dir`：包含单张图像的目录或图像文件路径（支持 .jpg/.png 等）
  - `yolo_dir` / `ocr_dir`：分别包含同名 `.txt`（YOLO label）和 `_res.json`（OCR 结果）的目录
      示例：
        image:  /tmp/xxx/uploaded_img.jpg
        YOLO:   /tmp/xxx/yolo/labels/uploaded_img.txt
        OCR:    /tmp/xxx/ocr/uploaded_img_res.json

📤 输出格式：
  List of dict:
    {
      "issue_id":     str,   # 项目编码（如 "K203"）
      "issue_name":   str,   # 项目名称（如 "医用纱布"）
      "issue_number": str,   # 手写数量（来自 YOLO，如 "1.5", "2+1"）
      "issue_box":    [x1, y1, x2, y2]  # 融合坐标（左上=OCR编码, 右下=YOLO框）
    }

⚙️ 配置说明：
  - `CLASS_NAMES`：YOLO 模型输出类别映射（需与训练一致）
  - `QUANTITY_KEYWORDS`：用于定位数量列的中文关键词集合
  - `CODE_PATTERN`：正则，定义“有效编码”格式（仅字母数字）

📌 注意：
  - 本模块设计为 **单图处理**，由 FastAPI 主接口调用
  - 不负责图像预处理、模型推理，仅做结果融合
  - 若表头未识别、无 YOLO 检测或无 OCR 文本，返回空列表

作者：Zhumengying (zmy)  
日期：2025-12-23  
项目：医疗物资智能识别系统
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from statistics import median, stdev
import re

# -------------------------------
# 配置（保持最小）
# -------------------------------
CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '0.05', '0.15', '0.1', '0.2', '0.3', '0.4', '0.5',
    '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9',
    '1+1', '2+1', '3+1', '2-1', '3-1'
]

QUANTITY_KEYWORDS = {"数量", "剂量", "用量", "值", "数值", "计量", "支数", "盒数"}
CODE_PATTERN = re.compile(r'^[0-9A-Za-z]+$')

# -------------------------------
# 工具函数
# -------------------------------
def load_yolo(yolo_file, w, h):
    dets = []
    if not yolo_file.exists():
        return dets
    with open(yolo_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cid, cx_n, cy_n, bw_n, bh_n = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                cx, cy = cx_n * w, cy_n * h
                x1, y1 = cx - bw_n * w / 2, cy - bh_n * h / 2
                x2, y2 = cx + bw_n * w / 2, cy + bh_n * h / 2
                val = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
                dets.append({'val': val, 'x': cx, 'y': cy, 'bbox': [x1, y1, x2, y2]})
            except (ValueError, IndexError):
                continue
    dets.sort(key=lambda d: d['y'])
    return dets

def load_ocr(ocr_file):
    items = []
    if not ocr_file.exists():
        return items
    with open(ocr_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for text, box in zip(data['rec_texts'], data['rec_boxes']):
        text = text.strip()
        if text:
            items.append({
                'text': text,
                'box': box,
                'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            })
    return items

def is_duplicate(ocr, yolo_centers, dist):
    ox, oy = ocr['center']
    return any(((ox - x)**2 + (oy - y)**2)**0.5 < dist for x, y in yolo_centers)

# -------------------------------
# 表头解析（简化）
# -------------------------------
def parse_table_header(ocr_items, img_w, img_h, y_max_ratio=0.3, min_fields=2):
    """
    针对固定结构：[编码][项目][数量] 重复
    大栏范围 = [上一个数量中心, 当前数量中心]
    """
    # --- Step 1: 定位表头行 ---
    quantity_items = [
        item for item in ocr_items
        if item['center'][1] <= img_h * y_max_ratio
           and any(kw in item['text'] for kw in QUANTITY_KEYWORDS)
    ]
    if not quantity_items:
        return [], 0.0

    # 先按 x 坐标排序，确保从左到右顺序（关键！）
    quantity_items_sorted = sorted(quantity_items, key=lambda x: x['center'][0])
    y_coords = [item['center'][1] for item in quantity_items_sorted]

    # 邻近去噪：保留至少有一个邻居在 neighbor_dist 内的点
    if len(y_coords) > 1:
        neighbor_dist = img_h * 0.05
        filtered_y = [
            y for i, y in enumerate(y_coords)
            if any(abs(y - y_coords[j]) <= neighbor_dist for j in range(len(y_coords)) if i != j)
        ]
        if filtered_y:
            y_coords = filtered_y

    # 重新按 x 排序后的 y_coords（若 filtered，需重新对应，但这里只用 y 值，不影响 median 和 tol）
    if len(y_coords) == 1:
        header_y = y_coords[0]
        y_tol = img_h * 0.015
    else:
        header_y = median(y_coords)
        # 计算相邻 y 坐标的最大绝对差值（反映局部倾斜）
        adjacent_abs_diffs = [
            abs(y_coords[i] - y_coords[i - 1])
            for i in range(1, len(y_coords))
        ]
        max_adjacent_diff = max(adjacent_abs_diffs)
        y_tol = max(img_h * 0.007, max_adjacent_diff)  # 1.2 为安全裕量，可调

    # 提取 header 行（在 header_y ± y_tol 范围内的所有 OCR 项）
    header = [item for item in ocr_items if abs(item['center'][1] - header_y) <= y_tol]
    header.sort(key=lambda x: x['center'][0])

    if len(header) < min_fields:
        return [], 0.0

    # --- Step 2: 找出所有“数量”字段 ---
    qty_info = []
    for i, item in enumerate(header):
        if any(kw in item['text'] for kw in QUANTITY_KEYWORDS):
            qty_info.append({'index': i, 'x': item['center'][0]})

    if not qty_info:
        return [], 0.0

    qty_x_list = [info['x'] for info in qty_info]
    num_groups = len(qty_x_list)

    # --- Step 3: 构建大栏范围 ---
    major_columns = []

    for i in range(num_groups):
        qty_x = qty_x_list[i]

        # 左边界 = 上一个数量 x（第一个为 0）
        left_x = 0.0 if i == 0 else qty_x_list[i - 1]
        # 右边界 = 当前数量 x
        right_x = qty_x

        major_range = (left_x, right_x)

        # 提取字段：从上一个数量+1 到当前数量
        start_idx = 0 if i == 0 else qty_info[i-1]['index'] + 1
        end_idx = qty_info[i]['index']
        sub_header = header[start_idx:end_idx + 1]

        if sub_header:
            major_columns.append({
                'major_range': major_range,
                'header_items': sub_header,
                'qty_x': qty_x
            })

    return major_columns, y_tol

# -------------------------------
# 匹配逻辑（你已优化）
# -------------------------------
def match_yolo_with_ocr(yolo_dets, ocr_items, major_columns, y_tol, dup_dist):
    if not major_columns or not yolo_dets:
        return []

    qty_x_list = [mc['qty_x'] for mc in major_columns]
    results = []

    yolo_centers = [(d['x'], d['y']) for d in yolo_dets]

    for yolo in yolo_dets:
        q_x, q_y, q_text = yolo['x'], yolo['y'], yolo['val']
        yolo_x2, yolo_y2 = yolo['bbox'][2], yolo['bbox'][3]

        # 找最近的 qty_x 对应的左边界
        left_boundary = min((abs(q_x - qx), mc['major_range'][0]) for qx, mc in zip(qty_x_list, major_columns))[1]

        # 收集候选 OCR
        candidates = [
            ocr for ocr in ocr_items
            if ocr['center'][0] >= left_boundary - 2
            and ocr['center'][0] <= yolo_x2
            and abs(ocr['center'][1] - q_y) <= y_tol
            and not is_duplicate(ocr, yolo_centers, dup_dist)
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda o: o['center'][0])

        # 精确左边界（编码）
        code_xs = [ocr['center'][0] for ocr in candidates if CODE_PATTERN.match(ocr['text'].strip())]
        true_left = min(code_xs, default=left_boundary)
        final_candidates = [ocr for ocr in candidates if ocr['center'][0] >= true_left - 2]
        if not final_candidates:
            continue

        first = final_candidates[0]
        results.append({
            "issue_id": first['text'],
            "issue_name": " ".join(ocr['text'] for ocr in final_candidates[1:]),
            "issue_number": q_text,
            "issue_box": [int(first['box'][0]), int(first['box'][1]), int(yolo_x2), int(yolo_y2)]
        })
    return results

# -------------------------------
# 核心函数：处理单图并返回结果
# -------------------------------
def run_fusion_pipeline(image_dir, yolo_dir, ocr_dir, output_dir=None):
    """
    处理单张图像（或目录中唯一图像），返回 fusion_result dict。
    为 FastAPI 设计：不处理批量，不 print，只返回结构化结果。
    """
    image_dir, yolo_dir, ocr_dir = map(Path, (image_dir, yolo_dir, ocr_dir))
    
    # 支持单图或目录（取第一张）
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in exts] if image_dir.is_dir() else [image_dir]
    if not images:
        raise ValueError("No valid image found")

    img_path = images[0]
    stem = img_path.stem
    yolo_file = yolo_dir / f"{stem}.txt"
    ocr_file = ocr_dir / f"{stem}_res.json"

    # 读图尺寸
    with Image.open(img_path) as img:
        w, h = img.size

    # 加载 YOLO 和 OCR
    yolo_dets = load_yolo(yolo_file, w, h)
    ocr_items = load_ocr(ocr_file)

    if not yolo_dets:
        return {"unique_id": stem, "items": [], "message": "no YOLO detections"}
    if not ocr_items:
        return {"unique_id": stem, "items": [], " message": "no OCR results"}

    # 解析表头
    major_columns, y_tol = parse_table_header(ocr_items, w, h)
    if not major_columns:
        return {"unique_id": stem, "items": [], "message": "header not found"}

    # 匹配
    dup_dist = h * 0.015
    items = match_yolo_with_ocr(yolo_dets, ocr_items, major_columns, y_tol, dup_dist)
    return items