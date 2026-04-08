# ======================================================================
# Fusion Pipeline: YOLO + PaddleOCR Result Integration
# Author: Zhumenying
# Date: 2025-12-24
#
# Purpose:
#   Fuse detection results from YOLO (quantity values) and PaddleOCR (text fields)
#   to reconstruct structured medical inventory records from scanned tables.
#
# Key Logic:
#   1. Locate the "quantity" column header via keywords (e.g., "数量", "剂量").
#   2. Segment the table into logical columns based on header positions.
#   3. For each YOLO-detected quantity:
#        - Identify its row and column
#        - Match with OCR text in the same spatial region
#        - Extract the first field as `issue_id`, subsequent as `issue_name`
#   4. Output structured JSON with bounding box aligned to OCR + YOLO.
#
# Input Format:
#   - `yolo_detections`: List of normalized detections from YOLO service
#   - `ocr_results`: Raw PaddleOCR output (in-memory, not JSON file)
#
# Output Format:
#   List of {
#       "issue_id": str,
#       "issue_name": str,
#       "issue_number": str,  # from YOLO
#       "issue_box": [x1, y1, x2, y2]
#   }
# ======================================================================

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
# 工具函数（新增：解析 OCR 内存结果）
# -------------------------------
def parse_ocr_results(ocr_results):
    """
    将 PaddleOCR 返回的内存结果转为 ocr_items 格式
    """
    items = []
    for text, box in zip(ocr_results['rec_texts'], ocr_results['rec_boxes']):
        text = text.strip()
        if text:
            items.append({
                'text': text,
                'box': box,
                'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            })
    return items


# -------------------------------
# 表头解析（不变）
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
# 匹配逻辑（不变）
# -------------------------------
def is_duplicate(ocr, yolo_centers, dist):
    ox, oy = ocr['center']
    return any(((ox - x)**2 + (oy - y)**2)**0.5 < dist for x, y in yolo_centers)

def match_yolo_with_ocr(yolo_dets, ocr_items, major_columns, y_tol, dup_dist):
    if not major_columns or not yolo_dets:
        return []

    qty_x_list = [mc['qty_x'] for mc in major_columns]
    results = []
    yolo_centers = [(d['x'], d['y']) for d in yolo_dets]

    for yolo in yolo_dets:
        q_x, q_y, q_text = yolo['x'], yolo['y'], yolo['val']
        yolo_x2, yolo_y2 = yolo['bbox'][2], yolo['bbox'][3]

        left_boundary = min((abs(q_x - qx), mc['major_range'][0]) for qx, mc in zip(qty_x_list, major_columns))[1]

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
# 可视化函数（居中显示 YOLO 值）
# -------------------------------
def draw_visualization(img_path, results, output_vis_path):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    w, h = img.size

    font = None
    fonts = ["simhei.ttf", "Microsoft-YaHei.ttf", "msyh.ttc", "DejaVuSans.ttf", "arial.ttf"]
    font_size = max(20, int(h * 0.03))
    for name in fonts:
        try:
            font = ImageFont.truetype(name, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    for row in results:
        bbox = row.get("issue_box")
        if not bbox:
            continue
        x1, y1, x2, y2 = map(float, bbox)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=max(2, int(h * 0.003)))

        yolo_val = ""
        for key, val in row.items():
            if key == "issue_number":
                yolo_val = str(val)
                break
        if not yolo_val:
            continue

        try:
            left, top, right, bottom = draw.textbbox((0, 0), yolo_val, font=font)
            tw, th = right - left, bottom - top
        except Exception:
            tw, th = draw.textsize(yolo_val, font=font)

        tx = (x1 + x2 - tw) / 2
        ty = (y1 + y2 - th) / 2

        draw.text((tx, ty), yolo_val, fill="red", font=font)

    img.save(output_vis_path)

# -------------------------------
# 核心函数：改用内存输入
# -------------------------------
def run_fusion_pipeline(image_path, yolo_detections, ocr_results, output_dir=None):
    """
    内存直传版融合函数
    :param image_path: 图像路径（用于获取宽高）
    :param yolo_detections: list[dict]，YOLO 服务返回的 detections
    :param ocr_results: list，PaddleOCR 返回的原始结果
    :param output_dir: 保留参数（可忽略）
    :return: items: list[dict]
    """
    # 1. 获取图像尺寸
    with Image.open(image_path) as img:
        w, h = img.size

    # 2. 解析 OCR 结果
    ocr_items = parse_ocr_results(ocr_results)

    # 3. 构建 YOLO detections（从归一化坐标转像素坐标）
    yolo_dets = []
    for det in yolo_detections:
        cx = det["cx"] * w
        cy = det["cy"] * h
        bbox_w = det["width"] * w
        bbox_h = det["height"] * h
        x1 = cx - bbox_w / 2
        y1 = cy - bbox_h / 2
        x2 = cx + bbox_w / 2
        y2 = cy + bbox_h / 2
        val = CLASS_NAMES[det["class_id"]] if det["class_id"] < len(CLASS_NAMES) else str(det["class_id"])
        yolo_dets.append({
            'val': val,
            'x': cx,
            'y': cy,
            'bbox': [x1, y1, x2, y2]
        })
    yolo_dets.sort(key=lambda d: d['y'])

    # 4. 表头解析 + 匹配
    if not yolo_dets:
        return []
    if not ocr_items:
        return []

    major_columns, y_tol = parse_table_header(ocr_items, w, h)
    if not major_columns:
        return []

    dup_dist = h * 0.015
    items = match_yolo_with_ocr(yolo_dets, ocr_items, major_columns, y_tol, dup_dist)
    draw_visualization(image_path, items, '/data/zmy/workspace/api/code/2api_0file/test.jpg')
    return items