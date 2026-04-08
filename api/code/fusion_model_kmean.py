"""
医疗表格结构化融合引擎（YOLO 右边界聚类驱动）
==============================================

🎯 目标：
  针对单数量列或简单多列的扫描医疗表格，通过 **YOLO 检测框的右边界聚类**自动划分逻辑列，
  将手写数量（YOLO）与左侧 OCR 文本按空间位置对齐，输出结构化文本 + 可视化结果。

🧠 核心逻辑：
  1. **列分割**：
      - 提取所有 YOLO 检测框的右边界（x_right）
      - 使用 K-Means 聚类（K=1~3）划分为 2~3 个垂直区域（如：[文本区][数量区]）
  2. **行匹配**：
      - 对每个 YOLO 框，归属到其 x 坐标所在的列区间
      - 在同一列内，收集 y 坐标相近（±0.7% 图像高度）且非重复的 OCR 文本
  3. **结果生成**：
      - 每行输出格式：`<OCR字段1>\t<OCR字段2>\t...\t<YOLO数量>`
      - 自动跳过字段数不足的行（MIN_OCR_FIELDS=2）
  4. **可视化**：
      - 合并 OCR 文本框 + YOLO 框为大包围框
      - 红底白字居中显示 YOLO 识别值

📥 输入要求：
  - `image_dir`：原始图像（.jpg/.png）
  - `yolo_label_dir`：YOLO 生成的 `{stem}.txt`（归一化坐标）
  - `ocr_json_dir`：PaddleOCR 输出的 `{stem}_res.json`，含：
        { "rec_texts": [...], "rec_boxes": [[x1,y1,x2,y2], ...] }

📤 输出：
  - `{stem}.txt`：每行一个制表符分隔的结构化记录
  - `{stem}_vis.jpg`：可视化图，红色框 + YOLO 值标签

⚙️ 配置说明：
  - `CLASS_NAMES`：YOLO 类别映射（需与训练一致）
  - `Y_TOLERANCE_PCT=0.7`：行高容差（% of 图像高度）
  - `DUPLICATE_DIST_PCT=1.5`：OCR 与 YOLO 重复判定距离
  - `MIN_OCR_FIELDS=2`：每行最少 OCR 字段数（防噪声）

📦 支持批量处理，适用于离线分析或 API 后处理阶段  
作者：Zhumengying
日期：2025-12-17
"""

import json
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

# -------------------------------
# 配置（保持不变）
# -------------------------------
CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '0.05', '0.15', '0.1', '0.2', '0.3', '0.4', '0.5',
    '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9',
    '1+1', '2+1', '3+1', '2-1', '3-1'
]

Y_TOLERANCE_PCT = 0.7      # 行高容差（% of image height）
DUPLICATE_DIST_PCT = 1.5   # OCR 与 YOLO 重复判定距离（% of image height）
MIN_OCR_FIELDS = 2         # 最少 OCR 字段数

# -------------------------------
# 工具函数（保持不变）
# -------------------------------
def load_yolo_detections(yolo_file, img_w, img_h, class_names):
    detections = []
    if not yolo_file.exists():
        return detections
    with open(yolo_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx_norm, cy_norm, w_norm, h_norm = parts[1], parts[2], parts[3], parts[4]
                cx_px = cx_norm * img_w
                cy_px = cy_norm * img_h
                x_right = (cx_norm + w_norm / 2) * img_w
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"unknown_{cls_id}"
                detections.append({
                    'class_name': cls_name,
                    'x': cx_px,
                    'y': cy_px,
                    'x_right': x_right,
                    'w_norm': w_norm,
                    'h_norm': h_norm
                })
    return detections

def load_paddle_results(paddle_file):
    ocr_items = []
    if not paddle_file.exists():
        return ocr_items
    with open(paddle_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for text, box in zip(data['rec_texts'], data['rec_boxes']):
        text = text.strip()
        if not text:
            continue
        ocr_items.append({
            'text': text,
            'box': box,
            'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        })
    return ocr_items

def cluster_columns_by_right_edge(yolo_detections, img_w):
    if not yolo_detections:
        return [], []
    x_coords = np.array([d['x_right'] for d in yolo_detections]).reshape(-1, 1)
    K = min(3, len(yolo_detections))
    if K == 1:
        centers = [np.mean(x_coords)]
    elif K == 2:
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(x_coords)
        centers = np.sort(kmeans.cluster_centers_.flatten())
    else:
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(x_coords)
        centers = np.sort(kmeans.cluster_centers_.flatten())
    if K == 1:
        r1 = centers[0]
        col_ranges = [(0, r1), (r1, img_w)]
    elif K == 2:
        r1, r2 = centers[0], centers[1]
        col_ranges = [(0, r1), (r1, r2), (r2, img_w)]
    else:
        r1, r2 = centers[0], centers[1]
        col_ranges = [(0, r1), (r1, r2), (r2, img_w)]
    return col_ranges, centers

def is_ocr_duplicate(ocr, yolo_centers, dist_threshold):
    o_x, o_y = ocr['center']
    for yx, yy in yolo_centers:
        if ((o_x - yx) ** 2 + (o_y - yy) ** 2) ** 0.5 < dist_threshold:
            return True
    return False

def match_yolo_with_ocr(yolo_detections, all_ocr_items, col_ranges, img_w, img_h,
                        y_tolerance, duplicate_dist):
    yolo_centers = [(d['x'], d['y']) for d in yolo_detections]
    final_results = []
    for yolo in yolo_detections:
        q_x, q_y, q_text = yolo['x'], yolo['y'], yolo['class_name']
        target_range = None
        for x_min, x_max in col_ranges:
            if x_min <= q_x <= x_max:
                target_range = (x_min, x_max)
                break
        if target_range is None:
            continue
        candidates = []
        for ocr in all_ocr_items:
            o_x, o_y = ocr['center']
            if not (abs(o_y - q_y) <= y_tolerance and target_range[0] <= o_x <= target_range[1]):
                continue
            if is_ocr_duplicate(ocr, yolo_centers, duplicate_dist):
                continue
            candidates.append(ocr)
        if len(candidates) < MIN_OCR_FIELDS:
            continue
        candidates.sort(key=lambda o: o['center'][0])
        paddle_texts = [ocr['text'] for ocr in candidates]
        line = "\t".join(paddle_texts) + "\t" + q_text
        paddle_boxes = [ocr['box'] for ocr in candidates]
        final_results.append((q_y, line, yolo, paddle_boxes))
    final_results.sort(key=lambda x: x[0])
    return final_results

def draw_visualization(img_path, final_results, output_vis_path):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    font = None
    fonts_to_try = [
        "simhei.ttf", "Microsoft-YaHei.ttf", "msyh.ttc",
        "PingFang.ttc", "DejaVuSans.ttf", "arial.ttf"
    ]
    font_size = max(20, int(img_h * 0.03))
    for font_name in fonts_to_try:
        try:
            font = ImageFont.truetype(font_name, size=font_size)
            break
        except:
            continue
    if font is None:
        font = ImageFont.load_default()
    for _, _, yolo, paddle_boxes in final_results:
        all_x, all_y = [], []
        for box in paddle_boxes:
            x1, y1, x2, y2 = box
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])
        cx_norm = yolo['x'] / img_w
        cy_norm = yolo['y'] / img_h
        w_norm = yolo['w_norm']
        h_norm = yolo['h_norm']
        yolo_x1 = (cx_norm - w_norm / 2) * img_w
        yolo_y1 = (cy_norm - h_norm / 2) * img_h
        yolo_x2 = (cx_norm + w_norm / 2) * img_w
        yolo_y2 = (cy_norm + h_norm / 2) * img_h
        all_x.extend([yolo_x1, yolo_x2])
        all_y.extend([yolo_y1, yolo_y2])
        big_x1, big_y1 = min(all_x), min(all_y)
        big_x2, big_y2 = max(all_x), max(all_y)
        draw.rectangle([big_x1, big_y1, big_x2, big_y2], outline="red", width=max(2, int(img_h * 0.003)))
        label = yolo['class_name']
        text_y = big_y1
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_width = right - left
        text_height = bottom - top
        box_center_x = (big_x1 + big_x2) / 2
        text_x = box_center_x - text_width / 2
        text_bbox = (text_x, text_y, text_x + text_width, text_y + text_height)
        draw.rectangle(text_bbox, fill="red")
        draw.text((text_x, text_y), label, fill="white", font=font)
    img.save(output_vis_path)

# -------------------------------
# 改造后的单图处理函数（显式传入路径）
# -------------------------------
def process_single_image_v2(
    img_path: Path,
    yolo_label_dir: Path,
    ocr_json_dir: Path,
    output_dir: Path
):
    img_stem = img_path.stem
    yolo_file = yolo_label_dir / f"{img_stem}.txt"
    paddle_file = ocr_json_dir / f"{img_stem}_res.json"
    output_txt = output_dir / f"{img_stem}.txt"

    try:
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        y_tolerance_px = img_h * (Y_TOLERANCE_PCT / 100.0)
        duplicate_dist_px = img_h * (DUPLICATE_DIST_PCT / 100.0)

        yolo_detections = load_yolo_detections(yolo_file, img_w, img_h, CLASS_NAMES)
        if not yolo_detections:
            print(f"⚠️  {img_stem}: 无 YOLO 检测结果")
            return False

        all_ocr_items = load_paddle_results(paddle_file)
        if not all_ocr_items:
            print(f"⚠️  {img_stem}: 无 PaddleOCR 结果")
            return False

        col_ranges, centers = cluster_columns_by_right_edge(yolo_detections, img_w)
        final_results = match_yolo_with_ocr(
            yolo_detections, all_ocr_items, col_ranges, img_w, img_h,
            y_tolerance=y_tolerance_px,
            duplicate_dist=duplicate_dist_px
        )
        if not final_results:
            print(f"⚠️  {img_stem}: 未匹配到任何结果")
            return False

        with open(output_txt, 'w', encoding='utf-8') as f:
            for _, line, _, _ in final_results:
                f.write(line + "\n")

        output_vis = output_dir / f"{img_stem}_vis.jpg"
        draw_visualization(img_path, final_results, str(output_vis))
        return True

    except Exception as e:
        print(f"❌ {img_path.name}: 处理失败 - {e}")
        import traceback
        traceback.print_exc()
        return False

# -------------------------------
# 主入口函数：供外部调用
# -------------------------------
def run_fusion_pipeline(
    image_dir: str,
    yolo_label_dir: str,
    ocr_json_dir: str,
    output_dir: str
):
    """
    融合 YOLO 检测结果与 PaddleOCR 识别结果。
    
    Args:
        image_dir: 图像文件夹路径（.jpg, .png 等）
        yolo_label_dir: YOLO 生成的 labels 文件夹（含 .txt）
        ocr_json_dir: PaddleOCR 生成的 JSON 文件夹（含 *_res.json）
        output_dir: 融合结果输出目录（生成 .txt 和 _vis.jpg）
    """
    image_dir = Path(image_dir)
    yolo_label_dir = Path(yolo_label_dir)
    ocr_json_dir = Path(ocr_json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    if not image_files:
        print("❌ 图像文件夹中未找到有效图像")
        return

    print(f"📁 找到 {len(image_files)} 张图像，开始融合处理...")
    success_count = 0
    for img_path in sorted(image_files):
        print(f"\n--- 处理: {img_path.name} ---")
        if process_single_image_v2(img_path, yolo_label_dir, ocr_json_dir, output_dir):
            success_count += 1

    print(f"\n✅ 融合完成！成功处理 {success_count}/{len(image_files)} 张图像，结果保存至: {output_dir}")