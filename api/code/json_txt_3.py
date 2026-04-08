"""
医疗表格结构化融合脚本（YOLO 右边界聚类驱动 · 开发版）
======================================================

🎯 目标：
  结合 YOLO 手写数量检测与 PaddleOCR 文本识别结果，生成：
    - 制表符分隔的 TXT 文件（格式：字段1\t字段2\t...\t数量）
    - 带标注框的可视化图像（_vis.jpg）
  通过 **YOLO 检测框右边界聚类**自动划分垂直列区域，实现跨栏对齐。

🧠 核心逻辑：
  1. **列分割**：对所有 YOLO 框的右边界（x_right）进行 K-Means 聚类（K=1~3）
  2. **行匹配**：基于 y 坐标容差（图像高度的 Y_TOLERANCE_PCT%）判断是否同行
  3. **字段提取**：在每列内收集非重复 OCR 文本，拼接为 TSV 行
  4. **可视化**：合并 OCR 文本框 + YOLO 框，红框 + 白字标注数量值

⚠️ 当前局限性（作者自述）：
  - **依赖每栏均有 YOLO 检测**：若数据仅出现在部分栏位，聚类可能失效
  - **列数固定为 2~3**：不适用于动态列数或复杂嵌套表格
  - **无表头语义理解**：字段顺序依赖空间位置，无法识别“编码”“名称”等语义

⚙️ 配置说明：
  - `Y_TOLERANCE_PCT=0.7`：行高容差（% of 图像高度）
  - `DUPLICATE_DIST_PCT=1.5`：OCR 与 YOLO 重复判定距离
  - `MIN_OCR_FIELDS=2`：每行最少 OCR 字段数（防噪声）

🛠️ 使用方式：
  - **单图调试**：设置 `DEBUG_SINGLE=True` + 指定 `SINGLE_IMAGE_PATH`
  - **批量处理**：`DEBUG_SINGLE=False`，自动处理 `test/` 目录下所有图像

📂 目录结构假设（硬编码）：
  base_dir/
  ├── test/               # ← 输入图像
  ├── v2_real_labels/     # ← YOLO 生成的 .txt
  ├── paddle_test/        # ← PaddleOCR 生成的 _res.json
  └── final_out_XXX/      # ← 输出 TXT + _vis.jpg

📌 适用场景：
  - 表格结构简单、列数固定（2~3 列）
  - 每列均有手写数量（YOLO 可检出）
  - 作为快速原型验证或特定数据集处理

作者：Zhumengying
日期：2025-12-12  
状态：开发中（需优化鲁棒性）
"""
# # 是想结合YOLO和paddle拍出来的结果，给出一个TXT文件和一个可视化结果
# 通过百分比计算相对位置，来判断在不在同一行
# 目前还是基于yolo结果，通过Kmeans聚类，这就要求我每一栏都得有数据，
# 要是数据只在一栏，可能会error
# 2025/12/12   zhumengying
import json
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path


# -------------------------------
# 配置（百分比形式）
# -------------------------------
CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '0.05', '0.15', '0.1', '0.2', '0.3', '0.4', '0.5',
    '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9',
    '1+1', '2+1', '3+1', '2-1', '3-1'
]

# 百分比参数（占图像高度的比例）
Y_TOLERANCE_PCT = 0.7      # 行高容差（如 2% 的图像高度）
DUPLICATE_DIST_PCT = 1.5   # OCR 与 YOLO 重复判定距离（如 2.5% 的图像高度）
MIN_OCR_FIELDS = 2         # 最少非数量字段数（保持为整数）

# -------------------------------
# 调试开关：True 为单图调试，False 为批量处理
# -------------------------------
DEBUG_SINGLE = False  # ← 改这里！True 调试单张，False 批处理
SINGLE_IMAGE_PATH = r"E:\Desktop\predict10\test\mmexport1760952157661_res.jpg"  # ← 改这里！

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
            'box': box,  # [x1, y1, x2, y2]
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
        all_x = []
        all_y = []

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

        big_x1 = min(all_x)
        big_y1 = min(all_y)
        big_x2 = max(all_x)
        big_y2 = max(all_y)

        draw.rectangle([big_x1, big_y1, big_x2, big_y2], outline="red", width=max(2, int(img_h * 0.003)))
        label = yolo['class_name']
        text_y = big_y1 # - int(img_h * 0.035)  # 稍微上移一点，避免重叠

        # 获取文本尺寸
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_width = right - left
        text_height = bottom - top

        # 计算包围盒的中心 x
        box_center_x = (big_x1 + big_x2) / 2

        # 文字居中：起始 x = 中心 - 文本宽/2
        text_x = box_center_x - text_width / 2

        # 重新计算 text_bbox（用于背景框）
        text_bbox = (text_x, text_y, text_x + text_width, text_y + text_height)

        draw.rectangle(text_bbox, fill="red")
        draw.text((text_x, text_y), label, fill="white", font=font)

    img.save(output_vis_path)


def process_single_image(img_path, base_dir, output_dir):
    img_stem = img_path.stem
    yolo_file = base_dir / "v2_real_labels" / f"{img_stem}.txt"
    paddle_file = base_dir / "paddle_test" / f"{img_stem}_res.json"
    output_txt = output_dir / f"{img_stem}.txt"

    try:
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        y_tolerance_px = img_h * (Y_TOLERANCE_PCT / 100.0)
        duplicate_dist_px = img_h * (DUPLICATE_DIST_PCT / 100.0)

        yolo_detections = load_yolo_detections(yolo_file, img_w, img_h, CLASS_NAMES)
        if not yolo_detections:
            print(f"⚠️  {img_stem}: 无 YOLO 检测结果")
            return

        all_ocr_items = load_paddle_results(paddle_file)
        if not all_ocr_items:
            print(f"⚠️  {img_stem}: 无 PaddleOCR 结果")
            return

        col_ranges, centers = cluster_columns_by_right_edge(yolo_detections, img_w)
        print(f"📊 {img_stem}: 右边界聚类中心: {[round(c, 1) for c in centers]}")
        print(f"🧩 大栏划分: {[f'[{round(r[0],1)}, {round(r[1],1)}]' for r in col_ranges]}")

        final_results = match_yolo_with_ocr(
            yolo_detections, all_ocr_items, col_ranges, img_w, img_h,
            y_tolerance=y_tolerance_px,
            duplicate_dist=duplicate_dist_px
        )
        if not final_results:
            print(f"⚠️  {img_stem}: 未匹配到任何结果")
            return

        with open(output_txt, 'w', encoding='utf-8') as f:
            for _, line, _, _ in final_results:
                f.write(line + "\n")
        print(f"✅ 文本已保存: {output_txt}")

        output_vis = str(output_txt).replace(".txt", "_vis.jpg")
        draw_visualization(img_path, final_results, output_vis)
        print(f"🖼️  可视化已保存: {output_vis}")

    except Exception as e:
        print(f"❌ {img_path.name}: 处理失败 - {e}")
        import traceback
        traceback.print_exc()


# -------------------------------
# 单图调试主函数
# -------------------------------
def main_single():
    img_path = Path(SINGLE_IMAGE_PATH)
    if not img_path.exists():
        raise FileNotFoundError(f"单图路径不存在: {img_path}")

    base_dir = img_path.parent.parent  # 假设结构: base/test/image.jpg → base 是 predict10
    output_dir = base_dir / "final_out_V2_S7"
    output_dir.mkdir(exist_ok=True)

    print(f"🔍 调试单图: {img_path}")
    process_single_image(img_path, base_dir, output_dir)


# -------------------------------
# 批量处理主函数
# -------------------------------
def main_batch():
    base_dir = Path(r"D:\Internship\OCR\dataset\real_test")
    test_img_dir = base_dir / "test"
    output_dir = base_dir / "final_out_test"
    output_dir.mkdir(exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        p for p in test_img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    if not image_files:
        print("❌ test 文件夹中未找到图像文件")
        return

    print(f"📁 找到 {len(image_files)} 张图像，开始批量处理...\n")

    for img_path in sorted(image_files):
        print(f"\n--- 处理: {img_path.name} ---")
        process_single_image(img_path, base_dir, output_dir)

    print(f"\n🎉 批量处理完成！结果已保存至: {output_dir}")


# -------------------------------
# 入口
# -------------------------------
if __name__ == "__main__":
    if DEBUG_SINGLE:
        main_single()
    else:
        main_batch()