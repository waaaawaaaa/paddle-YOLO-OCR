"""
医疗表格多栏结构化融合引擎（OCR 表头驱动）
==========================================

🎯 目标：
  针对含多个“数量”列的扫描医疗表格（如：[编码][名称][数量] [编码][名称][数量] ...），
  基于 OCR 识别的表头自动划分逻辑大栏，将 YOLO 检测的手写数量与对应栏位的 OCR 文本精准对齐，
  输出结构化 JSON + 可视化结果。

🧠 核心逻辑：
  1. **表头解析**：
      - 在图像上部（y ≤ 30%）搜索“数量”“剂量”等关键词
      - 按 x 坐标排序，划分多个“大栏”（如 [0, qty_x1], [qty_x1, qty_x2], ...）
      - 每个大栏包含若干表头字段（如“编码”“项目”）
  2. **行匹配**：
      - 对每个 YOLO 手写数量框，根据 x 坐标归属到最近的大栏
      - 在该栏内，收集同一行（y 容差）的 OCR 文本
  3. **字段对齐**：
      - 将 OCR 文本按 x 位置与表头字段一一匹配
      - 未匹配的 OCR 自动合并到中间字段（防漏）
      - YOLO 值强制覆盖所有“数量”字段
  4. **坐标融合**：
      - 输出 bbox 包含所有相关文本 + YOLO 框，用于可视化与定位

📥 输入要求：
  - `image_dir`：原始图像（.jpg/.png）
  - `yolo_dir`：YOLO 生成的 .txt 标签（归一化坐标）
  - `ocr_dir`：PaddleOCR 输出的 `{stem}_res.json`，含：
        { "rec_texts": [...], "rec_boxes": [[x1,y1,x2,y2], ...] }

📤 输出：
  - `{stem}.json`：每行为一个 dict，字段名来自 OCR 表头，值来自 OCR 或 YOLO
  - `{stem}_vis.jpg`：可视化图，绿色框 + 红色居中显示 YOLO 值

⚙️ 配置说明：
  - `CLASS_NAMES`：YOLO 模型类别映射（需与训练一致）
  - `QUANTITY_KEYWORDS`：用于识别数量列的中文关键词
  - `CODE_PATTERN`：定义“项目编码”格式（纯字母数字）
  - 百分比参数（`Y_TOLERANCE_PCT`, `DUPLICATE_DIST_PCT`）自动适配图像尺寸

📦 支持批量处理，适用于离线分析或 API 后处理阶段  
作者：Zhumengying 
日期：2025-12-19
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from statistics import median, stdev
import re

CODE_PATTERN = re.compile(r'^[0-9A-Za-z]+$')  # 放在函数外或模块顶部

# -------------------------------
# 配置
# -------------------------------
CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '0.05', '0.15', '0.1', '0.2', '0.3', '0.4', '0.5',
    '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9',
    '1+1', '2+1', '3+1', '2-1', '3-1'
]

QUANTITY_KEYWORDS = ["数量", "剂量", "用量", "值", "数值", "计量", "支数", "盒数"]

# 百分比参数（占图像高度）
Y_TOLERANCE_PCT = 0.7      # 行匹配容忍度（%）
DUPLICATE_DIST_PCT = 1.5   # OCR 与 YOLO 重复判定距离（%）
MIN_HEADER_FIELDS = 2      # 表头最少字段数


# -------------------------------
# 工具函数
# -------------------------------
def load_yolo(yolo_file, w, h, names):
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
                val = names[cid] if cid < len(names) else f"unknown_{cid}"
                dets.append({'val': val, 'x': cx, 'y': cy, 'bbox': [x1, y1, x2, y2]})
            except (ValueError, IndexError):
                continue
    # ✅ 按 y 坐标从上到下排序（cy 越小越靠上）
    dets.sort(key=lambda det: det['y'])
    return dets



def load_ocr(ocr_file):
    items = []
    if not ocr_file.exists():
        return items
    with open(ocr_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for text, box in zip(data['rec_texts'], data['rec_boxes']):
        text = text.strip()
        if not text:
            continue
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
# 表头定位与分栏函数
# -------------------------------
def parse_table_header_with_major_columns(ocr_items, img_w, img_h, quantity_keywords, y_max_ratio=0.3, min_fields=2):
    """
    针对固定结构：[编码][项目][数量] 重复
    大栏范围 = [上一个数量中心, 当前数量中心]
    """
    # --- Step 1: 定位表头行 ---
    quantity_items = [
        item for item in ocr_items
        if item['center'][1] <= img_h * y_max_ratio
           and any(kw in item['text'] for kw in quantity_keywords)
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
    # quantity_items = [
    #     item for item in ocr_items
    #     if item['center'][1] <= img_h * y_max_ratio
    #     and any(kw in item['text'] for kw in quantity_keywords)
    # ]
    # if not quantity_items:
    #     return [], 0.0
    #
    # y_coords = [item['center'][1] for item in quantity_items]
    # if len(y_coords) > 1:
    #     neighbor_dist = img_h * 0.05
    #     filtered_y = [
    #         y for i, y in enumerate(y_coords)
    #         if any(abs(y - y_coords[j]) <= neighbor_dist for j in range(len(y_coords)) if i != j)
    #     ]
    #     if filtered_y:
    #         y_coords = filtered_y
    #
    # header_y = y_coords[0] if len(y_coords) == 1 else median(y_coords)
    # y_tol = img_h * 0.015 if len(y_coords) == 1 else max(img_h * 0.005, stdev(y_coords))
    #
    # header = [item for item in ocr_items if abs(item['center'][1] - header_y) <= y_tol]
    # header.sort(key=lambda x: x['center'][0])
    # if len(header) < min_fields:
    #     return [], 0.0

    # --- Step 2: 找出所有“数量”字段 ---
    qty_info = []
    for i, item in enumerate(header):
        if any(kw in item['text'] for kw in quantity_keywords):
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

def match_yolo_with_ocr(yolo_detections, all_ocr_items, major_columns, y_tolerance, duplicate_dist):
    yolo_centers = [(d['x'], d['y']) for d in yolo_detections]
    final_results = []

    qty_x_list = [mc['qty_x'] for mc in major_columns]
    if not qty_x_list:
        return final_results

    for yolo in yolo_detections:
        q_x, q_y, q_text = yolo['x'], yolo['y'], yolo['val']
        yolo_x2 = yolo['bbox'][2]

        # Step 1: 基于 qty_x 距离找大栏
        distances = [abs(q_x - qty_x) for qty_x in qty_x_list]
        min_idx = distances.index(min(distances))
        target_major = major_columns[min_idx]

        header_items = target_major['header_items']
        left_boundary = target_major['major_range'][0]

        # Step 2: 宽松收集 OCR
        row_ocr_candidates = []
        for ocr in all_ocr_items:
            o_x, o_y = ocr['center']
            if (o_x <= yolo_x2 and
                abs(o_y - q_y) <= y_tolerance and
                not is_duplicate(ocr, yolo_centers, duplicate_dist) and
                o_x >= left_boundary):  # 宽松左边界
                row_ocr_candidates.append(ocr)

        if not row_ocr_candidates:
            continue

        row_ocr_candidates.sort(key=lambda o: o['center'][0])

        # Step 2.1: 用正则找编码，精确定位左边界
        code_candidates = [
            ocr for ocr in row_ocr_candidates
            if ocr['text'].strip() and CODE_PATTERN.match(ocr['text'].strip())
        ]
        true_left = min((ocr['center'][0] for ocr in code_candidates), default=left_boundary)

        # Step 2.2: 用精确左边界过滤
        final_ocr_candidates = [
            ocr for ocr in row_ocr_candidates
            if ocr['center'][0] >= true_left-2
        ]

        if not final_ocr_candidates:
            continue

        # row_ocr_candidates.sort(key=lambda o: o['center'][0])
        # === Step 4: 基于位置对齐分配字段 ===
        row = {}
        all_x, all_y = [], []

        # 为每个表头字段找最近的 OCR
        for i, field_item in enumerate(header_items):
            field_x = field_item['center'][0]
            field_name = field_item['text']

            # 在 final_ocr_candidates 中找 x 最接近的 OCR
            best_ocr = None
            min_dist = float('inf')
            for ocr in final_ocr_candidates:
                dist = abs(ocr['center'][0] - field_x)
                if dist < min_dist:
                    min_dist = dist
                    best_ocr = ocr

            if best_ocr is not None:
                row[field_name] = best_ocr['text']
                x1, y1, x2, y2 = best_ocr['box']
                all_x.extend([x1, x2])
                all_y.extend([y1, y2])

                # 从候选中移除已分配的 OCR（防止重复分配）
                final_ocr_candidates.remove(best_ocr)
            else:
                row[field_name] = ""

        # === Step 5: 处理剩余 OCR（即未被字段匹配的多余 OCR）===
        extra_ocr = final_ocr_candidates  # 剩下的就是多余的
        S = len(header_items)
        if extra_ocr:
            if S <= 2:
                row["issue_name"] = " ".join(ocr['text'] for ocr in extra_ocr)
                for ocr in extra_ocr:
                    x1, y1, x2, y2 = ocr['box']
                    all_x.extend([x1, x2])
                    all_y.extend([y1, y2])
            else:
                # S > 2: 合并到最近的中间字段（索引 1 到 S-2）
                mid_indices = list(range(1, S - 1))
                mid_x_positions = [header_items[i]['center'][0] for i in mid_indices]

                for ocr in extra_ocr:
                    ocr_x = ocr['center'][0]
                    distances = [abs(ocr_x - mx) for mx in mid_x_positions]
                    nearest_idx_in_mid = distances.index(min(distances))
                    target_field_idx = mid_indices[nearest_idx_in_mid]
                    target_field = header_items[target_field_idx]['text']

                    row[target_field] = (row[target_field] + " " + ocr['text']).strip()

                    x1, y1, x2, y2 = ocr['box']
                    all_x.extend([x1, x2])
                    all_y.extend([y1, y2])

        # === Step 6: 用 YOLO 值覆盖“数量”字段 ===
        for item in header_items:
            if any(kw in item['text'] for kw in QUANTITY_KEYWORDS):
                row[item['text']] = q_text

        # === Step 7: 添加 YOLO bbox ===
        yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo['bbox']
        all_x.extend([yolo_x1, yolo_x2])
        all_y.extend([yolo_y1, yolo_y2])
        row["bbox"] = [
            int(min(all_x)),
            int(min(all_y)),
            int(max(all_x)),
            int(max(all_y))
        ]

        final_results.append(row)

    return final_results

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
        bbox = row.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = map(float, bbox)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=max(2, int(h * 0.003)))

        yolo_val = ""
        for key, val in row.items():
            if key != "bbox" and any(kw in key for kw in QUANTITY_KEYWORDS):
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
# 主处理函数
# -------------------------------
def process_image(img_path, yolo_dir, ocr_dir, out_dir):
    stem = img_path.stem
    yolo_file = yolo_dir / f"{stem}.txt"
    ocr_file = ocr_dir / f"{stem}_res.json"

    try:
        with Image.open(img_path) as img:
            w, h = img.size

        yolo_dets = load_yolo(yolo_file, w, h, CLASS_NAMES)
        if not yolo_dets:
            print(f"⚠️ {stem}: 无 YOLO 检测")
            return False

        ocr_items = load_ocr(ocr_file)
        if not ocr_items:
            print(f"⚠️ {stem}: 无 OCR 结果")
            return False

        major_columns, y_tol_match = parse_table_header_with_major_columns(
            ocr_items, w, h, QUANTITY_KEYWORDS, y_max_ratio=0.3, min_fields=2
        )

        # === 打印大栏信息 ===
        if not major_columns:
            print(f"⚠️ {stem}: 无法定位有效表头")
            return False

        num_major = len(major_columns)
        ranges_str = [f"[{round(r['major_range'][0], 1)}, {round(r['major_range'][1], 1)}]" for r in major_columns]
        print(f"📊 {stem}: 检测到 {num_major} 个大栏")
        print(f"🧩 大栏划分: {ranges_str}")
        # ===================
        dup_dist = h * (DUPLICATE_DIST_PCT / 100.0)

        results = match_yolo_with_ocr(
            yolo_dets,
            ocr_items,
            major_columns,
            y_tolerance=y_tol_match,  # 来自 parse 函数
            duplicate_dist=dup_dist
        )

        if not results:
            print(f"⚠️ {stem}: 未匹配到有效结果")
            return False

        # === 保存为 JSON 文件 ===
        output_json = out_dir / f"{stem}.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON 已保存: {output_json}")
        # =======================

        draw_visualization(img_path, results, out_dir / f"{stem}_vis.jpg")

        print(f"✅ {stem}: {len(results)} 行")
        return True

    except Exception as e:
        print(f"❌ {stem}: {e}")
        return False


# -------------------------------
# 批量运行
# -------------------------------
def run_fusion_pipeline(image_dir, yolo_dir, ocr_dir, output_dir):
    image_dir, yolo_dir, ocr_dir, output_dir = map(Path, (image_dir, yolo_dir, ocr_dir, output_dir))
    output_dir.mkdir(exist_ok=True)

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in exts]
    if not images:
        print("❌ 无图像文件")
        return

    print(f"Processing {len(images)} images...")
    success = sum(process_image(img, yolo_dir, ocr_dir, output_dir) for img in sorted(images))
    print(f"\n✅ 成功: {success}/{len(images)}")


# -------------------------------
# 主程序
# -------------------------------
# if __name__ == "__main__":
#     IMAGE_DIR = "D:/Internship/OCR/dataset/real_test/test"
#     YOLO_DIR = "D:/Internship/OCR/dataset/real_test/v2_real_labels"
#     OCR_DIR = "D:/Internship/OCR/dataset/real_test/paddle_test"
#     OUTPUT_DIR = "D:/Internship/OCR/dataset/real_test/fusion"

#     run_pipeline(IMAGE_DIR, YOLO_DIR, OCR_DIR, OUTPUT_DIR)