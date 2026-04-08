# -*- coding: utf-8 -*-
# =============================================================================
# YOLO 目标检测性能评估脚本（严格 IoU + 类别匹配）
# 作者：Zhumin Ying (zhumengying)
# 日期：2025/12/12
#
# 功能说明：
#   本脚本用于评估 YOLO 格式的目标检测模型性能，严格按照标准检测评估协议：
#   - 仅当预测框与真实框 **类别一致** 且 **IoU ≥ 阈值** 时，才计为 True Positive (TP)；
#   - 支持多类别（含数字、小数、组合标签如 "1+1"）和任意分辨率图像；
#   - 自动读取图像尺寸，将归一化的 YOLO 坐标 (cx, cy, w, h) 转换为像素坐标进行 IoU 计算；
#   - 统计每类及整体的 TP/FP/FN，并计算 Precision、Recall、F1 和近似 mAP。
#
# 输入要求：
#   - images/       ：原始图像目录（.jpg / .png 等），用于获取图像宽高；
#   - labels_real/  ：真实标签（YOLO 格式，5 列：class_id cx cy w h）；
#   - v1_real_labels/：预测结果（YOLO 格式，6 列：class_id cx cy w h conf）；
#   - 所有文件按相同 stem（不含扩展名）对齐（如 image001.jpg ↔ image001.txt）。
#
# 输出指标：
#   - 每类：TP, FP, FN, Precision, Recall, F1；
#   - 整体：总 TP/FP/FN、Precision、Recall、F1；
#   - 近似 mAP：对所有出现过样本的有效类别的 F1 取平均（便于快速评估）；
#   - 漏检数（FN）、误检数（FP）、正确检出数（TP）等实用统计。
#
# 评估参数可调：
#   - IOU_THRESHOLD：默认 0.5，可按任务需求调整（如 0.75）；
#   - CONFIDENCE_THRESH：过滤低置信度预测，避免噪声干扰。
#
# 适用场景：
#   - 医疗物资清单识别（手写数字/标签检测）；
#   - 结构化文档中关键字段的定位评估；
#   - YOLO 模型在真实测试集上的消融实验或版本对比。
#
# 注意事项：
#   - 类别 ID 必须在 [0, NUM_CLASSES) 范围内，否则会被忽略；
#   - 若某类别在 GT 和预测中均未出现，将自动跳过不输出；
#   - 本脚本不计算严格 mAP（需 PR 曲线与多阈值），但 F1 平均可作为高效代理指标。
# =============================================================================
# """
# 目标检测评估脚本（YOLO 格式）
# 作者：zhumengying
# 日期：2025/12/12
#
# 功能：
# - 计算 YOLO 检测结果的性能指标
# - 支持多类别、多分辨率图像
# - 严格遵循目标检测评估标准（同类匹配 + IoU 阈值）
#
# 输入格式：
# - 真实标签（labels/）：class_id cx cy w h                （5 列）
# - 预测结果（predictions/）：class_id cx cy w h conf       （6 列）
# - 图像目录（images/）：.jpg / .png 等
#
# 输出指标：
# - 每类：TP, FP, FN, Precision, Recall, F1
# - 整体：Precision, Recall, F1, 近似 mAP
# - 漏检数、误检数、正确检出数
# """

import os
from pathlib import Path
from PIL import Image
import numpy as np

# =============================================================================
# 配置区 —— 请根据你的实际路径修改
# =============================================================================
CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '0.05', '0.15', '0.1', '0.2', '0.3', '0.4', '0.5',
    '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9',
    '1+1', '2+1', '3+1', '2-1', '3-1'
]
NUM_CLASSES = len(CLASS_NAMES)

# 路径配置（请替换为你的实际路径）
IMAGE_DIR = Path(r"D:\Internship\OCR\dataset\real_test\test")  # 🖼️ 原始图像目录
GROUND_TRUTH_DIR = Path(r"D:\Internship\OCR\dataset\real_test\labels_real")  # 🏷️ 真实标签（5列）
PREDICTIONS_DIR = Path(r"D:\Internship\OCR\dataset\real_test\v1_real_labels")  # 🔮 预测结果（6列）

# 评估参数
IOU_THRESHOLD = 0.5  # IoU 阈值（≥ 才算匹配）
CONFIDENCE_THRESH = 0.01  # 置信度过滤阈值（低于此值的预测将被忽略）


# =============================================================================
# 工具函数
# =============================================================================
def yolo_to_bbox(cx, cy, w, h, img_w, img_h):
    """
    将 YOLO 归一化坐标 (cx, cy, w, h) 转换为像素坐标 [x1, y1, x2, y2]
    """
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def compute_iou(box1, box2):
    """
    计算两个框的 IoU (Intersection over Union)
    box 格式: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # 计算交集
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # 计算并集
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = area1 + area2 - inter_area

    # 避免除零
    return inter_area / union_area if union_area > 0 else 0.0


def safe_div(a, b):
    """安全除法，避免除零"""
    return a / b if b > 0 else 0.0


# =============================================================================
# 主评估函数
# =============================================================================
def evaluate_detection_metrics(image_dir, gt_dir, pred_dir, iou_thresh=0.5):
    """
    评估 YOLO 检测性能

    评估规则（严格遵循标准）：
    1. 每个预测框只能与 **相同类别** 的真实框匹配
    2. 匹配条件：IoU >= iou_thresh
    3. 一个真实框只能被匹配一次（贪心匹配）

    返回：
        dict: 包含整体和每类统计信息
    """
    # 初始化每类统计：tp, fp, fn, gts（真实目标数）
    class_stats = {
        cls_id: {"tp": 0, "fp": 0, "fn": 0, "gts": 0}
        for cls_id in range(NUM_CLASSES)
    }

    # 支持的图像格式
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # 构建图像文件映射：{stem: Path}
    image_files = {
        p.stem: p for p in image_dir.iterdir()
        if p.suffix.lower() in img_extensions
    }

    if not image_files:
        raise FileNotFoundError(f"在 {image_dir} 中未找到图像文件！")

    print(f"🔍 开始评估 {len(image_files)} 张图像...")

    # 遍历每张图像
    for stem, img_path in image_files.items():
        gt_file = gt_dir / f"{stem}.txt"
        pred_file = pred_dir / f"{stem}.txt"

        # 自动读取图像尺寸
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        # === 1. 加载真实标签（Ground Truth）===
        gt_boxes = []  # [(box, class_id)]
        if gt_file.exists():
            with open(gt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = list(map(float, line.split()))
                    if len(parts) != 5:
                        continue  # 跳过无效行
                    cls_id, cx, cy, w, h = parts
                    cls_id = int(cls_id)
                    # 忽略超出类别范围的标签
                    if 0 <= cls_id < NUM_CLASSES:
                        box = yolo_to_bbox(cx, cy, w, h, img_w, img_h)
                        gt_boxes.append((box, cls_id))
                        class_stats[cls_id]["gts"] += 1  # 累计真实目标数

        # === 2. 加载预测结果 ===
        pred_boxes = []  # [(box, class_id)]
        if pred_file.exists():
            with open(pred_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = list(map(float, line.split()))
                    if len(parts) != 6:
                        continue  # 跳过无效行
                    cls_id, cx, cy, w, h, conf = parts
                    cls_id = int(cls_id)
                    # 过滤低置信度或无效类别
                    if conf < CONFIDENCE_THRESH or not (0 <= cls_id < NUM_CLASSES):
                        continue
                    box = yolo_to_bbox(cx, cy, w, h, img_w, img_h)
                    pred_boxes.append((box, cls_id))

        # === 3. 匹配预测框与真实框（按类别独立匹配）===
        gt_matched = [False] * len(gt_boxes)  # 标记 GT 是否已被匹配

        # 遍历每个预测框
        for pred_box, pred_cls in pred_boxes:
            best_iou = -1
            best_gt_idx = -1

            # 只与 **相同类别** 的 GT 比较
            for i, (gt_box, gt_cls) in enumerate(gt_boxes):
                if gt_cls != pred_cls:  # 类别不同 → 跳过
                    continue
                if gt_matched[i]:  # 已被匹配 → 跳过
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            # 判断是否匹配成功
            if best_iou >= iou_thresh and best_gt_idx != -1:
                class_stats[pred_cls]["tp"] += 1  # True Positive
                gt_matched[best_gt_idx] = True  # 标记 GT 已匹配
            else:
                class_stats[pred_cls]["fp"] += 1  # False Positive（误检）

        # === 4. 统计漏检（未被匹配的 GT）===
        for i, (gt_box, gt_cls) in enumerate(gt_boxes):
            if not gt_matched[i]:
                class_stats[gt_cls]["fn"] += 1  # False Negative（漏检）

    # === 5. 汇总整体统计 ===
    total_tp = sum(s["tp"] for s in class_stats.values())
    total_fp = sum(s["fp"] for s in class_stats.values())
    total_fn = sum(s["fn"] for s in class_stats.values())
    total_gt = sum(s["gts"] for s in class_stats.values())

    # 计算整体指标
    overall_precision = safe_div(total_tp, total_tp + total_fp)
    overall_recall = safe_div(total_tp, total_tp + total_fn)
    overall_f1 = safe_div(2 * overall_precision * overall_recall, overall_precision + overall_recall)

    # === 6. 打印详细结果 ===
    print("\n" + "=" * 80)
    print(f"📊 目标检测评估结果 (IoU ≥ {iou_thresh:.1f})")
    print("=" * 80)
    print("注：TP 需同时满足 → 类别正确 + IoU ≥ 阈值")
    print("-" * 80)
    print(f"{'类别':<12} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>9} {'Recall':>9} {'F1':>9}")
    print("-" * 80)

    # 统计有效类别（出现过 GT 或预测的类别）
    valid_classes = 0
    sum_f1 = 0.0

    for cls_id in range(NUM_CLASSES):
        s = class_stats[cls_id]
        # 跳过完全未出现的类别
        if s["gts"] == 0 and s["tp"] + s["fp"] == 0:
            continue
        valid_classes += 1

        p = safe_div(s["tp"], s["tp"] + s["fp"])
        r = safe_div(s["tp"], s["tp"] + s["fn"])
        f1 = safe_div(2 * p * r, p + r)
        sum_f1 += f1

        cls_name = CLASS_NAMES[cls_id]
        print(f"{cls_name:<12} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} {p:>9.2%} {r:>9.2%} {f1:>9.2%}")

    # 计算近似 mAP（用 F1 平均，便于理解；严格 mAP 需 PR 曲线）
    mean_f1 = safe_div(sum_f1, valid_classes)

    print("-" * 80)
    print(
        f"{'总计':<12} {total_tp:>4} {total_fp:>4} {total_fn:>4} {overall_precision:>9.2%} {overall_recall:>9.2%} {overall_f1:>9.2%}")
    print(f"\n🎯 近似 mAP@{iou_thresh:.1f} = {mean_f1:.2%}")

    print(f"\n📌 详细统计：")
    print(f"   - 真实目标总数：{total_gt}")
    print(f"   - 正确检出（TP）：{total_tp}")
    print(f"   - 误检（FP）    ：{total_fp}")
    print(f"   - 漏检（FN）    ：{total_fn}")
    print(f"   - 精确率（Precision）：{overall_precision:.2%}")
    print(f"   - 召回率（Recall）  ：{overall_recall:.2%}")
    print(f"   - F1-score        ：{overall_f1:.2%}")

    return {
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "approx_mAP": mean_f1,
        "class_stats": class_stats
    }


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == "__main__":
    # 运行评估
    results = evaluate_detection_metrics(
        image_dir=IMAGE_DIR,
        gt_dir=GROUND_TRUTH_DIR,
        pred_dir=PREDICTIONS_DIR,
        iou_thresh=IOU_THRESHOLD
    )