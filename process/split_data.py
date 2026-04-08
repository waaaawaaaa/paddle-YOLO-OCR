# =============================================================================
# YOLO 格式数据集自动划分脚本（按类别目录结构组织）
# 作者：Zhumin Ying
# 日期：2025/12/29
#
# 功能说明：
#   本脚本将按类别组织的原始数据（图像 + 对应标签）自动划分为 train / val / test 三部分，
#   并输出标准 YOLO 数据集结构（images/ + labels/），便于直接用于训练或评估。
#
# 输入数据结构要求：
#   src_root/
#   ├── classA/                 # 图像文件夹（仅含 .jpg 等图像）
#   │   ├── img1.jpg
#   │   └── ...
#   ├── classA_label/           # 对应标签文件夹（.txt，YOLO 格式）
#   │   ├── img1.txt
#   │   └── ...
#   ├── classB/
#   ├── classB_label/
#   └── ...
#
# 输出数据结构（符合 YOLOv5/v8 等框架规范）：
#   base_output/
#   ├── train/
#   │   ├── images/             # 所有训练图像
#   │   └── labels/             # 对应训练标签
#   ├── val/
#   │   ├── images/
#   │   └── labels/
#   └── test/
#       ├── images/
#       └── labels/
#
# 划分策略：
#   - 按 **每个类别独立划分**，避免类别分布偏移；
#   - 默认比例：80% train / 10% val / 10% test；
#   - 使用固定随机种子（42），确保划分结果可复现；
#   - 仅保留图像与标签同时存在的样本对（自动跳过缺失项并报警告）。
#
# 适用场景：
#   - 医疗/物资清单手写数字检测数据集构建；
#   - 多类别目标检测任务的本地数据预处理；
#   - 为 YOLO 系列模型（如 YOLOv5, YOLOv8, PP-YOLOE）准备训练数据。
#
# 注意事项：
#   - 标签文件必须与图像同名（仅扩展名不同），且位于同名加 "_label" 的目录中；
#   - 脚本仅处理 .jpg 图像（可通过 rglob 扩展支持其他格式）；
#   - 输出路径会自动创建，若已存在则追加内容（建议清空后重跑）。
# =============================================================================
import os
import shutil
import random
from pathlib import Path

# ===== 配置区 =====
src_root = Path(r"D:\Internship\OCR\data_own2")
base_output = Path(r"D:\Internship\OCR\dataset_own\datasets1")

# 创建目标目录结构
for split in ['train', 'val', 'test']:
    (base_output / split / 'images').mkdir(parents=True, exist_ok=True)
    (base_output / split / 'labels').mkdir(parents=True, exist_ok=True)

# 找出所有图像类别文件夹（排除 -label 结尾的）
image_dirs = [d for d in src_root.iterdir() if d.is_dir() and not d.name.endswith('_label')]

total_copied = 0

# 固定随机种子以确保可复现
random.seed(42)

def copy_pairs_to_split(pairs, split_name, base_output):
    """辅助函数：将一对文件复制到指定 split"""
    img_dst_dir = base_output / split_name / 'images'
    lbl_dst_dir = base_output / split_name / 'labels'
    count = 0
    for img_p, txt_p in pairs:
        shutil.copy2(img_p, img_dst_dir / img_p.name)
        shutil.copy2(txt_p, lbl_dst_dir / txt_p.name)
        count += 1
    return count

for img_dir in image_dirs:
    label_dir = src_root / (img_dir.name + "_label")
    if not label_dir.exists():
        print(f"⚠️ 警告：找不到 label 文件夹 {label_dir}，跳过 {img_dir.name}")
        continue

    # 收集当前类别下所有有效的 (img, txt) 对
    valid_pairs = []
    for img_path in sorted(img_dir.rglob("*.jpg")):
        rel_path = img_path.relative_to(img_dir)
        txt_path = label_dir / rel_path.with_suffix(".txt")
        if txt_path.exists():
            valid_pairs.append((img_path, txt_path))
        else:
            print(f"⚠️ 缺少标签: {txt_path}")

    if not valid_pairs:
        print(f"❌ {img_dir.name}：无有效样本，跳过")
        continue

    # 打乱当前类别的样本
    random.shuffle(valid_pairs)

    n = len(valid_pairs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_pairs = valid_pairs[:n_train]
    val_pairs = valid_pairs[n_train:n_train + n_val]
    test_pairs = valid_pairs[n_train + n_val:]

    # 复制并累加数量
    total_copied += copy_pairs_to_split(train_pairs, 'train', base_output)
    total_copied += copy_pairs_to_split(val_pairs, 'val', base_output)
    total_copied += copy_pairs_to_split(test_pairs, 'test', base_output)

    print(f"✅ 类别 {img_dir.name}：共 {n} 个样本 → "
          f"train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

print(f"\n🎉 总共处理 {total_copied} 个文件（图像+标签）")
print(f"📁 数据集已保存至: {base_output}")