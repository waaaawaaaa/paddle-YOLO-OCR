# =============================================================================
# 批量生成统一标签文件脚本（一对多复制）
# 作者：Zhumengying
# 日期：2025/12/02
#
# 功能说明：
#   本脚本将一个**固定的源标签文件**（如 yanneizhuyao_0001.txt）的内容，
#   复制生成为与目标图像文件夹中每张图像同名的 .txt 标签文件。
#   适用于所有图像具有相同目标类别的场景（例如：每张图都包含同一个固定物品）。
#
# 输入要求：
#   - image_folder：包含图像文件（.jpg/.jpeg/.png）的目录；
#   - label_folder：用于存放生成的标签文件（自动创建）；
#   - source_txt：一个有效的 YOLO 格式标签文件（5列：class_id cx cy w h）。
#
# 输出行为：
#   - 对 image_folder 中每个图像文件 xxx.jpg，
#     在 label_folder 中创建 xxx.txt，内容完全复制自 source_txt；
#   - 文件按图像文件名排序处理，确保可复现；
#   - 若 label_folder 不存在，自动创建。
#
# 适用场景：
#   - 快速构建“单类别单目标”数据集（如每张图都含一个“确认勾选”框）；
#   - 为模板化文档（如固定位置填写数量）批量生成初始标签；
#   - 预标注阶段的简易标签初始化。
#
# 注意事项：
#   - 所有图像将共享**完全相同的标签内容**（包括坐标和类别），
#     仅适用于目标位置/类别在所有图像中一致的情况；
#   - 若图像实际内容不同（如目标位置变化），此方法会产生错误标签；
#   - 请确保 source_txt 路径正确且为有效 YOLO 格式。
# =============================================================================
import os
from pathlib import Path

# === 配置 ===
image_folder = r"D:\Internship\OCR\data_own1\yanneizhuyao"                # 图像所在文件夹
label_folder = r"D:\Internship\OCR\data_own1\yanneizhuyao_label"          # 标签输出文件夹
source_txt = os.path.join(label_folder, "yanneizhuyao_0001.txt")          # 源标签文件

# 确保标签输出文件夹存在
os.makedirs(label_folder, exist_ok=True)

# 检查源标签是否存在
if not os.path.isfile(source_txt):
    raise FileNotFoundError(f"源标签文件不存在: {source_txt}")

# 获取所有图像文件（支持 .jpg / .jpeg / .png）
image_extensions = {'.jpg', '.jpeg', '.png'}
image_files = [
    f for f in os.listdir(image_folder)
    if Path(f).suffix.lower() in image_extensions
]

if not image_files:
    raise FileNotFoundError(f"在 {image_folder} 中未找到图像文件")

# 读取源标签内容
with open(source_txt, 'r', encoding='utf-8') as f:
    content = f.read()

# 为每个图像在 label_folder 中生成对应的 .txt 文件
for img_name in sorted(image_files):
    stem = Path(img_name).stem  # 如 "neiyan1_0001"
    target_txt = os.path.join(label_folder, f"{stem}.txt")  # 输出到 label_folder

    with open(target_txt, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ 已生成: {target_txt}")

print(f"🎉 共为 {len(image_files)} 张图像生成了标签文件（位于 {label_folder}）")