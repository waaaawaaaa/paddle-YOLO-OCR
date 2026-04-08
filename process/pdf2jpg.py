# =============================================================================
# PDF 转 JPEG 图像批量处理脚本（单文件模式）
# 作者：Zhumengying
# 日期：2025/12/01
#
# 功能说明：
#   本脚本将指定的单个 PDF 文件逐页转换为高质量 JPEG 图像，并按以下规则组织输出：
#   - 在指定的输出根目录下，以 PDF 文件名（不含扩展名）创建子文件夹；
#   - 每页图像保存为 "{pdf_stem}_{页码:04d}.jpg" 格式（如 mazui1_0001.jpg）；
#   - 图像质量设为 95，兼顾清晰度与文件大小；
#   - 支持中文路径与长文件名，使用 pathlib 和 os.path 确保跨平台兼容性。
#
# 适用场景：
#   OCR 前处理、文档图像数据集构建、医疗/物资清单 PDF 数字化等。
#
# 依赖库：
#   - pdf2image（需安装并配置 Poppler）
#   - Pillow（由 pdf2image 自动依赖）
#
# 使用方式：
#   1. 修改 `pdf_path` 为待处理的 PDF 路径；
#   2. 修改 `output_parent_folder` 为期望的输出根目录；
#   3. 运行脚本，结果将自动按子文件夹归类。
#
# 注意：
#   - 若需处理多个 PDF 并合并到同一文件夹，请参考下方注释掉的多文件模式；
#   - 建议 PDF 单页分辨率不超过 2048px，以控制后续 OCR 处理延迟。
# =============================================================================

import os
from pathlib import Path
from pdf2image import convert_from_path

# === 配置 ===
pdf_path = r"D:\Internship\OCR\data_pdf\mazui1.pdf"          # 你的 PDF 文件路径
output_parent_folder = r"D:\Internship\OCR\data_own1"    # 你指定的输出根目录

# 提取 PDF 文件名（不含扩展名）作为子文件夹名和前缀
pdf_stem = Path(pdf_path).stem  # 例如 "neiyan1_pdf"

# 在 output_parent_folder 下创建子文件夹：data_own1/neiyan1_pdf/
output_subfolder = os.path.join(output_parent_folder, pdf_stem)
os.makedirs(output_subfolder, exist_ok=True)

print(f"📁 输出子文件夹: {output_subfolder}")

# === 转换 PDF 为图像 ===
images = convert_from_path(pdf_path)

# 保存每一页为 JPG，格式：neiyan1_pdf_001.jpg, neiyan1_pdf_002.jpg, ...
for i, image in enumerate(images, start=1):
    filename = f"{pdf_stem}_{i:04d}.jpg"
    save_path = os.path.join(output_subfolder, filename)
    image.save(save_path, "JPEG", quality=95)
    print(f"✅ 已保存: {filename}")

print(f"🎉 共转换 {len(images)} 页，全部存入 '{output_subfolder}'")


# import os
# from pathlib import Path
# from pdf2image import convert_from_path
#
# # === 配置 ===
# pdf_paths = [
#     r"D:\Internship\OCR\data_pdf\yanneizhuyao1.pdf",
#     r"D:\Internship\OCR\data_pdf\yanneizhuyao2.pdf"
# ]
# output_folder = r"D:\Internship\OCR\data_own1\yanneizhuyao"  # 所有图放在一个文件夹
# prefix = "yanneizhuyao"  # 统一前缀
#
# # 创建输出文件夹
# os.makedirs(output_folder, exist_ok=True)
#
# # 全局计数器
# global_count = 1
#
# for pdf_path in pdf_paths:
#     if not os.path.isfile(pdf_path):
#         print(f"⚠️ 跳过不存在的文件: {pdf_path}")
#         continue
#
#     print(f"正在处理: {os.path.basename(pdf_path)}")
#     try:
#         images = convert_from_path(pdf_path)
#     except Exception as e:
#         print(f"❌ 转换失败 {pdf_path}: {e}")
#         continue
#
#     for image in images:
#         filename = f"{prefix}_{global_count:04d}.jpg"
#         save_path = os.path.join(output_folder, filename)
#         image.save(save_path, "JPEG", quality=95)
#         print(f"✅ 保存: {filename}")
#         global_count += 1
#
# print(f"🎉 共转换 {global_count - 1} 页，全部存入 {output_folder}")