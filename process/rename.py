# =============================================================================
# 批量文件重命名脚本（按顺序编号）
# 作者：Zhumengying
# 日期：2025/12/19
#
# 功能说明：
#   本脚本将指定目录下的所有文件（不递归子目录）按文件名排序后，
#   重命名为连续数字序号（如 1.jpg, 2.png, 3.jpg...），保留原始扩展名。
#
# 输入要求：
#   - 指定一个包含待重命名文件的目录（如 D:\...\real）；
#   - 文件可为任意类型（.jpg, .png, .txt 等），但不处理子文件夹；
#   - 文件按字典序排序后编号，确保每次运行结果一致。
#
# 重命名规则：
#   - 新文件名格式：{序号}{原始扩展名}（如 1.jpg, 2.txt）；
#   - 序号从 1 开始，连续递增；
#   - 若目标文件名已存在，则跳过当前文件并打印警告（防止覆盖）。
#
# 适用场景：
#   - 整理杂乱命名的图像/标签文件，便于后续按序号处理；
#   - 为数据集预处理提供统一命名规范；
#   - 配合其他脚本（如标注工具、训练流程）简化文件索引。
#
# 注意事项：
#   - 本脚本 **不处理子目录中的文件**，仅操作一级目录；
#   - 重命名不可逆，请提前备份重要数据；
#   - 若目录中已有 1.jpg、2.jpg 等文件，可能导致跳过或命名错位，建议在干净目录中使用。
# =============================================================================
import os
import glob

# 设置目录路径
folder = r"D:\Internship\OCR\data_own2\real"

# 获取所有文件（不含子目录），按名称排序
files = [f for f in glob.glob(os.path.join(folder, "*")) if os.path.isfile(f)]
files.sort()  # 按文件名排序，确保顺序一致

# 开始重命名
for idx, filepath in enumerate(files, start=1):
    old_name = os.path.basename(filepath)
    ext = os.path.splitext(old_name)[1]  # 获取扩展名，如 .jpg
    new_name = f"{idx}{ext}"
    new_path = os.path.join(folder, new_name)

    # 防止覆盖（如果已有同名文件，跳过或报错）
    if os.path.exists(new_path):
        print(f"⚠️ 跳过 {old_name}：目标文件 {new_name} 已存在")
        continue

    os.rename(filepath, new_path)
    print(f"✅ 重命名: {old_name} → {new_name}")

print("✅ 所有文件重命名完成！")