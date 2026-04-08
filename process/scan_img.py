# =============================================================================
# 交互式文档图像扫描增强脚本（透视矫正 + Sauvola 二值化）
# 作者：Zhumengying
# 日期：2025/12/11
#
# 功能说明：
#   本脚本通过用户交互点击图像中文档的四个角点，自动完成：
#   1. 透视变换（Perspective Correction）→ 将倾斜文档校正为正面视图；
#   2. 图像增强（CLAHE）→ 改善光照不均；
#   3. 自适应二值化（Sauvola 算法）→ 生成高对比度黑白扫描效果，保留细小文字；
#   4. 输出标准扫描版图像（.jpg），适用于 OCR 前处理。
#
# 交互流程：
#   - 程序自动缩放大图以适配屏幕（默认最大 1200×800）；
#   - 用户按顺时针或逆时针顺序点击文档的四个角点（左上 → 右上 → 右下 → 左下）；
#   - 点击后显示绿色圆点反馈，第四点点击后自动关闭窗口并处理；
#   - 按 ESC 可随时取消操作。
#
# 技术亮点：
#   - 使用 Sauvola 局部阈值算法，对阴影、光照变化、背景纹理鲁棒性强；
#   - 自动计算输出图像最佳尺寸（基于矫正后四边形）；
#   - 支持任意分辨率输入，缩放显示不影响原始坐标精度；
#   - 输出为纯黑白（0/255）高对比图像，显著提升 OCR 识别率。
#
# 适用场景：
#   - 手机拍摄的文档、表格、医疗清单等图像的数字化预处理；
#   - OCR 系统（如 PaddleOCR）前的图像标准化；
#   - 需要保留手写体或小字号印刷体的场景（Sauvola 优于全局阈值）。
#
# 依赖库：
#   - OpenCV (cv2)：图像读写、透视变换、CLAHE；
#   - scikit-image：Sauvola 自适应阈值；
#   - NumPy：数值计算。
#
# 注意事项：
#   - 请确保四点按角点顺序点击（无需严格顺序，但需对应四角）；
#   - 建议输出图像尺寸 ≤ 2048px（避免后续 OCR 延迟过高）；
#   - Sauvola 的 window_size 可根据图像清晰度调整（默认 25，细节多可减小）。
# =============================================================================

import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from pathlib import Path

# 全局变量
points = []
image_display = None
image_original = None
scale = 1.0

def click_and_crop(event, x, y, flags, param):
    global points, image_display, scale
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        orig_x = int(round(x / scale))
        orig_y = int(round(y / scale))
        points.append((orig_x, orig_y))

        disp_x = int(orig_x * scale)
        disp_y = int(orig_y * scale)
        cv2.circle(image_display, (disp_x, disp_y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 corners (click on document corners)", image_display)
        print(f"第 {len(points)} 个点: 原始坐标 ({orig_x}, {orig_y})")

        if len(points) == 4:
            cv2.destroyAllWindows()

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def scan_effect_with_perspective(image_path, output_path, max_display_width=1200, max_display_height=800):
    global points, image_display, image_original, scale
    points = []

    # 1. 读取原始图像
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    image_original = img.copy()

    # 2. 缩放图像用于显示
    h, w = img.shape[:2]
    scale_w = max_display_width / w
    scale_h = max_display_height / h
    scale = min(scale_w, scale_h, 1.0)

    new_w = int(w * scale)
    new_h = int(h * scale)
    image_display = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3. 创建窗口并设置回调
    window_name = "Select 4 corners (click on document corners)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, new_w, new_h)
    cv2.setMouseCallback(window_name, click_and_crop)

    print(f"原始尺寸: {w}x{h} | 显示尺寸: {new_w}x{new_h} (缩放比例: {scale:.2%})")
    print("请依次点击四个角点。按 ESC 可取消。")

    while len(points) < 4:
        cv2.imshow(window_name, image_display)
        key = cv2.waitKey(50)
        if key == 27:
            cv2.destroyAllWindows()
            print("操作已取消。")
            return

    cv2.destroyAllWindows()

    # 4. 透视矫正
    pts = np.array(points, dtype="float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_original, M, (maxWidth, maxHeight))

    # 5. 图像增强
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # CLAHE 增强（解决光照不均）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
    enhanced = clahe.apply(gray)

    # 6. 使用 Sauvola 算法进行二值化（推荐！）
    # Sauvola 对文档图像效果极佳，能保留细小文字，去除阴影
    window_size = 25  # 根据图像分辨率调整，越大越平滑，但可能丢失细节
    thresh_sauvola = threshold_sauvola(enhanced, window_size=window_size)
    binary = (enhanced > thresh_sauvola).astype(np.uint8) * 255

    # 7. 保存结果
    cv2.imwrite(str(output_path), binary)
    print(f"✅ 已保存扫描版: {output_path} (尺寸: {binary.shape[1]}x{binary.shape[0]})")

    # 8. 预览结果
    res_h, res_w = binary.shape
    res_scale = min(1000 / res_w, 700 / res_h, 1.0)
    preview = cv2.resize(binary, (int(res_w * res_scale), int(res_h * res_scale)))
    cv2.imshow("Scanned Result (press any key to close)", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===== 使用 =====
input_img = Path("D:/Internship/OCR/data_own2/real/mmexport1760952142264_res.jpg")
output_img = input_img.parent / f"{input_img.stem}_scan_corrected_sauvola.jpg"
scan_effect_with_perspective(input_img, output_img, max_display_width=1200, max_display_height=800)