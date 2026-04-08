# 📄 手术单识别 API 文档

> **服务名称**：Unified YOLO + PaddleOCR Fusion API  
> **接口路径**：`POST /unified/upload`  
> **作者**：Zhumy
> **最后更新**：2025-12-30

---

## 一、接口概述

该接口用于上传一张**手术单图像**，系统会自动：
1. 调用 **PaddleOCR** 识别表格中的文字（如项目编码、名称）
2. 调用 ** YOLO 服务** 检测数量位置的手写数字（如 "2+1"、"0.5"）
3. 在内存中**融合两者结果**，返回结构化的 JSON 数据

✅ **特点**：
- 无中间文件（纯内存处理）
- 端到端响应时间约 **2~3 秒**
- 支持并发请求（每请求独享临时目录）
- 基于目前手术单表格设计

---

## 二、接口参数说明

请求使用 `multipart/form-data` 格式，包含以下两个字段：

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `unique_id` | string | ✅ 是 | 请求的唯一标识符。 |
| `image` | file | ✅ 是 | 待识别的图像文件。<br>**支持格式**：`.jpg`, `.jpeg`, `.png`<br>**建议**：尺寸过大可能增加延迟 |

> 💡 注意：两个参数都必须提供，缺少任一参数将返回 400 错误。

---

## 三、接口调用方法

### 调用方式
- **HTTP 方法**：`POST`
- **URL**：`http://localhost:8000/unified/upload`
- **Content-Type**：`multipart/form-data`

### 命令行示例（curl）
```bash
curl -X POST "http://localhost:8000/unified/upload" \
  -F "unique_id=zmy" \
  -F "image=@/data/scan/1.jpg"
```

## 四、请求与响应示例

### ✅ 成功请求（HTTP 200）

**请求参数**：
- `unique_id = "zmy"`
- `image = [二进制图像数据]`

**成功响应（JSON）**：
<!-- ```json
{
  "unique_id": "zmy",
  "items": [
    {
      "issue_id": "400021",
      "issue_name": "一次性手术包",
      "issue_number": "2+1",
      "issue_box": [120, 340, 480, 390]
    },
    {
      "issue_id": "80932",
      "issue_name": "妥布霉素眼膏",
      "issue_number": "0.5",
      "issue_box": [125, 410, 475, 460]
    }
  ],
  "costtime": 2.843,
  "message": "success"
}
``` -->

### 📌 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `unique_id` | string | 与请求一致，用于结果匹配和日志追踪 |
| `items` | list | 识别出的结构化记录列表（可能为空） |
| `costtime` | float | 整个请求处理耗时（单位：秒，保留 3 位小数） |
| `message` | string | 状态信息：<br>• `"success"`：有有效结果<br>• `"no matches"`：未匹配到任何记录 |

#### `items` 中每个对象的字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `issue_id` | string | 项目编码（通常是数字，如 `"80932"`） |
| `issue_name` | string | 项目名称（如 `"妥布霉素眼膏"`） |
| `issue_number` | string | YOLO 识别出的手写数量（如 `"2+1"`, `"0.5"`） |
| `issue_box` | `[int, int, int, int]` | 融合后的坐标框 `[x1, y1, x2, y2]`（像素单位，左上角 → 右下角） |

## 五、错误码说明

| HTTP 状态码 | 错误响应示例 | 含义 | 可能原因 | 解决方法 |
|-------------|---------------|------|--------|--------|
| `400` | `{"detail": "Image not found on server: ..."}` | 客户端错误 | 图像上传失败、临时目录不可写 | 检查文件是否有效，确认服务对 `/api/tmp` 有写权限 |
| `500` | `{"unique_id": "xxx", "error": "YOLO service error: Connection failed", "costtime": 1.2}` | 服务端错误 | YOLO 服务未启动、OCR 崩溃、融合逻辑异常、GPU 显存不足等 | 1. 确保 YOLO 服务已运行在 `http://localhost:8001/predict`<br>2. 检查 GPU 资源（PaddleOCR 用 GPU 2，YOLO 用 GPU 3）<br>3. 查看服务日志定位具体错误 |

> ⚠️ 所有 `5xx` 错误响应都会包含 `unique_id` 和 `costtime` 字段，便于快速定位问题请求。
> 

---

## ⚙️ 服务依赖

### 前提条件
- **YOLO 服务必须运行在**：  
  `http://localhost:8001/predict`
- **且返回格式为**：
  ```json
  {
    "detections": [
      {
        "class_id": 5,
        "confidence": 0.92,
        "cx": 0.65,
        "cy": 0.32,
        "width": 0.1,
        "height": 0.05
      }
    ]
  }