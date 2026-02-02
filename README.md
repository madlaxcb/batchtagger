# 批量反推提示词（BatchTagger）

用于批量图片的标签推断工具，基于 WD14 ONNX 模型与 tags.csv，输出与图片同名的 .txt 标签文件。提供图形界面，支持阈值调整、排除标签、递归目录处理与失败处理策略。

<img width="905" height="911" alt="图片" src="https://github.com/user-attachments/assets/8a73f338-ef3e-47dc-aa2d-90fabc1150f2" />



## 主要功能

- 批量处理图片（jpg/png/webp/bmp/gif）并生成标签文件
- 通用/角色阈值可调，支持评分与角色标签输出
- 支持下划线替换为空格、排除标签列表
- 支持递归子目录、跳过失败或已存在输出
- 自动保存设置到 settings.json

## 运行环境

- Windows
- Python 3.10+（源码运行）
- 依赖库：PyQt6、onnxruntime、Pillow、numpy
- 模型与标签文件：WD14 onnx 模型与 tags.csv

## 使用方式

1. 运行程序：

   ```bash
   python batch_tagger.py
   ```

2. 在界面中填写：
   - 输入文件夹
   - 输出文件夹（为空则使用输入目录）
   - 模型（onnx）路径
   - 标签（tags.csv）路径
   - 可选的 ComfyUI 目录

3. 设置参数并点击“开始”。

## 输出说明

- 每张图片输出同名的 .txt 文件，写入推断得到的标签
- 输出目录不存在时会自动创建

## 配置与日志

- settings.json：保存界面输入与参数设置
- startup_error.log：启动异常记录
- diagnostics.json：诊断信息
- debug.log：调试日志

## 打包

使用 PyInstaller 的 spec 文件进行打包：

```bash
pyinstaller BatchTagger.spec
```

## 目录结构

- batch_tagger.py：主程序
- BatchTagger.spec：PyInstaller 打包配置
- comfyui-WD14-Tagger/：ComfyUI WD14 Tagger 依赖资源
