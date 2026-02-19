# 批量反推提示词（BatchTagger）

用于批量图片的标签推断工具，基于 WD14 ONNX 模型与 tags.csv，输出与图片同名的 .txt 标签文件。提供图形界面，支持阈值调整、排除标签、递归目录处理与失败处理策略。

<img width="902" height="843" alt="图片" src="https://github.com/user-attachments/assets/ff20d190-0ba7-44a4-8cf1-7d9b5fa8265a" />


<img width="2467" height="908" alt="图片" src="https://github.com/user-attachments/assets/cbfc7827-da1a-42a0-8a9d-6e0f9a5eb328" />


## 主要功能

- 批量处理图片（jpg/png/webp/bmp/gif）并生成标签文件
- 文件与文件夹拖拽输入
- 实时显示处理速度与剩余时间（ETA）
- 通用/角色阈值可调，支持评分与角色标签输出
- 支持下划线替换为空格、排除标签列表（支持按添加顺序/名称排序，重复检测，多套配置）
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
- comfyui-WD14-Tagger/：[ComfyUI WD14 Tagger](https://huggingface.co/SmilingWolf) 依赖资源

---

# BatchTagger

BatchTagger is a powerful GUI application for batch tagging images using ONNX Runtime and WD14 Tagger models. It allows users to efficiently tag large collections of images with high accuracy and customizable settings.

## Features

- **Batch Processing**: Tag multiple images in a directory recursively.
- **High Performance**: Optimized startup and processing speed using ONNX Runtime.
- **Smart Filtering**: Real-time validation for exclude tags to prevent duplicates.
- **User-Friendly Interface**: Easy-to-use GUI for setting thresholds, excluded tags, and model selection.
- **Responsive Layout**: Modern grid layout for settings to maximize screen real estate.
- **Cross-Platform**: Built with Python and Qt (PyQt6), compatible with Windows, Linux, and macOS.

## Installation

### Prerequisites

- Python 3.10 or higher
- Git

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/madlaxcb/batchtagger.git
   cd batchtagger
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have a compatible ONNX Runtime installed for your hardware (CPU/GPU).*

## Usage

### Running from Source

```bash
python batch_tagger.py
```

### Using the Executable

Download the latest release from the [Releases](https://github.com/madlaxcb/batchtagger/releases) page. Extract the archive and run `BatchTagger.exe`.

1. **Select Directory**: Choose the folder containing your images.
2. **Configure Settings**: Adjust thresholds for general and character tags.
3. **Exclude Tags**: Add any tags you want to exclude from the output.
4. **Start Tagging**: Click the "Start" button to begin processing.

## Building from Source

To build a standalone executable using PyInstaller:

1. Install development dependencies:
   ```bash
   pip install pyinstaller
   ```

2. Run the build command:
   ```bash
   pyinstaller BatchTagger.spec
   ```

The executable will be located in the `dist/BatchTagger/` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
