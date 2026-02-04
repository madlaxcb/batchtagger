import json
import os
import sys
import csv
import time
import tempfile
import locale
import ctypes
import traceback
import faulthandler

APP_DIR = os.path.abspath(
    os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__)
)
base_dir = APP_DIR
if getattr(sys, "frozen", False):
    base_dir = getattr(sys, "_MEIPASS", APP_DIR)
    capi_dir = os.path.join(base_dir, "onnxruntime", "capi")
    path_parts = [base_dir]
    if os.path.isdir(capi_dir):
        path_parts.append(capi_dir)
    os.environ["PATH"] = ";".join(path_parts + [os.environ.get("PATH", "")])
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(base_dir)
            if os.path.isdir(capi_dir):
                os.add_dll_directory(capi_dir)
        except Exception:
            pass
SITE_PACKAGES = os.path.join(APP_DIR, "python_env", "Lib", "site-packages")
if os.path.isdir(SITE_PACKAGES) and SITE_PACKAGES not in sys.path:
    sys.path.insert(0, SITE_PACKAGES)
ORT_CAPI_DIR = (
    os.path.join(base_dir, "onnxruntime", "capi")
    if getattr(sys, "frozen", False)
    else os.path.join(SITE_PACKAGES, "onnxruntime", "capi")
)
PY_ENV_DIR = os.path.join(APP_DIR, "python_env")
LIB_BIN_DIR = os.path.join(PY_ENV_DIR, "Library", "bin")
for dll_dir in [ORT_CAPI_DIR, PY_ENV_DIR, LIB_BIN_DIR]:
    if os.path.isdir(dll_dir):
        os.environ["PATH"] = ";".join([dll_dir, os.environ.get("PATH", "")])
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass

try:
    import onnxruntime as _ort
except Exception:
    _ort = None

from PyQt6.QtCore import (
    Qt,
    QObject,
    pyqtSignal,
    QThread,
    QProcess,
    QProcessEnvironment,
    QEvent,
    QPoint,
    QRect,
    QSize,
    QTimer,
)
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QCheckBox,
    QDoubleSpinBox,
    QProgressBar,
    QTextEdit,
    QToolTip,
    QGroupBox,
    QFormLayout,
    QLayout,
    QSizePolicy,
    QStackedLayout,
    QScrollArea,
    QComboBox,
)


SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")
CRASH_LOG_PATH = os.path.join(APP_DIR, "startup_error.log")
DIAG_LOG_PATH = os.path.join(APP_DIR, "diagnostics.json")
DEBUG_LOG_PATH = os.path.join(APP_DIR, "debug.log")


DEFAULT_SETTINGS = {
    "input_dir": "",
    "output_dir": "",
    "model_path": "",
    "tags_path": "",
    "general_threshold": 0.35,
    "character_threshold": 0.35,
    "include_rating": True,
    "include_character": True,
    "replace_underscore": True,
    "exclude_tags": "",
    "recursive": False,
    "comfyui_dir": "",
    "skip_failed": True,
    "skip_existing": False,
    "debug": False,
    "provider": "CPU",
}


def _write_crash(text):
    try:
        with open(CRASH_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
    except Exception:
        pass


def _write_debug(text):
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
    except Exception:
        pass


def init_crash_logging():
    def _excepthook(exc_type, exc, tb):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        content = (
            "==== crash ====\n"
            + stamp
            + "\n"
            + "".join(traceback.format_exception(exc_type, exc, tb))
        )
        _write_crash(content)
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook
    try:
        with open(CRASH_LOG_PATH, "a", encoding="utf-8") as f:
            faulthandler.enable(file=f)
    except Exception:
        pass


def write_diagnostics(settings):
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "app_dir": APP_DIR,
        "base_dir": base_dir,
        "python": sys.executable,
        "frozen": bool(getattr(sys, "frozen", False)),
        "platform": sys.platform,
        "version": sys.version,
        "settings": settings,
        "path": os.environ.get("PATH", ""),
        "crash_log": CRASH_LOG_PATH,
        "ort_capi_dir": ORT_CAPI_DIR,
        "py_env_dir": PY_ENV_DIR,
        "lib_bin_dir": LIB_BIN_DIR,
    }
    try:
        with open(DIAG_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def reset_debug_log():
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(DEBUG_LOG_PATH, "w", encoding="utf-8") as f:
            f.write(f"==== debug start {stamp} ====\n")
    except Exception:
        pass


def load_settings():
    if not os.path.isfile(SETTINGS_PATH):
        return dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = dict(DEFAULT_SETTINGS)
        merged.update({k: v for k, v in data.items() if k in DEFAULT_SETTINGS})
        return merged
    except Exception:
        return dict(DEFAULT_SETTINGS)


def save_settings(data):
    payload = dict(DEFAULT_SETTINGS)
    payload.update({k: v for k, v in data.items() if k in DEFAULT_SETTINGS})
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def find_images(root_dir, recursive):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    results = []
    if recursive:
        for base, _, files in os.walk(root_dir):
            for name in files:
                if os.path.splitext(name)[1].lower() in exts:
                    results.append(os.path.join(base, name))
    else:
        for name in os.listdir(root_dir):
            full = os.path.join(root_dir, name)
            if os.path.isfile(full) and os.path.splitext(name)[1].lower() in exts:
                results.append(full)
    results.sort()
    return results


def get_comfy_python(comfy_dir):
    if not comfy_dir:
        return None
    root_dir = os.path.abspath(comfy_dir)
    embed_candidates = []
    python_candidates = []
    for root, dirs, files in os.walk(root_dir):
        if "python.exe" in files:
            path = os.path.join(root, "python.exe")
            if os.path.basename(os.path.dirname(path)).lower() == "python_embeded":
                embed_candidates.append(path)
            else:
                python_candidates.append(path)
    if embed_candidates:
        return embed_candidates[0]
    if python_candidates:
        return python_candidates[0]
    return None


def ensure_dir(path):
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def has_avx2():
    try:
        return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(12))
    except Exception:
        return True


def detect_missing_ort_deps():
    candidates = [
        "onnxruntime.dll",
        "onnxruntime_providers_shared.dll",
        "libiomp5md.dll",
        "vcomp140.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
    ]
    missing = []
    for dll in candidates:
        if dll.startswith("onnxruntime"):
            path = os.path.join(ORT_CAPI_DIR, dll)
            if not os.path.isfile(path):
                missing.append(dll)
            continue
        try:
            ctypes.WinDLL(dll)
        except Exception:
            missing.append(dll)
    return missing


class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=6):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )
        return size

    def _do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0
        for item in self._items:
            space_x = self.spacing()
            space_y = self.spacing()
            item_size = item.sizeHint()
            next_x = x + item_size.width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item_size.width() + space_x
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(x, y, item_size.width(), item_size.height()))
            x = next_x
            line_height = max(line_height, item_size.height())
        return y + line_height - rect.y()


def read_tags_csv(tags_path):
    names = []
    general_index = None
    character_index = None
    with open(tags_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            name = row[1]
            category = row[2] if len(row) > 2 else "0"
            if general_index is None and category == "0":
                general_index = reader.line_num - 2
            if character_index is None and category == "4":
                character_index = reader.line_num - 2
            names.append(name)
    if general_index is None:
        general_index = 0
    if character_index is None:
        character_index = len(names)
    return names, general_index, character_index


def preprocess_image(image_path, size, layout):
    from PIL import Image
    import numpy as np

    image = Image.open(image_path).convert("RGB")
    ratio = float(size) / max(image.size)
    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    square = Image.new("RGB", (size, size), (255, 255, 255))
    square.paste(image, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    arr = np.asarray(square, dtype=np.float32)
    arr = arr[:, :, ::-1]
    if layout == "NHWC":
        arr = arr[None, ...]
    else:
        arr = np.transpose(arr, (2, 0, 1))
        arr = arr[None, ...]
    return arr


def load_model(model_path, provider="CPU"):
    os.environ.setdefault("ORT_LOGGING_LEVEL", "4")
    try:
        import onnxruntime as ort
    except Exception as exc:
        missing = detect_missing_ort_deps()
        if missing:
            raise RuntimeError(f"ONNXRuntime依赖缺失: {', '.join(missing)}") from exc
        raise

    providers = []
    if provider == "CUDA":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif provider == "DirectML":
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    elif provider == "CoreML":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        if provider != "CPU":
             print(json.dumps({"type": "log", "message": f"{provider}加载失败，尝试回退到CPU: {e}"}, ensure_ascii=False), flush=True)
             session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        else:
             raise e

    input_shape = session.get_inputs()[0].shape
    size = 448
    layout = "NCHW"
    if len(input_shape) >= 4:
        c = input_shape[1]
        h = input_shape[2]
        w = input_shape[3]
        if isinstance(c, int) and c in (1, 3):
            layout = "NCHW"
            if isinstance(h, int) and isinstance(w, int):
                size = min(h, w)
        else:
            h = input_shape[1]
            w = input_shape[2]
            c = input_shape[3]
            if isinstance(c, int) and c in (1, 3):
                layout = "NHWC"
                if isinstance(h, int) and isinstance(w, int):
                    size = min(h, w)
    return session, size, layout


def format_tags(
    scores,
    names,
    general_index,
    character_index,
    general_th,
    character_th,
    include_rating,
    include_character,
    replace_underscore,
    exclude_set,
):
    result = list(zip(names, scores))
    rating = []
    general = []
    character = []

    rating_section = result[:general_index]
    general_section = result[general_index:character_index]
    character_section = result[character_index:]

    if include_rating and rating_section:
        rating = [max(rating_section, key=lambda x: x[1])]
    general = [item for item in general_section if item[1] > general_th]
    if include_character:
        character = [item for item in character_section if item[1] > character_th]

    def norm_tag(tag):
        return tag.replace("_", " ") if replace_underscore else tag

    parts = []

    def append_tags(items):
        for item in items:
            name = norm_tag(item[0])
            if name.lower() in exclude_set:
                continue
            parts.append(name)

    append_tags(rating)
    append_tags(character)
    append_tags(general)
    return ", ".join(parts)


def parse_exclude_tags(text):
    items = []
    for raw in text.replace("\\n", ",").split(","):
        value = raw.strip()
        if value:
            items.append(value.lower())
    return set(items)


def build_worker_script():
    return """import json
import os
import sys
import csv

DEFAULT_SETTINGS = {
    "input_dir": "",
    "output_dir": "",
    "model_path": "",
    "tags_path": "",
    "general_threshold": 0.35,
    "character_threshold": 0.35,
    "include_rating": True,
    "include_character": True,
    "replace_underscore": True,
    "exclude_tags": "",
    "recursive": False,
    "comfyui_dir": "",
    "skip_failed": True,
    "skip_existing": False,
    "debug": False,
    "provider": "CPU",
}

def find_images(root_dir, recursive):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    results = []
    if recursive:
        for base, _, files in os.walk(root_dir):
            for name in files:
                if os.path.splitext(name)[1].lower() in exts:
                    results.append(os.path.join(base, name))
    else:
        for name in os.listdir(root_dir):
            full = os.path.join(root_dir, name)
            if os.path.isfile(full) and os.path.splitext(name)[1].lower() in exts:
                results.append(full)
    results.sort()
    return results

def ensure_dir(path):
    if not path:
        return
    os.makedirs(path, exist_ok=True)

def read_tags_csv(tags_path):
    names = []
    general_index = None
    character_index = None
    with open(tags_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            name = row[1]
            category = row[2] if len(row) > 2 else "0"
            if general_index is None and category == "0":
                general_index = reader.line_num - 2
            if character_index is None and category == "4":
                character_index = reader.line_num - 2
            names.append(name)
    if general_index is None:
        general_index = 0
    if character_index is None:
        character_index = len(names)
    return names, general_index, character_index

def preprocess_image(image_path, size, layout):
    from PIL import Image
    import numpy as np

    image = Image.open(image_path).convert("RGB")
    ratio = float(size) / max(image.size)
    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    square = Image.new("RGB", (size, size), (255, 255, 255))
    square.paste(image, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    arr = np.asarray(square, dtype=np.float32)
    arr = arr[:, :, ::-1]
    if layout == "NHWC":
        arr = arr[None, ...]
    else:
        arr = np.transpose(arr, (2, 0, 1))
        arr = arr[None, ...]
    return arr

def load_model(model_path, provider="CPU"):
    os.environ.setdefault("ORT_LOGGING_LEVEL", "4")
    import onnxruntime as ort

    providers = []
    if provider == "CUDA":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif provider == "DirectML":
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    elif provider == "CoreML":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        if provider != "CPU":
             print(json.dumps({"type": "log", "message": f"{provider}加载失败，尝试回退到CPU: {e}"}, ensure_ascii=False), flush=True)
             session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        else:
             raise e

    input_shape = session.get_inputs()[0].shape
    size = 448
    layout = "NCHW"
    if len(input_shape) >= 4:
        c = input_shape[1]
        h = input_shape[2]
        w = input_shape[3]
        if isinstance(c, int) and c in (1, 3):
            layout = "NCHW"
            if isinstance(h, int) and isinstance(w, int):
                size = min(h, w)
        else:
            h = input_shape[1]
            w = input_shape[2]
            c = input_shape[3]
            if isinstance(c, int) and c in (1, 3):
                layout = "NHWC"
                if isinstance(h, int) and isinstance(w, int):
                    size = min(h, w)
    return session, size, layout

def format_tags(scores, names, general_index, character_index, general_th, character_th, include_rating, include_character, replace_underscore, exclude_set):
    result = list(zip(names, scores))
    rating = []
    general = []
    character = []

    rating_section = result[:general_index]
    general_section = result[general_index:character_index]
    character_section = result[character_index:]

    if include_rating and rating_section:
        rating = [max(rating_section, key=lambda x: x[1])]
    general = [item for item in general_section if item[1] > general_th]
    if include_character:
        character = [item for item in character_section if item[1] > character_th]

    def norm_tag(tag):
        return tag.replace("_", " ") if replace_underscore else tag

    parts = []
    def append_tags(items):
        for item in items:
            name = norm_tag(item[0])
            if name.lower() in exclude_set:
                continue
            parts.append(name)

    append_tags(rating)
    append_tags(character)
    append_tags(general)
    return ", ".join(parts)

def parse_exclude_tags(text):
    items = []
    for raw in text.replace("\\n", ",").split(","):
        value = raw.strip()
        if value:
            items.append(value.lower())
    return set(items)

def run_worker(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        settings = json.load(f)
    settings = {**DEFAULT_SETTINGS, **settings}
    input_dir = settings["input_dir"]
    output_dir = settings["output_dir"] or input_dir
    model_path = settings["model_path"]
    tags_path = settings["tags_path"]
    provider = settings.get("provider", "CPU")

    ensure_dir(output_dir)
    images = find_images(input_dir, settings["recursive"])
    if not images:
        print(json.dumps({"type": "log", "message": "未找到可处理的图片"}, ensure_ascii=False), flush=True)
        return

    print(json.dumps({"type": "log", "message": f"待处理图片: {len(images)}"}, ensure_ascii=False), flush=True)
    print(json.dumps({"type": "log", "message": f"加载模型: {model_path} ({provider})"}, ensure_ascii=False), flush=True)
    session, size, layout = load_model(model_path, provider)
    names, general_index, character_index = read_tags_csv(tags_path)
    input_name = session.get_inputs()[0].name
    exclude_set = parse_exclude_tags(settings["exclude_tags"])
    print(json.dumps({"type": "log", "message": f"模型输入: {session.get_inputs()[0].shape} 布局: {layout} 尺寸: {size}"}, ensure_ascii=False), flush=True)
    print(json.dumps({"type": "log", "message": f"标签数量: {len(names)}"}, ensure_ascii=False), flush=True)
    print(json.dumps({"type": "log", "message": "设置: "
        f"通用阈值={settings['general_threshold']} "
        f"角色阈值={settings['character_threshold']} "
        f"包含评分={settings['include_rating']} "
        f"包含角色={settings['include_character']} "
        f"下划线替换={settings['replace_underscore']} "
        f"排除标签={settings['exclude_tags']} "
        f"递归={settings['recursive']} "
        f"失败重试={settings['skip_failed']} "
        f"仅新增={settings['skip_existing']} "
        f"Debug={settings['debug']}"}, ensure_ascii=False), flush=True)
    mismatch_logged = False

    total = len(images)
    skip_failed = settings["skip_failed"]
    skip_existing = settings["skip_existing"]
    start_time = time.time()
    
    for idx, image_path in enumerate(images, start=1):
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_dir, f"{base}.txt")
        if skip_existing and os.path.isfile(out_path):
            print(json.dumps({"type": "log", "message": f"跳过已存在({idx}/{total}): {os.path.basename(image_path)}"}, ensure_ascii=False), flush=True)
            print(json.dumps({"type": "progress", "current": idx, "total": total}, ensure_ascii=False), flush=True)
            continue
        elapsed = time.time() - start_time
        avg_time = elapsed / idx if idx > 0 else 0
        remaining = avg_time * (total - idx)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
        speed = 1 / avg_time if avg_time > 0 else 0
        
        print(json.dumps({"type": "log", "message": f"处理中({idx}/{total}) [{speed:.1f}it/s ETA: {eta_str}]: {os.path.basename(image_path)}"}, ensure_ascii=False), flush=True)
        try:
            arr = preprocess_image(image_path, size, layout)
            outputs = session.run(None, {input_name: arr})
            scores = outputs[0][0]
            if len(scores) != len(names):
                min_len = min(len(scores), len(names))
                if not mismatch_logged:
                    print(json.dumps({"type": "log", "message": f"标签数量不一致: 分数={len(scores)} 标签={len(names)}"}, ensure_ascii=False), flush=True)
                    mismatch_logged = True
                scores = scores[:min_len]
                names = names[:min_len]
                if general_index > min_len:
                    general_index = min_len
                if character_index > min_len:
                    character_index = min_len
            tags = format_tags(
                scores,
                names,
                general_index,
                character_index,
                settings["general_threshold"],
                settings["character_threshold"],
                settings["include_rating"],
                settings["include_character"],
                settings["replace_underscore"],
                exclude_set,
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(tags)
        except Exception as exc:
            if skip_failed:
                print(json.dumps({"type": "log", "message": f"处理失败已跳过({idx}/{total}): {os.path.basename(image_path)} {exc}"}, ensure_ascii=False), flush=True)
                print(json.dumps({"type": "progress", "current": idx, "total": total}, ensure_ascii=False), flush=True)
                continue
            raise
        print(json.dumps({"type": "progress", "current": idx, "total": total}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        run_worker(sys.argv[1])
"""


class TaggerWorker(QObject):
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            input_dir = self.settings["input_dir"]
            output_dir = self.settings["output_dir"] or input_dir
            model_path = self.settings["model_path"]
            tags_path = self.settings["tags_path"]
            ensure_dir(output_dir)

            images = find_images(input_dir, self.settings["recursive"])
            if not images:
                self.failed.emit("未找到可处理的图片")
                return

            if not has_avx2():
                self.failed.emit("本机CPU不支持AVX2，无法在本地加载ONNXRuntime")
                return

            provider = self.settings.get("provider", "CPU")
            self.log.emit(f"加载模型: {model_path} ({provider})")
            session, size, layout = load_model(model_path, provider)
            names, general_index, character_index = read_tags_csv(tags_path)
            input_name = session.get_inputs()[0].name
            exclude_set = parse_exclude_tags(self.settings["exclude_tags"])
            self.log.emit(
                f"模型输入: {session.get_inputs()[0].shape} 布局: {layout} 尺寸: {size}"
            )
            self.log.emit(f"执行器: {session.get_providers()[0]}")
            self.log.emit(f"标签数量: {len(names)}")
            self.log.emit(
                "设置: "
                f"通用阈值={self.settings['general_threshold']} "
                f"角色阈值={self.settings['character_threshold']} "
                f"包含评分={self.settings['include_rating']} "
                f"包含角色={self.settings['include_character']} "
                f"下划线替换={self.settings['replace_underscore']} "
                f"排除标签={self.settings['exclude_tags']} "
                f"递归={self.settings['recursive']} "
                f"失败重试={self.settings['skip_failed']} "
                f"仅新增={self.settings['skip_existing']} "
                f"Debug={self.settings['debug']}"
            )
            mismatch_logged = False
            total = len(images)
            self.log.emit(f"待处理图片: {total}")
            skip_failed = self.settings["skip_failed"]
            skip_existing = self.settings["skip_existing"]
            start_time = time.time()
            
            for idx, image_path in enumerate(images, start=1):
                if self._stopped:
                    self.log.emit("已停止")
                    break
                base = os.path.splitext(os.path.basename(image_path))[0]
                out_path = os.path.join(output_dir, f"{base}.txt")
                if skip_existing and os.path.isfile(out_path):
                    self.log.emit(
                        f"跳过已存在({idx}/{total}): {os.path.basename(image_path)}"
                    )
                    self.progress.emit(idx, total)
                    continue
                
                elapsed = time.time() - start_time
                avg_time = elapsed / idx if idx > 0 else 0
                remaining = avg_time * (total - idx)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
                speed = 1 / avg_time if avg_time > 0 else 0
                
                self.log.emit(f"处理中({idx}/{total}) [{speed:.1f}it/s ETA: {eta_str}]: {os.path.basename(image_path)}")
                try:
                    arr = preprocess_image(image_path, size, layout)
                    outputs = session.run(None, {input_name: arr})
                    scores = outputs[0][0]
                    if len(scores) != len(names):
                        min_len = min(len(scores), len(names))
                        if not mismatch_logged:
                            self.log.emit(
                                f"标签数量不一致: 分数={len(scores)} 标签={len(names)}"
                            )
                            mismatch_logged = True
                        scores = scores[:min_len]
                        names = names[:min_len]
                        if general_index > min_len:
                            general_index = min_len
                        if character_index > min_len:
                            character_index = min_len
                    tags = format_tags(
                        scores,
                        names,
                        general_index,
                        character_index,
                        self.settings["general_threshold"],
                        self.settings["character_threshold"],
                        self.settings["include_rating"],
                        self.settings["include_character"],
                        self.settings["replace_underscore"],
                        exclude_set,
                    )
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(tags)
                except Exception as exc:
                    if skip_failed:
                        self.log.emit(
                            f"处理失败已跳过({idx}/{total}): {os.path.basename(image_path)} {exc}"
                        )
                        self.progress.emit(idx, total)
                        continue
                    raise
                self.progress.emit(idx, total)
            self.finished.emit()
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("批量反推提示词")
        self.settings = load_settings()
        self.debug_log_path = DEBUG_LOG_PATH
        self.worker_thread = None
        self.worker = None
        self.proc = None
        self.exclude_tags = None
        self.exclude_tags_items = []
        self._suppress_settings_save = False
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._auto_save_settings)
        self.setAcceptDrops(True)
        self._build_ui()
        self._apply_settings()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        
        path = urls[0].toLocalFile()
        if os.path.isdir(path):
            self.input_dir.setText(path)
            # If output is empty, maybe set it? Default logic handles empty output anyway.
        elif os.path.isfile(path):
             # Maybe it's a model or tags file?
             ext = os.path.splitext(path)[1].lower()
             if ext == ".onnx":
                 self.model_path.setText(path)
             elif ext == ".csv":
                 self.tags_path.setText(path)

    def _build_ui(self):
        layout = QVBoxLayout()

        path_group = QGroupBox("路径")
        self._set_tooltip(
            path_group,
            "说明：设置输入/输出/模型/标签/ComfyUI路径\n示例：按实际目录填写",
        )
        path_form = QFormLayout()
        self.input_dir = QLineEdit()
        self.output_dir = QLineEdit()
        self.model_path = QLineEdit()
        self.tags_path = QLineEdit()
        self.comfyui_dir = QLineEdit()
        self.comfy_status = QLabel("")

        btn_input = QPushButton("选择")
        btn_output = QPushButton("选择")
        btn_model = QPushButton("选择")
        btn_tags = QPushButton("选择")
        btn_comfy = QPushButton("选择")

        btn_input.clicked.connect(self._pick_input)
        btn_output.clicked.connect(self._pick_output)
        btn_model.clicked.connect(self._pick_model)
        btn_tags.clicked.connect(self._pick_tags)
        btn_comfy.clicked.connect(self._pick_comfy)
        self.input_dir.textChanged.connect(self._validate_paths)
        self.output_dir.textChanged.connect(self._validate_paths)
        self.model_path.textChanged.connect(self._validate_paths)
        self.tags_path.textChanged.connect(self._validate_paths)
        self.comfyui_dir.textChanged.connect(self._validate_paths)
        self.input_dir.textChanged.connect(self._schedule_save)
        self.output_dir.textChanged.connect(self._schedule_save)
        self.model_path.textChanged.connect(self._schedule_save)
        self.tags_path.textChanged.connect(self._schedule_save)
        self.comfyui_dir.textChanged.connect(self._schedule_save)
        
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["CPU", "CUDA", "DirectML", "CoreML"])
        self.provider_combo.currentIndexChanged.connect(self._schedule_save)

        row_input = QHBoxLayout()
        row_input.addWidget(self.input_dir)
        row_input.addWidget(btn_input)
        row_output = QHBoxLayout()
        row_output.addWidget(self.output_dir)
        row_output.addWidget(btn_output)
        row_model = QHBoxLayout()
        row_model.addWidget(self.model_path)
        row_model.addWidget(btn_model)
        row_tags = QHBoxLayout()
        row_tags.addWidget(self.tags_path)
        row_tags.addWidget(btn_tags)
        row_comfy = QHBoxLayout()
        row_comfy.addWidget(self.comfyui_dir)
        row_comfy.addWidget(btn_comfy)

        label_input = QLabel("输入文件夹")
        label_output = QLabel("输出文件夹(空=输入)")
        label_model = QLabel("模型(onnx)")
        label_tags = QLabel("标签(tags.csv)")
        label_comfy = QLabel("ComfyUI目录")
        label_comfy_env = QLabel("ComfyUI环境")
        label_provider = QLabel("执行器")
        self._set_tooltip(label_input, "说明：待处理图片所在文件夹\n示例：D:\\images")
        self._set_tooltip(
            label_output,
            "说明：标签输出目录，留空则使用输入目录\n示例：D:\\images\\tags",
        )
        self._set_tooltip(
            label_model, "说明：WD14 onnx模型文件路径\n示例：D:\\models\\wd14.onnx"
        )
        self._set_tooltip(
            label_tags, "说明：标签CSV文件路径\n示例：D:\\models\\tags.csv"
        )
        self._set_tooltip(
            label_comfy, "说明：包含python.exe的ComfyUI目录\n示例：D:\\ComfyUI"
        )
        self._set_tooltip(
            label_comfy_env,
            "说明：识别到的ComfyUI环境python.exe\n示例：D:\\ComfyUI\\python\\python.exe",
        )
        self._set_tooltip(
            label_provider, "说明：选择ONNX推理后端\n注意：CUDA/DirectML需要对应环境支持",
        )
        path_form.addRow(label_input, self._wrap(row_input))
        path_form.addRow(label_output, self._wrap(row_output))
        path_form.addRow(label_model, self._wrap(row_model))
        path_form.addRow(label_tags, self._wrap(row_tags))
        path_form.addRow(label_comfy, self._wrap(row_comfy))
        path_form.addRow(label_comfy_env, self.comfy_status)
        path_form.addRow(label_provider, self.provider_combo)
        path_group.setLayout(path_form)

        opt_group = QGroupBox("参数")
        self._set_tooltip(
            opt_group, "说明：设置阈值、过滤与处理选项\n示例：默认阈值0.35"
        )
        opt_form = QFormLayout()
        self.general_th = QDoubleSpinBox()
        self.general_th.setRange(0.0, 1.0)
        self.general_th.setSingleStep(0.01)
        self.character_th = QDoubleSpinBox()
        self.character_th.setRange(0.0, 1.0)
        self.character_th.setSingleStep(0.01)
        self.general_th.valueChanged.connect(self._schedule_save)
        self.character_th.valueChanged.connect(self._schedule_save)
        self.exclude_tags = QLineEdit()
        self.exclude_tags.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.exclude_tag_list_label = QLabel("排除标签列表")
        self.exclude_tag_sort_combo = QComboBox()
        self.exclude_tag_sort_combo.addItems(
            [
                "按添加顺序 (新->旧)",
                "按添加顺序 (旧->新)",
                "按名称 (A->Z)",
                "按名称 (Z->A)",
            ]
        )
        self.exclude_tag_sort_combo.setCurrentIndex(0)
        self.exclude_tag_sort_combo.currentIndexChanged.connect(
            self._render_exclude_tag_list
        )

        self.exclude_tag_list_container = QWidget()
        self.exclude_tag_list_layout = FlowLayout()
        self.exclude_tag_list_container.setLayout(self.exclude_tag_list_layout)
        self.exclude_tag_list_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.exclude_tag_list_scroll = QScrollArea()
        self.exclude_tag_list_scroll.setWidgetResizable(False)
        self.exclude_tag_list_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.exclude_tag_list_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.exclude_tag_list_scroll.setWidget(self.exclude_tag_list_container)
        self.exclude_tag_list_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self.include_rating = QCheckBox("包含评分标签")
        self.include_character = QCheckBox("包含角色标签")
        self.replace_underscore = QCheckBox("下划线替换为空格")
        self.recursive = QCheckBox("包含子目录")
        self.skip_failed = QCheckBox("失败重试")
        self.skip_existing = QCheckBox("仅处理新增图片")
        self.debug_mode = QCheckBox("Debug模式")
        self.include_rating.toggled.connect(self._schedule_save)
        self.include_character.toggled.connect(self._schedule_save)
        self.replace_underscore.toggled.connect(self._schedule_save)
        self.recursive.toggled.connect(self._schedule_save)
        self.skip_failed.toggled.connect(self._schedule_save)
        self.skip_existing.toggled.connect(self._schedule_save)
        self.debug_mode.toggled.connect(self._schedule_save)

        label_general = QLabel("通用阈值")
        label_character = QLabel("角色阈值")
        label_exclude = QLabel("排除标签")
        self._set_tooltip(label_general, "说明：通用标签阈值，越高越严格\n示例：0.35")
        self._set_tooltip(label_character, "说明：角色标签阈值，越高越严格\n示例：0.35")
        self._set_tooltip(
            label_exclude, "说明：用逗号或换行分隔要排除的标签\n示例：lowres, bad hands"
        )
        opt_form.addRow(label_general, self.general_th)
        opt_form.addRow(label_character, self.character_th)
        opt_form.addRow(label_exclude, self.exclude_tags)
        
        row_exclude_header = QHBoxLayout()
        row_exclude_header.addWidget(self.exclude_tag_list_label)
        row_exclude_header.addStretch()
        row_exclude_header.addWidget(self.exclude_tag_sort_combo)
        
        opt_form.addRow(row_exclude_header)
        opt_form.addRow(self.exclude_tag_list_scroll)
        opt_form.addRow(self.include_rating)
        opt_form.addRow(self.include_character)
        opt_form.addRow(self.replace_underscore)
        opt_form.addRow(self.recursive)
        opt_form.addRow(self.skip_failed)
        opt_form.addRow(self.skip_existing)
        opt_form.addRow(self.debug_mode)
        opt_group.setLayout(opt_form)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始")
        self.btn_stop = QPushButton("停止")
        self.btn_precheck = QPushButton("预检")
        self.btn_open_output = QPushButton("打开输出目录")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_precheck.clicked.connect(self._precheck)
        self.btn_open_output.clicked.connect(self._open_output_dir)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_precheck)
        btn_row.addWidget(self.btn_open_output)

        self.progress = QProgressBar()
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout.addWidget(path_group)
        layout.addWidget(opt_group)
        layout.addLayout(btn_row)
        layout.addWidget(self.progress)
        layout.addWidget(self.log)
        self.setLayout(layout)

        self._set_tooltip(
            self.input_dir, "说明：待处理图片所在文件夹\n示例：D:\\images"
        )
        self._set_tooltip(
            self.output_dir,
            "说明：标签输出目录，留空则使用输入目录\n示例：D:\\images\\tags",
        )
        self._set_tooltip(
            self.model_path, "说明：WD14 onnx模型文件路径\n示例：D:\\models\\wd14.onnx"
        )
        self._set_tooltip(
            self.tags_path, "说明：标签CSV文件路径\n示例：D:\\models\\tags.csv"
        )
        self._set_tooltip(
            self.comfyui_dir, "说明：包含python.exe的ComfyUI目录\n示例：D:\\ComfyUI"
        )
        self._set_tooltip(
            self.comfy_status,
            "说明：识别到的ComfyUI环境python.exe\n示例：D:\\ComfyUI\\python\\python.exe",
        )
        self._set_tooltip(btn_input, "说明：选择输入文件夹\n示例：D:\\images")
        self._set_tooltip(btn_output, "说明：选择输出文件夹\n示例：D:\\images\\tags")
        self._set_tooltip(btn_model, "说明：选择onnx模型文件\n示例：wd14.onnx")
        self._set_tooltip(btn_tags, "说明：选择标签CSV文件\n示例：tags.csv")
        self._set_tooltip(btn_comfy, "说明：选择ComfyUI目录\n示例：D:\\ComfyUI")
        self._set_tooltip(self.general_th, "说明：通用标签阈值，越高越严格\n示例：0.35")
        self._set_tooltip(
            self.character_th, "说明：角色标签阈值，越高越严格\n示例：0.35"
        )
        self._set_tooltip(
            self.exclude_tags,
            "说明：用逗号或换行分隔要排除的标签\n示例：lowres, bad hands",
        )
        self._set_tooltip(
            self.exclude_tag_list_scroll,
            "说明：当前排除标签列表，点击×移除\n示例：bad hands",
        )
        self._set_tooltip(
            self.include_rating, "说明：是否输出评分标签\n示例：勾选后会包含rating"
        )
        self._set_tooltip(
            self.include_character, "说明：是否输出角色标签\n示例：勾选后包含角色名"
        )
        self._set_tooltip(
            self.replace_underscore,
            "说明：将标签中的下划线替换为空格\n示例：blue_hair → blue hair",
        )
        self._set_tooltip(
            self.recursive, "说明：是否包含子目录内图片\n示例：勾选后遍历子文件夹"
        )
        self._set_tooltip(
            self.skip_failed, "说明：单图失败时跳过，不终止任务\n示例：识别失败仍继续"
        )
        self._set_tooltip(
            self.skip_existing,
            "说明：已有同名txt时跳过\n示例：image1.txt已存在则不处理",
        )
        self._set_tooltip(
            self.debug_mode, "说明：启用后生成诊断文件\n示例：diagnostics.json"
        )
        self._set_tooltip(
            self.btn_start, "说明：开始处理当前设置\n示例：点击后开始生成标签"
        )
        self._set_tooltip(self.btn_stop, "说明：停止当前任务\n示例：处理中点击可停止")
        self._set_tooltip(
            self.btn_precheck, "说明：检查路径/模型/标签/图片数量\n示例：开始前先预检"
        )
        self._set_tooltip(
            self.btn_open_output, "说明：打开输出目录\n示例：一键打开标签文件夹"
        )
        self._set_tooltip(self.progress, "说明：显示处理进度(当前/总数)\n示例：12/100")
        self._set_tooltip(self.log, "说明：实时日志输出\n示例：处理中(12/100): xxx.png")
        self.exclude_tags.returnPressed.connect(self._on_exclude_tags_enter)
        self.exclude_tags.installEventFilter(self)
        self.exclude_tag_list_container.installEventFilter(self)
        self.exclude_tag_list_scroll.installEventFilter(self)
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)

        base_height = self.exclude_tags.sizeHint().height()
        if base_height > 0:
            self.exclude_tags.setFixedHeight(base_height)
            self.exclude_tag_row_height = base_height + 6
        else:
            self.exclude_tag_row_height = 28

    def _wrap(self, layout):
        container = QWidget()
        container.setLayout(layout)
        return container

    def _schedule_save(self):
        if self._suppress_settings_save:
            return
        self._save_timer.start(200)

    def _auto_save_settings(self):
        if self._suppress_settings_save:
            return
        self._collect_settings()

    def _set_tooltip(self, widget, text):
        widget.setToolTip(text)
        widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.ToolTip:
            text = obj.toolTip() if hasattr(obj, "toolTip") else ""
            if text:
                self._show_tooltip(obj, text)
                return True
        if event.type() == QEvent.Type.Leave:
            QToolTip.hideText()
        return super().eventFilter(obj, event)

    def _show_tooltip(self, widget, text):
        pos = widget.mapToGlobal(widget.rect().topRight())
        pos = QPoint(pos.x() + 12, pos.y() + 12)
        screen = (
            widget.screen().availableGeometry()
            if widget.screen()
            else QApplication.primaryScreen().availableGeometry()
        )
        x = min(max(pos.x(), screen.left() + 8), screen.right() - 240)
        y = min(max(pos.y(), screen.top() + 8), screen.bottom() - 120)
        QToolTip.showText(QPoint(x, y), text, widget)

    def _split_exclude_tags(self, text):
        raw = text.replace("，", ",").replace("\\n", ",").replace("\n", ",")
        parts = [item.strip() for item in raw.split(",") if item.strip()]
        seen = set()
        result = []
        for item in parts:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _clear_exclude_tag_list(self):
        while self.exclude_tag_list_layout.count():
            item = self.exclude_tag_list_layout.takeAt(0)
            widget = item.widget() if item else None
            if widget:
                widget.setParent(None)
        self._update_exclude_tag_list_height()

    def _update_exclude_tag_list_height(self):
        width = (
            self.exclude_tag_list_scroll.viewport().width()
            if self.exclude_tag_list_scroll
            else 0
        )
        if width <= 0:
            width = max(
                self.exclude_tag_list_scroll.width(), self.exclude_tags.width(), 1
            )
            sb = self.exclude_tag_list_scroll.verticalScrollBar()
            if sb:
                width -= sb.sizeHint().width()

        self.exclude_tag_list_container.setFixedWidth(width)
        height = self.exclude_tag_list_layout.heightForWidth(width)
        height = max(height, self.exclude_tag_row_height)
        max_height = self.exclude_tag_row_height * 5
        self.exclude_tag_list_container.setFixedSize(width, height)
        self.exclude_tag_list_scroll.setFixedHeight(max_height)

    def _create_exclude_tag_chip(self, tag):
        container = QWidget(self.exclude_tag_list_container)
        container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout()
        row.setContentsMargins(6, 2, 6, 2)
        row.setSpacing(4)
        row.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        stack_container = QWidget(container)
        stack = QStackedLayout()
        tag_btn = QPushButton(tag, stack_container)
        tag_btn.setFlat(True)
        copy_btn = QPushButton("复制", stack_container)
        copy_btn.setFlat(True)
        btn_width = max(tag_btn.sizeHint().width(), copy_btn.sizeHint().width())
        btn_height = max(tag_btn.sizeHint().height(), copy_btn.sizeHint().height())
        tag_btn.setFixedSize(btn_width, btn_height)
        copy_btn.setFixedSize(btn_width, btn_height)
        stack_container.setFixedSize(btn_width, btn_height)
        stack.addWidget(tag_btn)
        stack.addWidget(copy_btn)
        stack_container.setLayout(stack)
        tag_btn.clicked.connect(lambda: stack.setCurrentWidget(copy_btn))
        copy_btn.clicked.connect(lambda: self._copy_exclude_tag(tag, stack, tag_btn))
        close_btn = QPushButton("×", container)
        close_btn.setFixedSize(20, 20)
        close_btn.clicked.connect(lambda _, t=tag: self._remove_exclude_tag(t))
        row.addWidget(stack_container)
        row.addWidget(close_btn)
        container.setLayout(row)
        return container

    def _copy_exclude_tag(self, tag, stack, tag_btn):
        QApplication.clipboard().setText(tag)
        stack.setCurrentWidget(tag_btn)

    def _render_exclude_tag_list(self):
        self._clear_exclude_tag_list()
        
        sort_mode = self.exclude_tag_sort_combo.currentIndex()
        display_items = list(self.exclude_tags_items)
        
        if sort_mode == 0:  # Added Order (New->Old)
            display_items.reverse()
        elif sort_mode == 1:  # Added Order (Old->New)
            pass
        elif sort_mode == 2:  # Name (A->Z)
            display_items.sort(key=lambda x: x.lower())
        elif sort_mode == 3:  # Name (Z->A)
            display_items.sort(key=lambda x: x.lower(), reverse=True)
            
        for tag in display_items:
            self.exclude_tag_list_layout.addWidget(self._create_exclude_tag_chip(tag))
        self.exclude_tag_list_layout.invalidate()
        self.exclude_tag_list_container.updateGeometry()
        self.exclude_tag_list_scroll.viewport().updateGeometry()
        self._update_exclude_tag_list_height()
        QTimer.singleShot(0, self._update_exclude_tag_list_height)

    def _set_exclude_tags_items(self, tags):
        self.exclude_tags_items = list(tags)
        self._render_exclude_tag_list()
        self._schedule_save()

    def _add_exclude_tags(self, tags):
        if not tags:
            return
        
        existing = {t.lower() for t in self.exclude_tags_items}
        new_items = []
        for tag in tags:
            if tag.lower() not in existing:
                new_items.append(tag)
                existing.add(tag.lower())
        
        self.exclude_tags_items.extend(new_items)
        self._render_exclude_tag_list()
        self._schedule_save()

    def _on_exclude_tags_enter(self):
        text = self.exclude_tags.text().strip()
        if not text:
            return
        tags = self._split_exclude_tags(text)
        if not tags:
            return
        self._add_exclude_tags(tags)
        self.exclude_tags.setText("")
        self._render_exclude_tag_list()

    def _remove_exclude_tag(self, tag):
        updated = []
        removed = False
        for value in self.exclude_tags_items:
            if not removed and value.lower() == tag.lower():
                removed = True
                continue
            updated.append(value)
        self.exclude_tags_items = updated
        self._render_exclude_tag_list()
        self._schedule_save()

    def _apply_settings(self):
        self._suppress_settings_save = True
        self.input_dir.setText(self.settings["input_dir"])
        self.output_dir.setText(self.settings["output_dir"])
        self.model_path.setText(self.settings["model_path"])
        self.tags_path.setText(self.settings["tags_path"])
        self.comfyui_dir.setText(self.settings["comfyui_dir"])
        self.provider_combo.setCurrentText(self.settings.get("provider", "CPU"))
        self.general_th.setValue(self.settings["general_threshold"])
        self.character_th.setValue(self.settings["character_threshold"])
        self._set_exclude_tags_items(
            self._split_exclude_tags(self.settings["exclude_tags"])
        )
        self.exclude_tags.setText("")
        self.include_rating.setChecked(self.settings["include_rating"])
        self.include_character.setChecked(self.settings["include_character"])
        self.replace_underscore.setChecked(self.settings["replace_underscore"])
        self.recursive.setChecked(self.settings["recursive"])
        self.skip_failed.setChecked(self.settings["skip_failed"])
        self.skip_existing.setChecked(self.settings["skip_existing"])
        self.debug_mode.setChecked(self.settings["debug"])
        self._update_comfy_status()
        self._validate_paths()
        self._suppress_settings_save = False

    def _collect_settings(self):
        self.settings = {
            "input_dir": self.input_dir.text().strip(),
            "output_dir": self.output_dir.text().strip(),
            "model_path": self.model_path.text().strip(),
            "tags_path": self.tags_path.text().strip(),
            "general_threshold": self.general_th.value(),
            "character_threshold": self.character_th.value(),
            "exclude_tags": ", ".join(self.exclude_tags_items),
            "include_rating": self.include_rating.isChecked(),
            "include_character": self.include_character.isChecked(),
            "replace_underscore": self.replace_underscore.isChecked(),
            "recursive": self.recursive.isChecked(),
            "skip_failed": self.skip_failed.isChecked(),
            "skip_existing": self.skip_existing.isChecked(),
            "debug": self.debug_mode.isChecked(),
            "comfyui_dir": self.comfyui_dir.text().strip(),
            "provider": self.provider_combo.currentText(),
        }
        save_settings(self.settings)

    def _pick_input(self):
        path = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if path:
            self.input_dir.setText(path)

    def _pick_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_dir.setText(path)

    def _pick_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", filter="ONNX (*.onnx)")
        if path:
            self.model_path.setText(path)

    def _pick_tags(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择标签CSV", filter="CSV (*.csv)")
        if path:
            self.tags_path.setText(path)

    def _pick_comfy(self):
        path = QFileDialog.getExistingDirectory(self, "选择ComfyUI目录")
        if path:
            self.comfyui_dir.setText(path)
            self._update_comfy_status()

    def _append_log(self, text):
        self.log.append(text)
        self._append_debug_log(text)

    def _append_debug_log(self, text):
        if not self.settings.get("debug"):
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_debug(f"[{stamp}] {text}")

    def _decode_proc_data(self, data):
        if not data:
            return ""
        encodings = ["utf-8", locale.getpreferredencoding(False), "gb18030", "cp936"]
        for enc in encodings:
            try:
                return data.decode(enc)
            except Exception:
                continue
        return data.decode("utf-8", errors="replace")

    def _set_running(self, running):
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_precheck.setEnabled(not running)
        self.btn_open_output.setEnabled(not running)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_exclude_tag_list_height()

    def _start(self):
        self._collect_settings()
        if self.settings.get("debug") and not os.path.isfile(self.debug_log_path):
            reset_debug_log()
            self._append_debug_log("启动任务")
        self._validate_paths()
        self.log.clear()
        if not self._precheck():
            return

        self.progress.setMaximum(1)
        self.progress.setValue(0)
        self.progress.setFormat("%v/%m")
        self._set_running(True)
        self._append_log("开始处理")

        self._start_external()

    def _start_local(self):
        self.worker_thread = QThread()
        self.worker = TaggerWorker(self.settings)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._append_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.start()

    def _validate_paths(self):
        def mark(widget, ok):
            widget.setStyleSheet("" if ok else "border: 1px solid #d9534f;")

        input_ok = os.path.isdir(self.input_dir.text().strip())
        output_text = self.output_dir.text().strip()
        output_ok = True if not output_text else os.path.isdir(output_text)
        model_ok = os.path.isfile(self.model_path.text().strip())
        tags_ok = os.path.isfile(self.tags_path.text().strip())
        comfy_dir = self.comfyui_dir.text().strip()
        comfy_ok = os.path.isdir(comfy_dir) and bool(get_comfy_python(comfy_dir))
        mark(self.input_dir, input_ok)
        mark(self.output_dir, output_ok)
        mark(self.model_path, model_ok)
        mark(self.tags_path, tags_ok)
        mark(self.comfyui_dir, comfy_ok)
        self._update_comfy_status()
        return input_ok and output_ok and model_ok and tags_ok and comfy_ok

    def _precheck(self):
        self._collect_settings()
        self._validate_paths()
        issues = []
        if not os.path.isdir(self.settings["input_dir"]):
            issues.append("输入文件夹无效")
        output_dir = self.settings["output_dir"] or self.settings["input_dir"]
        if output_dir:
            ensure_dir(output_dir)
        if output_dir and not os.path.isdir(output_dir):
            issues.append("输出文件夹无效")
        if not os.path.isfile(self.settings["model_path"]):
            issues.append("模型文件无效")
        if not os.path.isfile(self.settings["tags_path"]):
            issues.append("标签CSV无效")
        comfy_py = (
            get_comfy_python(self.settings["comfyui_dir"])
            if os.path.isdir(self.settings["comfyui_dir"])
            else None
        )
        if not comfy_py:
            issues.append("ComfyUI目录内未找到python.exe")
        count = (
            len(find_images(self.settings["input_dir"], self.settings["recursive"]))
            if os.path.isdir(self.settings["input_dir"])
            else 0
        )
        self._append_log(f"预检图片数量: {count}")
        if count == 0:
            issues.append("未找到可处理的图片")
        if issues:
            for it in issues:
                self._append_log(f"预检问题: {it}")
        else:
            self._append_log("预检通过")
        return not issues

    def _open_output_dir(self):
        out = self.output_dir.text().strip() or self.input_dir.text().strip()
        if out and os.path.isdir(out):
            os.startfile(out)
        else:
            self._append_log("输出目录无效，无法打开")

    def _start_external(self):
        comfy_python = get_comfy_python(self.settings["comfyui_dir"])
        if not comfy_python:
            self._append_log("未找到ComfyUI环境的python.exe")
            self._set_running(False)
            return
        self._append_log(f"使用ComfyUI环境: {comfy_python}")

        payload = dict(self.settings)
        payload["_worker"] = True
        temp_path = os.path.join(APP_DIR, "_worker_config.json")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        self.proc = QProcess(self)
        self.proc.setProgram(comfy_python)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        env.insert("ORT_LOGGING_LEVEL", "4")
        self.proc.setProcessEnvironment(env)
        if getattr(sys, "frozen", False):
            worker_path = os.path.join(tempfile.gettempdir(), "batch_tagger_worker.py")
            with open(worker_path, "w", encoding="utf-8") as f:
                f.write(build_worker_script())
            self.proc.setArguments([worker_path, temp_path])
        else:
            self.proc.setArguments([os.path.abspath(__file__), "--worker", temp_path])
        self.proc.readyReadStandardOutput.connect(self._read_proc_stdout)
        self.proc.readyReadStandardError.connect(self._read_proc_stderr)
        self.proc.finished.connect(self._on_proc_finished)
        self.proc.start()

    def _read_proc_stdout(self):
        data = self._decode_proc_data(bytes(self.proc.readAllStandardOutput()))
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            self._append_debug_log(f"STDOUT {line}")
            try:
                msg = json.loads(line)
                if msg.get("type") == "progress":
                    self._on_progress(msg.get("current", 0), msg.get("total", 0))
                elif msg.get("type") == "log":
                    self._append_log(msg.get("message", ""))
            except Exception:
                self._append_log(line)

    def _read_proc_stderr(self):
        data = self._decode_proc_data(bytes(self.proc.readAllStandardError()))
        if data.strip():
            self._append_debug_log(f"STDERR {data.strip()}")
            self._append_log(data.strip())

    def _on_proc_finished(self, code, _status):
        self._append_log(f"进程结束: {code}")
        self._set_running(False)

    def _on_progress(self, current, total):
        if total > 0:
            if self.progress.maximum() != total:
                self.progress.setMaximum(total)
                self.progress.setFormat("%v/%m")
            self.progress.setValue(current)

    def _on_failed(self, message):
        self._append_log(f"失败: {message}")
        self._set_running(False)

    def _on_finished(self):
        self._append_log("完成")
        self._set_running(False)

    def _stop(self):
        if self.worker:
            self.worker.stop()
        if self.proc:
            self.proc.kill()
        self._set_running(False)

    def _update_comfy_status(self):
        path = self.comfyui_dir.text().strip()
        if not path:
            self.comfy_status.setText("未设置")
            return
        comfy_python = get_comfy_python(path)
        if comfy_python:
            self.comfy_status.setText(f"已识别: {comfy_python}")
        else:
            self.comfy_status.setText("未找到")


def run_worker(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        settings = json.load(f)
    settings = {**DEFAULT_SETTINGS, **settings}
    input_dir = settings["input_dir"]
    output_dir = settings["output_dir"] or input_dir
    model_path = settings["model_path"]
    tags_path = settings["tags_path"]
    provider = settings.get("provider", "CPU")

    ensure_dir(output_dir)
    images = find_images(input_dir, settings["recursive"])
    if not images:
        print(
            json.dumps(
                {"type": "log", "message": "未找到可处理的图片"}, ensure_ascii=False
            ),
            flush=True,
        )
        return

    if not has_avx2():
        print(
            json.dumps(
                {
                    "type": "log",
                    "message": "本机CPU不支持AVX2，无法在本地加载ONNXRuntime",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    print(
        json.dumps(
            {"type": "log", "message": f"加载模型: {model_path} ({provider})"}, ensure_ascii=False
        ),
        flush=True,
    )
    session, size, layout = load_model(model_path, provider)
    names, general_index, character_index = read_tags_csv(tags_path)
    input_name = session.get_inputs()[0].name
    exclude_set = parse_exclude_tags(settings["exclude_tags"])
    print(
        json.dumps(
            {
                "type": "log",
                "message": f"模型输入: {session.get_inputs()[0].shape} 布局: {layout} 尺寸: {size}",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(
        json.dumps(
            {"type": "log", "message": "执行器: CPUExecutionProvider"},
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(
        json.dumps(
            {"type": "log", "message": f"标签数量: {len(names)}"}, ensure_ascii=False
        ),
        flush=True,
    )
    print(
        json.dumps(
            {
                "type": "log",
                "message": "设置: "
                f"通用阈值={settings['general_threshold']} "
                f"角色阈值={settings['character_threshold']} "
                f"包含评分={settings['include_rating']} "
                f"包含角色={settings['include_character']} "
                f"下划线替换={settings['replace_underscore']} "
                f"排除标签={settings['exclude_tags']} "
                f"递归={settings['recursive']} "
                f"失败重试={settings['skip_failed']} "
                f"仅新增={settings['skip_existing']} "
                f"Debug={settings['debug']}",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    mismatch_logged = False

    total = len(images)
    skip_failed = settings["skip_failed"]
    skip_existing = settings["skip_existing"]
    for idx, image_path in enumerate(images, start=1):
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_dir, f"{base}.txt")
        if skip_existing and os.path.isfile(out_path):
            print(
                json.dumps(
                    {
                        "type": "log",
                        "message": f"跳过已存在({idx}/{total}): {os.path.basename(image_path)}",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            print(
                json.dumps(
                    {"type": "progress", "current": idx, "total": total},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        print(
            json.dumps(
                {
                    "type": "log",
                    "message": f"处理中({idx}/{total}): {os.path.basename(image_path)}",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        try:
            arr = preprocess_image(image_path, size, layout)
            outputs = session.run(None, {input_name: arr})
            scores = outputs[0][0]
            if len(scores) != len(names):
                min_len = min(len(scores), len(names))
                if not mismatch_logged:
                    print(
                        json.dumps(
                            {
                                "type": "log",
                                "message": f"标签数量不一致: 分数={len(scores)} 标签={len(names)}",
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    mismatch_logged = True
                scores = scores[:min_len]
                names = names[:min_len]
                if general_index > min_len:
                    general_index = min_len
                if character_index > min_len:
                    character_index = min_len
            tags = format_tags(
                scores,
                names,
                general_index,
                character_index,
                settings["general_threshold"],
                settings["character_threshold"],
                settings["include_rating"],
                settings["include_character"],
                settings["replace_underscore"],
                exclude_set,
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(tags)
        except Exception as exc:
            if skip_failed:
                print(
                    json.dumps(
                        {
                            "type": "log",
                            "message": f"处理失败已跳过({idx}/{total}): {os.path.basename(image_path)} {exc}",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                print(
                    json.dumps(
                        {"type": "progress", "current": idx, "total": total},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            raise
        print(
            json.dumps(
                {"type": "progress", "current": idx, "total": total}, ensure_ascii=False
            ),
            flush=True,
        )


def main():
    init_crash_logging()
    settings = load_settings()
    if settings.get("debug"):
        reset_debug_log()
        write_diagnostics(settings)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_debug(f"[{stamp}] 应用启动")
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2])
        return
    try:
        app = QApplication(sys.argv)
        w = MainWindow()
        w.resize(900, 700)
        w.show()
        sys.exit(app.exec())
    except Exception:
        _write_crash("".join(traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
