import gradio as gr
import cv2
import numpy as np
import tempfile
from pathlib import Path
import os
import zipfile
import re
import pandas as pd
from yolo_model import YOLOImageProcessor
from wrgb_fit_regression import BrightnessRegression

base_tmp = tempfile.gettempdir()
os.makedirs(base_tmp, exist_ok=True)

# 全局模型路徑，可根據需求修改
MODEL_PATH = "/app/best.pt"
USE_YOLO = os.getenv("USE_YOLO", "true").lower() in ("1", "true", "yes")
# ---------- 單張圖 YOLO+亮度 ----------


def process_and_calc_brightness(image: np.ndarray, use_yolo: bool = USE_YOLO):
    """
    計算影像的平均亮度，
    根據 use_yolo 參數決定是否使用 YOLO 推論。

    參數:
        image (np.ndarray): 原始 BGR 或灰階影像。
        use_yolo (bool): 是否啟用 YOLO 模型裁切來計算各物件亮度。
                          False 時直接以全影像計算。

    回傳:
        crops (List[np.ndarray]): 用於顯示的裁切圖 (若 use_yolo=False，則回傳原圖灰階版本)。
        text (str): 亮度計算結果文字。
    """
    # 檢查輸入影像
    if image is None:
        return [], "❗️請先上傳圖片"

    # 若不使用 YOLO，直接計算整張影像亮度
    if not use_yolo:
        # 彩色轉灰階
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        avg = float(np.mean(gray))
        text = f"🔹 平均亮度 (全影像): {avg:.2f}"
        return [gray], text

    # 以下為 use_yolo=True 時的流程
    # 確認模型檔案存在
    if not os.path.isfile(MODEL_PATH):
        return [], f"❗️找不到模型: {MODEL_PATH}"

    crops, brightness_list = [], []
    with tempfile.TemporaryDirectory() as upload_dir, tempfile.TemporaryDirectory() as crop_dir:
        # 儲存輸入影像
        in_path = os.path.join(upload_dir, "input.png")
        bgr = image[:, :, ::-1] if image.ndim == 3 else image
        cv2.imwrite(in_path, bgr)

        # YOLO 推論並裁切
        yolo = YOLOImageProcessor(MODEL_PATH, upload_dir, crop_dir)
        yolo.process_images()

        # 讀取裁切結果並計算亮度
        for fn in sorted(os.listdir(crop_dir)):
            fp = os.path.join(crop_dir, fn)
            crop = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if crop is None:
                continue
            crops.append(crop)
            brightness_list.append(float(np.mean(crop)))

    # 若未偵測到任何物件
    if not brightness_list:
        return [], "ℹ️ 未偵測到任何物件"

    # 計算並組合結果文字
    avg = sum(brightness_list) / len(brightness_list)
    lines = [f"{i+1}. {v:.2f}" for i, v in enumerate(brightness_list)]
    text = (
        "🔹 各物件平均亮度:\n" +
        "\n".join(lines) +
        f"\n\n⭐️ 整體平均: {avg:.2f}"
    )
    return crops, text

# ---------- 批量處理 Zip 圖片 ----------


import os
import tempfile
import zipfile
import subprocess
import shutil

def extract_zip_images(zip_file):
    """
    解壓 zip（支援 deflate 以外壓縮方法），
    捕捉 Errno 36: File name too long，只保留到第一組 WRGB (第一個 B\d+) 的部分。
    回傳所有圖片檔路徑。
    zip_file: gr.File 返回物件，.name 屬性是本機暫存檔案路徑
    """
    tmp = tempfile.mkdtemp()
    imgs = []
    try:
        with zipfile.ZipFile(zip_file.name, 'r') as zf:
            for info in zf.infolist():
                filename = os.path.basename(info.filename)
                low = filename.lower()
                if "photometricstereo" in low:
                    continue
                if not low.endswith(('.png','jpg','jpeg','bmp','tiff')):
                    continue

                try:
                    # 正常解壓
                    zf.extract(info, tmp)
                    imgs.append(os.path.join(tmp, info.filename))
                except OSError as e:
                    if e.errno == 36:
                        # 只保留到第一個 B\d+（即第一組 WRGB 資料）
                        name, ext = os.path.splitext(filename)
                        m = re.match(r"(.+?B\d+)", name)
                        clipped = m.group(1) if m else name
                        safe_name = f"{clipped}{ext}"
                        target = os.path.join(tmp, safe_name)

                        # 手動寫檔
                        with zf.open(info) as src, open(target, 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                        imgs.append(target)
                    else:
                        raise
    except NotImplementedError:
        # fallback 到系統 unzip，不做再剪裁了
        subprocess.run(
            ["unzip", "-o", zip_file.name, "-d", tmp],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    # 若上面沒任何檔案，才去遍歷一次
    if not imgs:
        for root, _, files in os.walk(tmp):
            for f in files:
                low = f.lower()
                if "photometricstereo" in low:
                    continue
                if low.endswith(('.png','jpg','jpeg','bmp','tiff')):
                    imgs.append(os.path.join(root, f))

    return imgs


def run_yolo_on_folder(image_paths):
    """若 USE_YOLO=False，直接 copy 原圖回傳資料夾；否則跑 YOLO 裁切。"""
    if not USE_YOLO:
        tmp = tempfile.mkdtemp()
        for p in image_paths:
            dst = os.path.join(tmp, os.path.basename(p))
            shutil.copy(p, dst)
        return tmp

    # below: 原本的 YOLO 流程
    u = tempfile.mkdtemp()
    c = tempfile.mkdtemp()
    # 複製原圖到 u
    for p in image_paths:
        os.makedirs(
            os.path.dirname(
                os.path.join(
                    u,
                    os.path.relpath(p, os.path.commonpath(image_paths))
                )
            ),
            exist_ok=True
        )
        cv2.imwrite(
            os.path.join(
                u,
                os.path.relpath(p, os.path.commonpath(image_paths))
            ),
            cv2.imread(p)
        )
    yolo = YOLOImageProcessor(MODEL_PATH, u, c)
    yolo.process_images()
    return c


def calc_brightness_folder(crop_dir):
    """計算 crop_dir 所有裁切圖的亮度，回傳 dict{filename:brightness}"""
    res={}
    for fn in sorted(os.listdir(crop_dir)):
        fp=os.path.join(crop_dir,fn)
        img=cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        res[fn]=float(np.mean(img))
    return res


def parse_filename_params(fname):
    """從 filename 提取 W,R,G,B\d+，回傳 dict"""
    params={}
    for ch in ['W','R','G','B','H',"E"]:
        m=re.search(fr"{ch}(\d+)", fname)
        params[ch]=int(m.group(1)) if m else 0
    return params

def dataframe_from_records(records):
    df = pd.DataFrame(records)
    def parse(fname):
        vals = {}
        for ch in ['D','W','R','G','B','H','E']:
            m = re.search(fr"{ch}(\d+)", fname, re.IGNORECASE)
            vals[ch] = int(m.group(1)) if m else 0
        return vals

    # 對每個 filename 呼叫 parse，回傳 dict 做成新的 DataFrame
    params = pd.DataFrame([parse(f) for f in df['filename']])
    df_final = pd.concat([params, df['brightness'].rename('Br')], axis=1)
    out_csv = os.path.abspath('results.csv')
    df_final.to_csv(out_csv, index=False)
    return out_csv
    
def wrgb_regression(csv_file):
    """
    上傳 results.csv 之後：
    1) 先跑 degree=1；若 R² < 0.99 再跑 degree=2，擇優
    2) 依最終階數畫殘差圖 & 單通道擬合圖
    3) 回傳 [殘差圖, 擬合圖] 路徑 + 報告文字
    """
    if csv_file is None:
        return [], "❗️請先上傳 CSV 檔"

    # 建立回歸物件
    reg = BrightnessRegression(csv_file.name)

    # 一鍵自動擬合 + 畫圖（channel='auto' 會自行挑有值的 WRGB 通道）
    info = reg.auto_fit_and_plot(channel="auto")   # or channel="W" 若一定要 W

    # 報告文字
    report = (
        f"◆ 採用階數: {info['degree']}\n"
        f"◆ 全特徵訓練 R²: {info['r2']:.4f}"
    )

    # 回傳圖檔路徑（殘差圖, 擬合圖）與報告
    return [info["residual_png"], info["univariate_png"]], report

def main(zip_file):
    """
    主流程：解壓 zip -> YOLO 裁切 -> 計算亮度 -> 生成 CSV
    返回 CSV 檔案路徑
    """
    records = []
    imgs = extract_zip_images(zip_file)
    crop_dir = run_yolo_on_folder(imgs)
    brightness_map = calc_brightness_folder(crop_dir)
    for fn, bri in brightness_map.items():
        params = parse_filename_params(fn)
        rec = {'filename': fn, 'brightness': bri}
        rec.update(params)
        records.append(rec)
    return dataframe_from_records(records)

# ---------- Gradio UI ----------
with gr.Blocks(title="亮度分析與迴歸分析") as demo:
    with gr.Tabs():
        with gr.TabItem("單張圖片處理"):
            img_in=gr.Image(type="numpy")
            gallery=gr.Gallery()
            txt=gr.Textbox()
            img_in.upload(process_and_calc_brightness, img_in, [gallery, txt])
        with gr.TabItem("zip檔案處理"):
            zip_in=gr.File(file_types=['.zip','.zup'], label="📦 上傳壓縮檔 (ZIP/ZUP)")
            btn=gr.Button("處理 ZIP 並下載 CSV")
            out=gr.File(label="⬇️ 下載 results.csv")
            btn.click(main, zip_in, out)
        with gr.TabItem("WRGB迴歸分析"):
            csv_in = gr.File(file_types=['.csv'], label="📄 上傳 results.csv")
            run_btn = gr.Button("執行 WRGB 迴歸分析")
            gallery = gr.Gallery(label="殘差圖 & 擬合圖", columns=2, height="auto")
            report  = gr.Textbox(label="分析報告", lines=3)
            run_btn.click(
                fn=wrgb_regression,
                inputs=[csv_in],
                outputs=[gallery, report]
            )
if __name__ == "__main__":
    # 在此修改模型路徑
    MODEL_PATH = "/app/best.pt"
    demo.launch(server_name="0.0.0.0", server_port=8800)