import gradio as gr
import cv2
import pandas as pd
import os
import datetime
import shutil
import zipfile
import tempfile

from image_evaluation import ImageQualityAnalyzer
from word_change import FilenameTranslator

# ------------------ 分析圖片 ------------------
import os
import cv2
import pandas as pd
import numpy as np
import zipfile
import tempfile
from datetime import datetime
from image_evaluation import ImageQualityAnalyzer  # 確保這行路徑正確！如果你的 class 實際在 app.py

def extract_zip_preserve_chinese(zip_file_path, dest_dir, from_enc='cp437', to_enc='gbk'):
    """
    Extracts a ZIP file while re-decoding filenames from `from_enc` to `to_enc`.
    Adjust encodings if your ZIP uses a different scheme.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for zf_info in zip_ref.infolist():
            try:
                # Re-decode the filename: first get the bytes as interpreted in cp437,
                # then decode them using the expected Chinese encoding (gbk).
                decoded_filename = zf_info.filename.encode(from_enc).decode(to_enc)
            except Exception as e:
                # In case of failure, fall back to original filename
                decoded_filename = zf_info.filename
            # Update the in-memory filename before extraction
            zf_info.filename = decoded_filename
            zip_ref.extract(zf_info, dest_dir)

def analyze_input(uploaded):
    csv_path = os.path.join(".gradio", "flagged", "dataset1.csv")
    preview_folder = os.path.join(".gradio", "flagged", "uploaded_image")
    os.makedirs(preview_folder, exist_ok=True)
    analyzer = ImageQualityAnalyzer()
    records = []
    result = ""

    def process_image(image, original_path):
        image_name = os.path.basename(original_path)
        save_path = os.path.join(preview_folder, image_name)
        cv2.imwrite(save_path, image)

        metrics = {
            "sharpness": analyzer.calculate_sharpness(image),
            "exposure": analyzer.calculate_exposure(image),
            "contrast": analyzer.calculate_contrast(image),
            "uniformity": analyzer.calculate_light_uniformity(image),
            "noise": analyzer.calculate_noise(image)
        }
        scores, total = ImageQualityAnalyzer.evaluate_quality(metrics)

        output_text = "影像品質分析結果：\n"
        for (k, v), score in zip(metrics.items(), scores):
            output_text += f"  {k.capitalize()}: {v:.2f} → Score {score}\n"
        output_text += f"\nTotal Quality: {total:.1f}%"

        records.append({
            "uploaded_image": save_path,
            "output": output_text
        })
        return output_text
    
    translator = FilenameTranslator()

    # ✅ 單張圖片 (來自 Gr.Image)
    if isinstance(uploaded, np.ndarray):
        image_name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image_bgr = cv2.cvtColor(uploaded, cv2.COLOR_RGB2BGR)
        result = process_image(image_bgr, image_name)

    # ✅ ZIP 檔案

    elif hasattr(uploaded, "name") and uploaded.name.endswith(".zip"):
        result = "📦 ZIP 解壓分析結果：\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the ZIP while preserving the complete Chinese filenames
            extract_zip_preserve_chinese(uploaded.name, temp_dir)
           #with zipfile.ZipFile(uploaded.name, 'r') as zip_ref:
           #     zip_ref.extractall(temp_dir)
            for root, _, files in os.walk(temp_dir):
                for fname in files:
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        full_path = os.path.join(root, fname)
                        # Translate filename

                        if "光度立体" in fname:
                            try:
                                os.remove(full_path)
                                result += f"{fname} → 包含 '光度立体' 已被刪除\n"
                            except Exception as e:
                                result += f"{fname} → ⚠️ 刪除失敗: {e}\n"
                            continue

                        # Translate filename
                        new_fname = translator.translate(fname)
                        image = cv2.imread(full_path)
                        if image is not None:
                            _ = process_image(image, new_fname)
                            result += f"{new_fname} → 分析完成\n"
                        else:
                            result += f"{new_fname} → ⚠️ 無法讀取\n"

    # ✅ 單張圖片檔案 (from gr.File)
    elif hasattr(uploaded, "name") and uploaded.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Check if the filename contains "光度立体" and delete if so
        if "光度立体" in uploaded.name:
            try:
                os.remove(uploaded.name)
                result = f"{uploaded.name} → 包含 '光度立体' 已被刪除"
            except Exception as e:
                result = f"{uploaded.name} → ⚠️ 刪除失敗: {e}"
        else:
            new_fname = translator.translate(uploaded.name)
            image = cv2.imread(uploaded.name)
            if image is not None:
                result = process_image(image, new_fname)
                records.append({"uploaded_image": new_fname})
            else:
                result = "⚠️ 無法讀取圖片內容"
    else:
        return "❌ 請上傳圖片、ZIP 或有效的圖片格式"


    df_new = pd.DataFrame(records)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df_old = pd.read_csv(csv_path)
        # Get the existing filenames by taking the basename from the uploaded_image column.
        existing_filenames = set(df_old["uploaded_image"].apply(os.path.basename).tolist())
        # Keep only those new entries that are not already in the CSV.
        df_new = df_new[~df_new["uploaded_image"].apply(os.path.basename).isin(existing_filenames)]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(csv_path, index=False)

    if df_new.empty:
        return result + "\n⚠️ 所有圖片皆已存在，未新增資料。"
    else:
        return result + f"\n✅ 新增 {len(df_new)} 筆資料到 CSV"


# ------------------ 顯示前五名 ------------------
def show_flagged_data():
    try:
        csv_path = os.path.join(".gradio", "flagged", "dataset1.csv")
        if not os.path.exists(csv_path):
            return pd.DataFrame(columns=["Image","Total Quality","Sharpness", "Exposure", "Contrast", "Uniformity", "Noise", "Timestamp"])

        df = pd.read_csv(csv_path)

        # 1. Fill empty output cells with empty string, then convert everything to string.
        df["output"] = df["output"].fillna("").astype(str)

        # 用正則表達式解析數值
        import re

        def extract_metric(text, metric_name):
            pattern = rf"{metric_name}:\s*([0-9.]+)"
            match = re.search(pattern, text)
            return float(match.group(1)) if match else None

        def extract_quality(text):
            match = re.search(r"Total Quality: ([0-9.]+)%", text)
            return float(match.group(1)) if match else None

        df["Total Quality"] = df["output"].apply(extract_quality)
        df["Sharpness"] = df["output"].apply(lambda x: extract_metric(x, "Sharpness"))
        df["Exposure"] = df["output"].apply(lambda x: extract_metric(x, "Exposure"))
        df["Contrast"] = df["output"].apply(lambda x: extract_metric(x, "Contrast"))
        df["Uniformity"] = df["output"].apply(lambda x: extract_metric(x, "Uniformity"))
        df["Noise"] = df["output"].apply(lambda x: extract_metric(x, "Noise"))

        df["Image"] = df["uploaded_image"].apply(os.path.basename)
        df["Timestamp"] = pd.to_datetime(df["uploaded_image"].apply(lambda f: datetime.fromtimestamp(os.path.getctime(f)) if os.path.exists(f) else pd.NaT))

        top5 = df.sort_values(by="Total Quality", ascending=False).head(5)

        return top5[["Image", "Sharpness", "Exposure", "Contrast", "Uniformity", "Noise", "Total Quality", "Timestamp"]]

    except Exception as e:
        return pd.DataFrame({"錯誤": [f"⚠️ 讀取失敗：{e}"]})


def display_selected_image(image_name):
    try:
        csv_path = os.path.join(".gradio", "flagged", "dataset1.csv")
        if not os.path.exists(csv_path):
            return None

        df = pd.read_csv(csv_path)

        # 生成 Image 欄位（避免 NaN）
        df["Image"] = df["uploaded_image"].apply(lambda x: os.path.basename(str(x)) if pd.notnull(x) else "")

        match = df[df["Image"] == image_name]

        if not match.empty:
            image_path = match.iloc[0]["uploaded_image"]
            if isinstance(image_path, str) and os.path.exists(image_path):
                return image_path
        return None

    except Exception as e:
        print("⚠️ 預覽錯誤：", e)
        return None



# ------------------ 備份與刪除 CSV 檔案 ------------------
def backup_csv():
    src = os.path.join(".gradio", "flagged", "dataset1.csv")
    dst = os.path.join("flagged", "backup_dataset.csv")
    if os.path.exists(src):
        os.makedirs("flagged", exist_ok=True)
        shutil.copy(src, dst)
        return f"✅ 備份成功，儲存為：{dst}"
    else:
        return "⚠️ 找不到 dataset1.csv"


def clear_images_csv():
    """
    1. 清空 .gradio/flagged/dataset1.csv
    2. 刪除 .gradio/flagged/uploaded_image/ 內所有檔案
    """
    base_dir   = ".gradio/flagged"
    csv_path   = os.path.join(base_dir, "dataset1.csv")
    image_dir  = os.path.join(base_dir, "uploaded_image")

    try:
        # --- 1️⃣ 重設 CSV ---
        if os.path.exists(csv_path):
            pd.DataFrame(columns=["uploaded_image", "output"]).to_csv(csv_path, index=False)
        else:
            # 如果檔案不存在也無妨，直接當作已清空
            pass

        # --- 2️⃣ 刪除已上傳圖片 ---
        if os.path.isdir(image_dir):
            # 逐檔刪除，保留資料夾結構（避免後續寫檔失敗）
            for root, _, files in os.walk(image_dir):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception as e:
                        print("⚠️ 無法刪除", f, "：", e)

        return "✅ 清除成功：CSV 已重設，uploaded_image 內檔案已刪除"
    except Exception as e:
        return f"❌ 清除失敗：{e}"



# ------------------ Gradio UI ------------------
with gr.Blocks() as demo:
    demo.flagging_dir = ".gradio/flagged"
    demo.allow_flagging = "manual"

    gr.Markdown("## 📸 Image Quality Analyzer")

    with gr.Row():
        with gr.Column():
            image_input = gr.File(file_types=["image", ".zip"], label="上傳圖片或 ZIP 壓縮資料夾")
            submit_btn = gr.Button("分析圖片")
            output_text = gr.Textbox(label="分析結果", lines=8)
        with gr.Column():
            history_btn = gr.Button("📊 顯示品質最佳前五名")
            df_output = gr.Dataframe(
                interactive=True,
                headers=["Image", "Sharpness", "Exposure", "Contrast", "Uniformity", "Noise", "Total Quality", "Timestamp"],
            )
            selected_image = gr.Image(label="🔍 預覽圖片")  # 預覽圖
            preview_buttons = [gr.Button(f"查看第{i+1}名圖片") for i in range(5)]  # 🔘 五個按鈕
            backup_btn = gr.Button("📁 備份 CSV")
            backup_result = gr.Textbox(label="備份結果")
            clear_btn = gr.Button("🗑️ 清除 dataset1.csv")
            clear_result = gr.Textbox(label="清除結果")

    submit_btn.click(analyze_input, inputs=image_input, outputs=output_text)
    history_btn.click(show_flagged_data, outputs=df_output)

    # 🔁 建立綁定，每個按鈕綁定一個前五名圖片名稱
    def get_top_image(index):
        try:
            csv_path = os.path.join(".gradio", "flagged", "dataset1.csv")
            df = pd.read_csv(csv_path)
            df["Image"] = df["uploaded_image"].apply(lambda x: os.path.basename(str(x)) if pd.notnull(x) else "")
            df["Total Quality"] = df["output"].str.extract(r"Total Quality: ([0-9.]+)%").astype(float)
            top5 = df.sort_values(by="Total Quality", ascending=False).head(5)
            return top5.iloc[index]["Image"] if index < len(top5) else None
        except:
            return None

    for i, btn in enumerate(preview_buttons):
        btn.click(
            lambda idx=i: display_selected_image(get_top_image(idx)),
            outputs=selected_image
        )

    backup_btn.click(backup_csv, outputs=backup_result)
    clear_btn.click(clear_images_csv, outputs=clear_result)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
