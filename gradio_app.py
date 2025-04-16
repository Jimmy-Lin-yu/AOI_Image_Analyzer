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
from yolo_model import YOLOImageProcessor
from datetime import datetime
# ------------------ 分析圖片 ------------------
import os
import cv2
import pandas as pd
import numpy as np
import zipfile
import tempfile
from datetime import datetime


# ------------------ ZIP 解壓支援中文檔名 ------------------
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

# ------------------ (1) 上傳檔案儲存 ------------------
def store_uploaded_files(uploaded, upload_folder):
    """
    根據上傳內容（單張圖片、ZIP 壓縮包或單個檔案），將圖片存入 upload_folder，
    並回傳處理結果訊息。
    """
    translator = FilenameTranslator()
    result = ""
    os.makedirs(upload_folder, exist_ok=True)
    
    # 處理單張圖像（來自 gr.Image，型別 np.ndarray）
    if isinstance(uploaded, np.ndarray):
        image_name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image_bgr = cv2.cvtColor(uploaded, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(upload_folder, image_name)
        cv2.imwrite(save_path, image_bgr)
        result += f"已儲存單張圖片：{image_name}\n"
    
    # 處理 ZIP 壓縮包
    elif hasattr(uploaded, "name") and uploaded.name.endswith(".zip"):
        result += "📦 解壓 ZIP 檔案中...\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_zip_preserve_chinese(uploaded.name, temp_dir)
            for root, _, files in os.walk(temp_dir):
                for fname in files:
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        # 刪除包含 "光度立体" 的檔案
                        if "光度立体" in fname:
                            try:
                                os.remove(os.path.join(root, fname))
                                result += f"{fname} → 包含 '光度立体' 已刪除\n"
                            except Exception as e:
                                result += f"{fname} → 刪除失敗: {e}\n"
                            continue
                        new_fname = translator.translate(fname)
                        src_path = os.path.join(root, fname)
                        dst_path = os.path.join(upload_folder, new_fname)
                        shutil.copy(src_path, dst_path)
                        result += f"已複製圖片：{new_fname}\n"
    
    # 處理單個圖片檔案 (gr.File)
    elif hasattr(uploaded, "name") and uploaded.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        if "光度立体" in uploaded.name:
            try:
                os.remove(uploaded.name)
                result += f"{uploaded.name} → 包含 '光度立体' 已刪除\n"
            except Exception as e:
                result += f"{uploaded.name} → 刪除失敗: {e}\n"
        else:
            new_fname = translator.translate(uploaded.name)
            dst_path = os.path.join(upload_folder, new_fname)
            shutil.copy(uploaded.name, dst_path)
            result += f"已複製單張圖片：{new_fname}\n"
    else:
        return "❌ 請上傳圖片、ZIP 或有效的圖片格式"
    
    return result


# ------------------ (2) YOLO 模型處理 ------------------
def process_uploaded_images_with_yolo(upload_folder, crop_folder, yolo_model_path):
    """
    呼叫 YOLO 模型處理 upload_folder 中的所有圖片，
    將裁切後的圖片存入 crop_folder，
    檔名格式若未符合「原檔名_crop.jpg」則調整。
    """
    os.makedirs(crop_folder, exist_ok=True)
    
    yolo_processor = YOLOImageProcessor(yolo_model_path, upload_folder, crop_folder)
    yolo_processor.process_images()
    result = f"✅ YOLO 處理完成，已將裁切後圖片存入 {crop_folder}\n"
    return result

# ------------------ (3) 裁切後圖像品質分析與 CSV 更新 ------------------
def quality_analysis_on_cropped(crop_folder, csv_path):
    """
    針對 crop_folder 中的裁切後圖片進行影像品質分析，
    若檔名不符合 'filename_crop.jpg' 格式則重新命名，
    並將分析結果記錄到 CSV (csv_path)。
    """
    analyzer = ImageQualityAnalyzer()
    records = []
    result = ""
    
    for file in os.listdir(crop_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            base, ext = os.path.splitext(file)
            # 檢查檔名是否包含 _crop，若無則重新命名
            if not base.endswith("_crop"):
                new_name = f"{base}_crop.jpg"
                old_path = os.path.join(crop_folder, file)
                new_path = os.path.join(crop_folder, new_name)
                os.rename(old_path, new_path)
            else:
                new_name = file
                new_path = os.path.join(crop_folder, new_name)
            
            image = cv2.imread(new_path)
            if image is None:
                result += f"{new_name} → ⚠️ 無法讀取進行品質分析\n"
                continue
            
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
                "uploaded_image": new_path,
                "output": output_text
            })
            result += f"{new_name} → 分析完成\n"
    
    # 將分析結果寫入 CSV，若 CSV 存在則避免重複記錄
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df_old = pd.read_csv(csv_path)
        existing = set(df_old["uploaded_image"].apply(os.path.basename).tolist())
        df_new = pd.DataFrame(records)
        df_new = df_new[~df_new["uploaded_image"].apply(os.path.basename).isin(existing)]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = pd.DataFrame(records)
    df_all.to_csv(csv_path, index=False)
    result += f"✅ 品質分析完成，共 {len(records)} 筆資料記錄到 CSV\n"
    return result

# ------------------ 主流程：分析上傳資料 ------------------
def analyze_input(uploaded):
    """
    流程：
      1. 將上傳的圖片（或 ZIP）存入指定上傳資料夾 (.gradio/flagged/uploaded_image)
      2. 呼叫 YOLO 模型處理上傳資料夾中的所有圖片，
         將裁切後的圖片以「原檔名_crop.jpg」存到 .gradio/flagged/crop_image
      3. 針對裁切後圖片進行影像品質分析，並將結果記錄到 CSV（dataset1.csv）
    """
    base_flag_dir = os.path.join(".gradio", "flagged")
    upload_folder = os.path.join(base_flag_dir, "uploaded_image")
    crop_folder   = os.path.join(base_flag_dir, "crop_image")
    csv_path      = os.path.join(base_flag_dir, "dataset1.csv")
    
    # Step 1: 儲存上傳檔案
    msg_store = store_uploaded_files(uploaded, upload_folder)
    
    # Step 2: 使用 YOLO 處理上傳的圖片，注意 YOLO_MODEL_PATH 為全域變數
    msg_yolo = process_uploaded_images_with_yolo(upload_folder, crop_folder, YOLO_MODEL_PATH)
    
    # Step 3: 品質分析並更新 CSV
    msg_quality = quality_analysis_on_cropped(crop_folder, csv_path)
    
    return msg_store + msg_yolo + msg_quality


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

# ------------------ 預覽選定圖片 ------------------
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
    3. 刪除 .gradio/flagged/crop_image/ 內所有檔案
    """
    base_dir   = ".gradio/flagged"
    csv_path   = os.path.join(base_dir, "dataset1.csv")
    upload_dir  = os.path.join(base_dir, "uploaded_image")
    crop_dir = os.path.join(base_dir, "crop_image")

    try:
        # --- 1️⃣ 重設 CSV ---
        if os.path.exists(csv_path):
            pd.DataFrame(columns=["uploaded_image", "output"]).to_csv(csv_path, index=False)
        else:
            # 如果檔案不存在也無妨，直接當作已清空
            pass

        # --- 2️⃣ 刪除 uploaded_image 內所有檔案 ---
        if os.path.isdir(upload_dir):
            for root, _, files in os.walk(upload_dir):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception as e:
                        print("⚠️ 無法刪除", f, "：", e)

        # --- 3️⃣ 刪除 crop_image 內所有檔案 ---
        if os.path.isdir(crop_dir):
            for root, _, files in os.walk(crop_dir):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception as e:
                        print("⚠️ 無法刪除", f, "：", e)

        return "✅ 清除成功：CSV 已重設，uploaded_image 與 crop_image 內檔案已刪除"
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
    YOLO_MODEL_PATH = "/app/best.pt" 
    demo.launch(server_name="0.0.0.0", server_port=7860)
