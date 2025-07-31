# evaluation_app.py
import gradio as gr
import cv2
import pandas as pd
import os
import datetime
import shutil
from pathlib import Path
from image_evaluation import ImageQualityAnalyzer
# from yolo_model import YOLOImageProcessor #暫時停止使用YOLO
from image_zip_manage import ImageZipManager
from u2net_cropper import U2NetCropper
from datetime import datetime
# ------------------ 分析圖片 ------------------
import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime


# ------------------ (1) 上傳檔案儲存 ------------------
def store_uploaded_files(uploaded, upload_folder):
    """
    根據上傳內容（單張圖片、ZIP 或單個檔案），
    將影像存到 upload_folder，並回傳處理結果訊息。
    ZIP 分支先呼叫 ImageZipManager.decompress_only() 只做解壓，
    再在這裡對每個檔案呼叫 mgr.translate() 並一次處理翻譯檔名。
    """
    mgr = ImageZipManager()
    result = ""
    os.makedirs(upload_folder, exist_ok=True)

    # 1a. 處理單張 np.ndarray
    if isinstance(uploaded, np.ndarray):
        image_name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image_bgr  = cv2.cvtColor(uploaded, cv2.COLOR_RGB2BGR)
        save_path  = os.path.join(upload_folder, image_name)
        cv2.imwrite(save_path, image_bgr)
        result += f"已儲存單張圖片：{image_name}\n"

    # 1b. 處理 ZIP
    elif hasattr(uploaded, "name") and uploaded.name.lower().endswith(".zip"):
        result += "📦 ZIP 解壓並翻譯中...\n"
        try:
            # 先只解壓，不翻譯
            extracted = mgr.decompress_images(uploaded.name, upload_folder)
            for src in extracted:
                base = os.path.basename(src)
                # 先略過 bad_keyword
                if mgr.bad_kw in base:
                    os.remove(src)
                    result += f"{base} → 含關鍵字已刪除\n"
                    continue
                # translate & rename
                new_base = mgr.translate(base)
                dst = os.path.join(upload_folder, new_base)
                os.rename(src, dst)
                result += f"翻譯檔名：{base} → {new_base}\n"
        except Exception as e:
            result += f"❌ ZIP 處理失敗：{e}\n"

    # 1c. 處理單檔影像
    elif hasattr(uploaded, "name") and \
         uploaded.name.lower().endswith((".png",".jpg",".jpeg",".bmp",".tiff")):
        base = os.path.basename(uploaded.name)
        if mgr.bad_kw in base:
            result += f"{base} → 含關鍵字已略過\n"
        else:
            new_base = mgr.translate(base)
            dst = os.path.join(upload_folder, new_base)
            try:
                shutil.copy(uploaded.name, dst)
                result += f"已儲存：{new_base}\n"
            except Exception as e:
                result += f"❌ 複製失敗：{e}\n"

    else:
        return "❌ 請上傳圖片、ZIP 或有效的影像檔"

    return result


# ------------------ (2) YOLO 模型處理 ------------------ #暫時停止使用YOLO
# def process_uploaded_images_with_yolo(upload_folder, crop_folder, yolo_model_path):
#     """
#     呼叫 YOLO 模型處理 upload_folder 中的所有圖片，
#     將裁切後的圖片存入 crop_folder，
#     檔名格式若未符合「原檔名_crop.jpg」則調整。
#     """
#     os.makedirs(crop_folder, exist_ok=True)
    
#     yolo_processor = YOLOImageProcessor(upload_folder, crop_folder)
#     yolo_processor.process_images()
#     result = f"✅ YOLO 處理完成，已將裁切後圖片存入 {crop_folder}\n"
#     return result


def process_uploaded_images_with_u2net(upload_folder, crop_folder, model_path):
    os.makedirs(crop_folder, exist_ok=True)
    cropper = U2NetCropper(model_path=model_path)
    saved = cropper.crop_images(str(upload_folder), str(crop_folder))
    return f"✅ U²-Net 切割完成，共 {len(saved)} 張圖存入 {crop_folder}"

# ------------------ (3) 裁切後圖像品質分析與 CSV 更新 ------------------
def quality_analysis_on_cropped(crop_folder: Path, csv_path: Path) -> str:
    """
    針對 crop_folder 中的裁切後圖片進行影像品質分析，
    若檔名不符合 'filename_crop.jpg' 格式則重新命名，
    並將分析結果記錄到 CSV (csv_path)。
    """
    analyzer = ImageQualityAnalyzer(str(crop_folder))
    records = []
    result = ""
    

    # ---- 先準備舊的 DataFrame，避免 KeyError ----
    if csv_path.exists() and csv_path.stat().st_size > 0:
        df_old = pd.read_csv(csv_path)
        if "uploaded_image" not in df_old.columns:
            # 如果缺欄，就清掉重新開始
            df_old = pd.DataFrame(columns=["uploaded_image", "output"])
    else:
        # CSV 不存在或為空，就建立空表
        df_old = pd.DataFrame(columns=["uploaded_image", "output"])


    for file in os.listdir(crop_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg"," .bmp", ".tiff")):
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
                # "defect": analyzer.calculate_defect_score(image)
            }
            scores, total = analyzer.evaluate_quality(metrics)
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


# ------------------ 顯示前五名 ------------------
def show_flagged_data():
    try:
        csv_path = os.path.join(".gradio", "flagged", "dataset1.csv")
        if not os.path.exists(csv_path):
            return pd.DataFrame(columns=["Image","Total Quality","Sharpness", "Exposure", "Contrast", "Uniformity", "Timestamp"])

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
        # df["Defect"] = df["output"].apply(lambda x: extract_metric(x, "Defect"))

        df["Image"] = df["uploaded_image"].apply(os.path.basename)
        df["Timestamp"] = pd.to_datetime(df["uploaded_image"].apply(lambda f: datetime.fromtimestamp(os.path.getctime(f)) if os.path.exists(f) else pd.NaT))

        top5 = df.sort_values(by="Total Quality", ascending=False).head(5)
        print("⚠️ 讀取成功：", top5)
        return top5[["Image", "Sharpness", "Exposure", "Contrast", "Uniformity", "Total Quality", "Timestamp"]]

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

# ------------------ 每個按鈕綁定一個前五名圖片名稱 ------------------
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




# ------------------ 主流程：分析上傳資料 ------------------

#頁面1:成像品質評分
def analyze_input(uploaded):
    """
    Gradio〈成像品質評分〉頁面的主流程：
      1. 儲存上傳檔 (ZIP / 單圖) 至 .gradio/flagged/uploaded_image
      2. 以 YOLO 產生裁切圖 → .gradio/flagged/crop_image
      3. 對裁切圖做品質評分並寫入 dataset1.csv
    """
    base_dir        = Path(".gradio/flagged")
    upload_dir      = base_dir / "uploaded_image"
    crop_dir        = base_dir / "crop_image"
    csv_path        = base_dir / "dataset1.csv"
    messages: list[str] = []                      # 收集各步驟回傳文字

    def _append(text):                            # 小工具：避免重複寫 "+="
        if text:
            messages.append(text.strip())

    # ── STEP-1 儲存檔案 ──────────────────────────
    try:
        _append(store_uploaded_files(uploaded, upload_dir))
    except Exception as e:
        _append(f"❌ 儲存檔案失敗：{e}")

    # ── STEP-2 YOLO 裁切 ────────────────────────
    # try:
    #     _append(process_uploaded_images_with_yolo(upload_dir, crop_dir, YOLO_MODEL_PATH))
    # except Exception as e:
    #     _append(f"❌ YOLO 處理失敗：{e}")

    try:
         _append(process_uploaded_images_with_u2net(upload_dir, crop_dir, U2net_MODEL_PATH))
    except Exception as e:
         _append(f"❌ YOLO 處理失敗：{e}")

    # ── STEP-3 品質評分 ─────────────────────────
    try:
        _append(quality_analysis_on_cropped(crop_dir, csv_path))
    except Exception as e:
        _append(f"❌ 品質評分失敗：{e}")

    # 最終回傳
    return "\n".join(messages) if messages else "⚠️ 無任何結果"


# ------------------ Gradio UI ------------------
with gr.Blocks(title="成像品質評分系統") as demo:
    demo.flagging_dir = ".gradio/flagged"
    demo.allow_flagging = "manual"

    gr.Markdown("## 📸 Image Quality Analyzer")
    with gr.TabItem("成像品質評分"):
        with gr.Row():
            with gr.Column():
                image_input = gr.File(file_types=["image", ".zip"], label="上傳圖片或 ZIP 壓縮資料夾")
                submit_btn = gr.Button("分析圖片")
                output_text = gr.Textbox(label="分析結果", lines=8)
            with gr.Column():
                history_btn = gr.Button("📊 顯示品質最佳前五名")
                df_output = gr.Dataframe(
                    interactive=True,
                    headers=["Image", "Sharpness", "Exposure", "Contrast", "Uniformity", "Total Quality", "Timestamp"],
                )
                selected_image = gr.Image(label="🔍 預覽圖片")  # 預覽圖
                preview_buttons = [gr.Button(f"查看第{i+1}名圖片") for i in range(5)]  # 🔘 五個按鈕
                backup_btn = gr.Button("📁 備份 CSV")
                backup_result = gr.Textbox(label="備份結果")
                clear_btn = gr.Button("🗑️ 清除 dataset1.csv")
                clear_result = gr.Textbox(label="清除結果")

        submit_btn.click(analyze_input, inputs=image_input, outputs=output_text)
        history_btn.click(show_flagged_data, outputs=df_output)

        for i, btn in enumerate(preview_buttons):
            btn.click(
                lambda idx=i: display_selected_image(get_top_image(idx)),
                outputs=selected_image
            )

        backup_btn.click(backup_csv, outputs=backup_result)
        clear_btn.click(clear_images_csv, outputs=clear_result)

        
if __name__ == "__main__":
    # YOLO_MODEL_PATH = "/app/best.pt" 
    U2net_MODEL_PATH = "/app/u2net.pth"
    demo.launch(server_name="0.0.0.0", server_port=7860)
