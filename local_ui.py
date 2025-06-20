import gradio as gr
import requests
import pandas as pd
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from image_zip_manage import ImageZipManager

# ======== 設定區 ========
REMOTE_URL    = "http://192.168.0.5:7960/infer_zip"
LOCAL_FOLDER  = r"D:\Users\JimmyLin\Desktop\imgSaves"
TEMP_ZIP_PATH = Path("temp_upload.zip")
# =========================

def infer_folder():
    mgr = ImageZipManager()
    # 1) 壓縮
    try:
        # compress_images 回傳 str，所以包成 Path
        zip_path = Path(mgr.compress_images(LOCAL_FOLDER, zip_path=str(TEMP_ZIP_PATH)))
    except Exception as e:
        return pd.DataFrame({"Error":[f"壓縮失敗：{e}"]}), []

    # 2) 上傳
    try:
        with open(zip_path, "rb") as f:
            resp = requests.post(
                REMOTE_URL,
                files={"file": (zip_path.name, f, "application/zip")},
                timeout=600
            )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return pd.DataFrame({"Error":[f"呼叫失敗：{e}"]}), []

    # 3) 解析所有結果
    all_rows = []
    img_list = []
    for item in data.get("results", []):
        all_rows.append({
            "Image":         item["filename"],
            "Sharpness":     item["sharpness"],
            "Exposure":      item["exposure"],
            "Contrast":      item["contrast"],
            "Uniformity":    item["uniformity"],
            "Total Quality": item["total_quality"],
        })
        img_list.append((item["total_quality"], item["cropped_image"]))

    # 4) 建全表並排序
    df_all = pd.DataFrame(all_rows)
    df_all = df_all.sort_values("Total Quality", ascending=False).reset_index(drop=True)

    # 5) 處理前五張圖，右上角寫上 No.X 紅色字
    top5 = sorted(img_list, key=lambda x: x[0], reverse=True)[:5]
    gallery_imgs = []
    for rank, (_, b64str) in enumerate(top5, start=1):
        img_data = base64.b64decode(b64str)
        pil_img  = Image.open(BytesIO(img_data)).convert("RGB")
        np_img   = np.array(pil_img)

        # 準備文字
        text = f"No.{rank}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        color = (255, 0, 0)  # BGR 紅色

        # 計算文字大小
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        # 右上角位置 (留 10px 邊距)
        x = np_img.shape[1] - text_w - 10
        y = text_h + 10

        # 寫上紅色文字
        cv2.putText(np_img, text, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)
        gallery_imgs.append(Image.fromarray(np_img))

    return df_all, gallery_imgs

# ------------------ Gradio UI ------------------
with gr.Blocks(title="足粒成像自動評分系統") as demo:
    gr.Markdown(f"### 📂  固定資料夾：`{LOCAL_FOLDER}`")
    run_btn = gr.Button("開始推論")

    with gr.Row():
        df_out  = gr.Dataframe(
            headers=["Image","Sharpness","Exposure","Contrast","Uniformity","Total Quality"],
            label="全部結果",
            interactive=True
        )
        gallery = gr.Gallery(
            label="前五名裁切後影像",
            columns=5,           # 一排五
            elem_id="top5_gallery"
        )

    # run_btn 會一次把完整 DataFrame 傳給 df_out，
    # 再把前五張裁切圖傳給 gallery
    run_btn.click(infer_folder, outputs=[df_out, gallery])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=5000, share=True)
