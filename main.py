# ==================== all_in_one_app.py ====================
import gradio as gr
# --- 共用 import（把三支腳本原本用到的集中放這） ------------
import os, cv2, json, re, shutil, zipfile, subprocess, tempfile
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
# ----------------------------------------------------------
from evaluation_app      import (
    store_uploaded_files,
    process_uploaded_images_with_yolo,
    quality_analysis_on_cropped,
    show_flagged_data,
    display_selected_image,
    get_top_image,
)
from brightness_calculator_app import (
    process_and_calc_brightness, main, wrgb_regression
)
from json_generate_app import (
    create_single_color_json, create_json,
    create_division_color_json, generate_sampling_json,create_hsv2rgb_json,
    MODELS, KEY_MAP             # ← 兩個常用常數
)




# ----------------------------------------------------------

YOLO_MODEL_PATH = "/app/best.pt"   # ⬅️ 統一放這

# ===========================================================
# ★★★ ① 成像品質評分頁（原 evaluation_app） ★★★
# ===========================================================
def build_evaluation_tab():
    # ------------ 把 evaluation_app 內的「工具函式」搬進來 ------------
    # (store_uploaded_files、process_uploaded_images_with_yolo … etc.)
    # ＝完全複製，略
    # ----------------------------------------------------------------

    def analyze_input(uploaded):
        # 與舊 evaluate_app 相同（記得用上面的工具函式）
        base_dir   = Path(".gradio/flagged")
        upload_dir = base_dir / "uploaded_image"
        crop_dir   = base_dir / "crop_image"
        csv_path   = base_dir / "dataset1.csv"
        msgs=[]
        try:
            msgs.append(store_uploaded_files(uploaded, upload_dir))
            msgs.append(process_uploaded_images_with_yolo(upload_dir, crop_dir, YOLO_MODEL_PATH))
            msgs.append(quality_analysis_on_cropped(crop_dir, csv_path))
        except Exception as e:
            msgs.append(f"❌ 錯誤：{e}")
        return "\n".join(msgs)

    with gr.TabItem("成像品質評分"):                    # ← 左側分頁名
        with gr.Row():
            with gr.Column():
                file_in  = gr.File(file_types=["image", ".zip"], label="上傳圖片或 ZIP")
                run_btn  = gr.Button("分析")
                out_txt  = gr.Textbox(lines=8, label="執行結果")
            with gr.Column():
                hist_btn = gr.Button("顯示前五名")
                df_out   = gr.Dataframe(
                    headers=["Image","Sharpness","Exposure","Contrast","Uniformity","Total Quality","Timestamp"],
                    interactive=True)
                img_prev = gr.Image(label="預覽")
                # 五顆查看按鈕
                view_btns=[gr.Button(f"查看第{i+1}名") for i in range(5)]

        # --- 綁定 callback ---
        run_btn.click(analyze_input, file_in, out_txt)
        hist_btn.click(show_flagged_data, None, df_out)
        for i,b in enumerate(view_btns):
            b.click(lambda idx=i: display_selected_image(get_top_image(idx)), None, img_prev)

# ===========================================================
# ★★★ ② 亮度計算 / WRGB 迴歸頁（原 brightness_calculator_app） ★★★
# ===========================================================
def build_brightness_tab():
    # → 直接使用原本 brightness_calculator_app 中的函式
    with gr.TabItem("亮度分析 / 迴歸"):
        with gr.Tabs():
            # ------- 單圖模式 -------
            with gr.TabItem("單張"):
                img_in  = gr.Image(type="numpy")
                gallery = gr.Gallery()
                txt_out = gr.Textbox()
                img_in.upload(process_and_calc_brightness, img_in, [gallery, txt_out])

            # ------- ZIP 批量 -------
            with gr.TabItem("ZIP 批量"):
                zip_in  = gr.File(file_types=['.zip','.zup'])
                btn_csv = gr.Button("處理並下載 CSV")
                file_csv= gr.File()
                btn_csv.click(main, zip_in, file_csv)

            # ------- WRGB 迴歸 -------
            with gr.TabItem("WRGB 迴歸"):
                csv_in  = gr.File(file_types=['.csv'])
                run_reg = gr.Button("執行迴歸")
                gal     = gr.Gallery(columns=2)
                report  = gr.Textbox(lines=4)
                run_reg.click(wrgb_regression, csv_in, [gal, report])

# ===========================================================
# ★★★ ③ JSON 產生器頁（原 json_generate_app） ★★★
# ===========================================================
def build_json_tab():
    with gr.TabItem("JSON 產生工具"):
        with gr.Tabs():
            # ========= 3-1 單色光 =========
            with gr.TabItem("單色光"):
                sk      = gr.File(label="骨架 JSON")
                model   = gr.Dropdown(MODELS, label="模型")
                ch      = gr.Dropdown(['W','R','G','B'], label="Channel")
                btn     = gr.Button("生成")
                file_out= gr.File()
                btn.click(create_single_color_json, [sk, model, ch], file_out)

            # ========= 3-2 自動矩陣 =========
            with gr.TabItem("矩陣生成"):
                sk      = gr.File(label="骨架 JSON")
                b_cnt   = gr.Dropdown([1,2,3], value=2, label="亮度數")
                c_cnt   = gr.Dropdown([1,2,3], value=3, label="色強度數")
                # 5 個模型各自的 textbox（亮度 / 顏色）
                b_boxes, c_boxes = [], []
                for m in MODELS:
                    with gr.TabItem(m):
                        b   = gr.Textbox(value="1024,512", label="亮度清單")
                        c   = gr.Textbox(value="1000,500,0", label="顏色清單")
                        b_boxes.append(b); c_boxes.append(c)
                btn_gen = gr.Button("生成 JSON")
                f_out   = gr.File()
                inter   = [v for pair in zip(b_boxes,c_boxes) for v in pair]  # interleave
                btn_gen.click(create_json, [sk, b_cnt, c_cnt]+inter, f_out)

            # ========= 3-3 等分 =========
            with gr.TabItem("等分組合"):
                sk  = gr.File(label="骨架 JSON")
                md  = gr.Dropdown(MODELS, label="模型")
                wmx = gr.Number(label="Wmax"); rmx = gr.Number(label="Rmax")
                gmx = gr.Number(label="Gmax"); bmx = gr.Number(label="Bmax")
                div = gr.Number(value=1, precision=0, label="等分")
                btn = gr.Button("生成")
                f   = gr.File()
                btn.click(create_division_color_json,
                          [sk,md,wmx,rmx,gmx,bmx,div], f)

            # ========= 3-4 抽樣 =========
            with gr.TabItem("WRGB 抽樣"):
                sk  = gr.File(label="骨架 JSON")
                md  = gr.Dropdown(MODELS, label="模型")
                with gr.Row():
                    wcsv, rcsv, gcsv, bcsv = [gr.File(file_types=['.csv']) for _ in range(4)]
                with gr.Row():
                    wmax = gr.Number(1024, precision=0, label="Wmax")
                    rmax = gr.Number(1024, precision=0, label="Rmax")
                    gmax = gr.Number(1024, precision=0, label="Gmax")
                    bmax = gr.Number(1024, precision=0, label="Bmax")
                with gr.Row():
                    brmin= gr.Number(0, label="Br_min"); brmax=gr.Number(1024, label="Br_max")
                samp = gr.Number(200, precision=0, label="樣本數")
                btn  = gr.Button("生成抽樣 JSON")
                fout = gr.File(); func_txt = gr.Textbox(lines=2)
                btn.click(generate_sampling_json,
                          [sk, md, wcsv, rcsv, gcsv, bcsv,
                           wmax,rmax,gmax,bmax, brmin, brmax, samp],
                          [fout, func_txt])
                
            # ========= 3-5 HSV2RGB =========
            with gr.TabItem("HSV2RGB 自動打光"):
                sk_hsv        = gr.File(label="📄 上傳骨架 JSON")
                model_hsv     = gr.Dropdown(MODELS, label="選擇模型")
                divisions_hsv = gr.Number(label="等分數量 (N)", value=8, precision=0)
                V_fixed = gr.Number(label="亮度  V (上限100)*", value=50.0, precision=1)
                S_repr = gr.Number(label="飽和度 S(上限100)*", value=100.0, precision=1)
            

                gen_hsv_btn   = gr.Button("生成 HSV2RGB JSON")
                gallery_hsv   = gr.Gallery(
                    label="10-bit 代表色表格 & HSV 彩虹盤",
                    columns=2, height="auto"
                )
                out_hsv_file  = gr.File(label="⬇️ 下載 JSON")

                gen_hsv_btn.click(
                    fn=create_hsv2rgb_json,                                       # 改成你的 HSV 版函式
                    inputs=[sk_hsv, model_hsv, divisions_hsv, V_fixed, S_repr],
                    outputs=[out_hsv_file, gallery_hsv]
                )

# ===========================================================
# ★★★ 把三個 Tab 組在一起 ★★★
# ===========================================================
with gr.Blocks(title="📦 All-in-One LED 工具箱") as demo:
    with gr.Tabs():          
        build_json_tab()                 # JSON 產生器
        build_brightness_tab()           # 亮度 / 迴歸
        build_evaluation_tab()           # 成像評分

# ---------------- 啟動 ----------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=9000, share=True)
