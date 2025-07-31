# remote_api.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import base64
import cv2
import os
import pandas as pd
import uuid
import numpy as np
# 如果你的檔名是 image_zip_manager.py，就這樣 import：
from image_zip_manage import ImageZipManager  
# from yolo_model import YOLOImageProcessor #暫時停止使用
from u2_net_inference import U2NetProcessor
from image_evaluation import ImageQualityAnalyzer

# ───────── 基本設定 ──────────────────────────────
WORK_ROOT = Path("/tmp/aoi_runs")               # 所有暫存／歷史資料夾
WORK_ROOT.mkdir(exist_ok=True, parents=True)

# U²-Net 處理器，全域可重用
u2net = U2NetProcessor(model_path="/app/u2net.pth")

app = Flask(__name__)
# ────────────────────────────────────────────────

@app.route("/infer_zip", methods=["POST"])
def infer_zip():
    # 1) 檢查上傳
    if "file" not in request.files:
        return jsonify({"error": "No file field"}), 400
    zip_file = request.files["file"]
    if zip_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # 2) 建 run 目錄
    run_id    = uuid.uuid4().hex[:8]
    run_dir   = WORK_ROOT / run_id
    upload_dir= run_dir / "uploaded"
    crop_dir  = run_dir / "crop"
    upload_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    # 3) 存 ZIP
    zip_path = run_dir / secure_filename(zip_file.filename)
    zip_file.save(zip_path)

    # 4) 解壓＋翻譯＋過濾
    mgr = ImageZipManager()
    extracted = mgr.decompress_images(str(zip_path), str(upload_dir))
    for path_str in extracted:
        p = Path(path_str)
        if mgr.bad_kw in p.name:
            p.unlink(missing_ok=True)
            continue
        new_name = mgr.translate(p.name)
        if new_name != p.name:
            p.rename(p.with_name(new_name))

    # 5) YOLO 裁切
    # # yolo = YOLOImageProcessor( str(upload_dir),str(crop_dir))
    # yolo.process_images()

    # 5) U2net裁切
    u2net.crop_images(str(upload_dir), str(crop_dir))


    # 6) 品質分析
    analyzer = ImageQualityAnalyzer(str(crop_dir))
    results = analyzer.process_auto_evaluate()

    
    # 7) 寫／併歷史 CSV（可選）
    csv_path = WORK_ROOT / "auto_dataset.csv"
    df_new   = pd.DataFrame(results)
    return jsonify({"results": df_new.to_dict(orient="records")})
    
@app.route("/status", methods=["GET"])
def status():
    """健康檢查端點"""
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7960, debug=False)
