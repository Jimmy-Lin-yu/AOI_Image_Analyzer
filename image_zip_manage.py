# image_zip_manager.py

import os
import zipfile
import shutil
import tempfile
from datetime import datetime

class ImageZipManager:
    """
    專責圖片資料夾 ↔ ZIP 壓縮／解壓縮，
    並同時執行：
      - 中文檔名重解碼 (cp437→gbk)
      - 刪除含特定關鍵字檔案 (預設"光度立体")
      - 檔名翻譯 (內建中→英對照表)
    """
    IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    def __init__(self,
                 from_enc: str = 'cp437',
                 to_enc:   str = 'gbk',
                 bad_keyword: str = "光度立体"):
        # 解壓時的編碼設定與壞檔關鍵字
        self.from_enc = from_enc
        self.to_enc   = to_enc
        self.bad_kw   = bad_keyword

        # 檔名翻譯對照表
        self.translation_map = {
            "全色域环形100": "full_color_ring_100",
            "全色域条形30100": "full_color_bar_30100",
            "全色域同轴60":   "full_color_coaxial_60",
            "4层4角度全圆88":  "4_layer_4_angle_full_circle_88",
            "全色域圆顶":      "full_color_dome",
            "亮度":           "brightness",
            "白":             "W",
            "红":             "R",
            "绿":             "G",
            "蓝":             "B",
            "高":             "height",
            "曝光":           "exposure"
        }

    @staticmethod
    def _is_image(fname: str) -> bool:
        return fname.lower().endswith(ImageZipManager.IMAGE_EXTS)

    def translate(self, filename: str) -> str:
        """
        將 filename 中的中文關鍵字依照 translation_map 換成英文。
        """
        out = filename
        for zh, en in self.translation_map.items():
            out = out.replace(zh, en)
        return out

    def compress_images(self, src_dir: str, zip_path: str | None = None) -> str:
        """
        將 src_dir 下所有影像檔壓成 ZIP，回傳 ZIP 檔路徑。
        zip_path 省略時預設放在 src_dir 同層，檔名為 {資料夾名稱}.zip。
        """
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(f"找不到資料夾：{src_dir}")

        if zip_path is None:
            parent, name = os.path.split(os.path.abspath(src_dir))
            zip_path = os.path.join(parent, f"{name}.zip")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(src_dir):
                for f in files:
                    if self._is_image(f):
                        absf = os.path.join(root, f)
                        arc  = os.path.relpath(absf, start=src_dir)
                        zf.write(absf, arc)
        return zip_path

    def decompress_images(self, zip_path: str, dest_dir: str) -> list[str]:
        """
        解壓 zip_path 到 dest_dir，只執行：
          1. cp437→gbk 重解碼
          2. 解壓所有檔案到 dest_dir
        回傳所有解壓後檔案（包含非影像）的絕對路徑清單。
        """
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"找不到 ZIP 檔：{zip_path}")

        os.makedirs(dest_dir, exist_ok=True)
        extracted = []

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for zinfo in zf.infolist():
                # 先嘗試改名（cp437→gbk）
                try:
                    zinfo.filename = zinfo.filename.encode(self.from_enc).decode(self.to_enc)
                except:
                    pass

                # 解壓到目標資料夾
                zf.extract(zinfo, dest_dir)

                # 收集完整路徑
                extracted_path = os.path.join(dest_dir, zinfo.filename)
                extracted.append(extracted_path)

        return extracted


if __name__ == "__main__":
    # 範例：壓縮與解壓測試
    mgr = ImageZipManager()
    folder = r"C:\Users\Jimmy\Pictures\test_images"
    zipf   = mgr.compress_images(folder)
    print(f"已壓縮到：{zipf}")

    out_dir = r"C:\Users\Jimmy\Pictures\unpack_out"
    saved = mgr.decompress_images(zipf, out_dir)
    print("解壓並處理結果：")
    for p in saved:
        print("  ", p)
