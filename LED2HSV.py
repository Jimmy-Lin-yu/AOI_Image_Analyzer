import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

class HSVHueDisc:
    """
    用 HSV 生成彩虹圓盤（中心白、邊緣彩虹），
    並標出等角代表色點。全程記憶體處理。
    底圖與輸出圖都存到 LED2HSV 資料夾。
    """

    def __init__(self,
                 n_divisions: int = 8,
                 radius: int = 110,
                 grid: float = 0.5,       # 建議改成 0.5 讓邊緣更圓滑
                 V_fixed: float = 50,
                 S_repr: float = 50,
                 folder: str = "LED2HSV"):
        self.N      = n_divisions
        self.R      = radius
        self.ds     = grid
        self.V  = max(0.0, min(float(V_fixed), 100.0)) / 100.0
        self.Sr = max(0.0, min(float(S_repr), 100.0)) / 100.0
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)

        # 在記憶體裡先生成底圖
        self._base = self._make_base()
        # 再算 N+1 個代表點
        self._reps = self._make_representatives()

    def _make_base(self):
        # 建格點
        g  = np.arange(-self.R, self.R + self.ds, self.ds)
        xx, yy = np.meshgrid(g, g)
        rr     = np.sqrt(xx**2 + yy**2)
        mask   = rr <= self.R

        H = (np.arctan2(yy, xx) / (2*np.pi)) % 1.0
        S = np.clip(rr / self.R, 0, 1)
        #固定self.V 為 1 
        V = np.ones_like(S) * 1

        hsv = np.stack([H, S, V], axis=-1)   # float 0–1
        rgb = hsv_to_rgb(hsv)                # float 0–1
        rgb[~mask] = 0                       # 圓外全黑
        return rgb
    
    def _hsv_to_led10bit(self, h, s, v):
        """
        HSV 0‥1 → 10-bit LED (0‥1023, int)
        """
        rgb01 = hsv_to_rgb(np.array([[[h, s, v]]])).reshape(3)   # 0‥1
        return tuple(np.round(rgb01 * 1023).astype(int))         # 轉 10-bit

    def _make_representatives(self):
        reps = []
        step = 360.0 / self.N
        for k in range(self.N):
            h_deg = k * step
            h01   = (h_deg / 360.0) % 1.0
            # 坐標：用飽和度 Sr * 半徑
            r_len = self.Sr * self.R
            x = r_len * np.cos(2*np.pi*h01)
            y = r_len * np.sin(2*np.pi*h01)

            # 亮度、飽和度採目前設定的 self.V / self.Sr
            R10, G10, B10 = self._hsv_to_led10bit(h01, self.Sr, self.V)
            rgb01 = np.array([R10, G10, B10], dtype=float) / 1023.0  # 轉回 0‥1

            reps.append({
            "h_deg": h_deg, "x": x, "y": y,
            "rgb01": rgb01,
            "R10": int(R10), "G10": int(G10), "B10": int(B10)
            })

        # # 中心灰點（S=0, H=0 不影響色相）
        # # rgb01c = hsv_to_rgb(np.array([[[0, 0, self.V]]]))[0,0]
        # rgb01c = hsv_to_rgb(np.stack([
        #    np.zeros((1,1)),  # H = 0（不影響）
        #    np.zeros((1,1)),  # S = 0
        #    np.full((1,1), self.V)
        # ], axis=-1))[0,0]

        # 中心灰點（S=0）
        R10c, G10c, B10c = self._hsv_to_led10bit(0.0, 0.0, self.V)
        rgb01c = np.array([R10c, G10c, B10c], dtype=float) / 1023.0
        reps.append({
            "h_deg": None, "x": 0.0, "y": 0.0,
            "rgb01": rgb01c,
            "R10": int(R10c), "G10": int(G10c), "B10": int(B10c)
        })
        return reps
          
        # R10c, G10c, B10c = (rgb01c * 1023).round().astype(int)
        # reps.append({
        #     "h_deg": None, "x": 0.0, "y": 0.0,
        #     "rgb01": rgb01c,
        #     "R10": int(R10c), "G10": int(G10c), "B10": int(B10c)
        # })
        # return reps

    def representatives(self):
        """回傳代表色資訊（含 x,y,rgb01,R10,G10,B10）"""
        return self._reps

    def plot(
        self,
        dot_size: int = 200,
        save_as: str | None = None,
        dpi: int = 300,
        ):
        """
        ① 繪出 HSV 彩虹圓盤 + 代表點  
        ② 另外輸出一張「10-bit LED 代表色表格」JPEG  
        ③ 直接回傳 (彩虹盤路徑, 表格路徑) 供外部使用
        """
        # --------------------------------------------------
        # ① 彩虹圓盤 + 代表點
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
        ax.set_facecolor("black")
        ax.axis("off")

        # 背景
        ax.imshow(
            self._base,
            extent=[-self.R, self.R, -self.R, self.R],
            origin="lower",
            interpolation="bicubic",
            zorder=0,
        )

        ax.axhline(0, color="white", lw=1, zorder=1)
        ax.axvline(0, color="white", lw=1, zorder=1)

        # 疊代表點
        for r in self._reps:
            x, y = r["x"], r["y"]
            rgb01 = np.array([r["R10"], r["G10"], r["B10"]], float) / 1023

            ax.scatter(
                x,
                y,
                s=dot_size * 1.3,
                c="black",
                edgecolors="black",
                linewidths=1.4,
                zorder=2,
            )
            ax.scatter(
                x,
                y,
                s=dot_size,
                c=[rgb01],
                edgecolors="none",
                zorder=3,
            )

            lab = "C" if r["h_deg"] is None else f"{int(r['h_deg']):d}°"
            ax.text(
                x * 1.1,
                y * 1.1,
                lab,
                color="black",
                fontsize=6,
                ha="center",
                va="center",
                zorder=4,
            )

        # --- 彩虹盤檔名 ------------------------------------
        if save_as is None:
            disc_path = os.path.join(
                self.folder,
                f"HSV_Disc_{self.Sr:.2f}_{self.V:.2f}_{self.N}div.jpg",
            )
        else:
            disc_path = save_as

        plt.tight_layout()
        plt.savefig(
            disc_path,
            dpi=dpi,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"✔ 彩虹圓盤已儲存：{disc_path}")

        # --------------------------------------------------
        # ② 10-bit LED 代表色表格（黑底白字）
        # --------------------------------------------------
        lines = [
            "Hue   S*   V*   R10   G10   B10",
            "-------------------------------",
        ]
        for r in self._reps:
            hue_txt = "  C" if r["h_deg"] is None else f"{int(r['h_deg']):3d}"
            Sstar = f"{int(self.Sr * 100):>3d}"
            Vstar = f"{int(self.V * 100):>3d}"
            lines.append(
                f"{hue_txt}°  {Sstar}  {Vstar}  "
                f"{r['R10']:4d}  {r['G10']:4d}  {r['B10']:4d}"
            )

        fig_h = max(2, 0.45 * len(lines))
        fig_w = 4
        fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h), facecolor="black")
        ax2.set_facecolor("black")
        ax2.axis("off")

        x0, y0 = 0.03, 0.97
        dy = 0.9 / (len(lines) - 1)
        for i, txt in enumerate(lines):
            ax2.text(
                x0,
                y0 - i * dy,
                txt,
                color="white",
                fontfamily="monospace",
                fontsize=10,
                ha="left",
                va="top",
                transform=ax2.transAxes,
            )

        table_path = os.path.join(
            self.folder,
            f"HSV_Rep_{self.Sr:.2f}_{self.V:.2f}_{self.N}div.jpg",
        )
        plt.tight_layout()
        plt.savefig(
            table_path,
            dpi=200,
            facecolor=fig2.get_facecolor(),
            bbox_inches="tight",
        )
        plt.close(fig2)
        print(f"✔ 10-bit 代表色表格已儲存：{table_path}")

        # --------------------------------------------------
        # ③ 回傳兩張圖的絕對路徑，外部函式直接接收
        # --------------------------------------------------
        return os.path.abspath(disc_path), os.path.abspath(table_path)
