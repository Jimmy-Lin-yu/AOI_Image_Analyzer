# # LED2LAB.py

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.color import lab2rgb


# class LabHueDisc:
#     """
#     畫出固定 L* 截面的 CIELAB a*-b* 色域圓盤，
#     並標出等角代表色 (0°, 360/分割數, 2×360/分割數 …)。

#     參數
#     ----
#     L_fixed   : float   圓盤高度 (= 明度 L*)
#     C_repr    : float   代表點彩度 (= Chroma C*)
#     n_divisions: int    將 360° 分為幾等分 (例如 8 → 0°,45°,90°…315°)
#     radius    : int     圓盤顯示半徑 (a* / b* 範圍)
#     grid      : int     底圖取樣間隔 (1 = 最細，但耗時較久)
#     """

#     def __init__(self,
#                  L_fixed: float = 50,
#                  C_repr: float = 50,
#                  n_divisions: int = 8,
#                  radius: int = 110,
#                  grid: int = 1):
#         self.L  = L_fixed
#         self.C  = C_repr
#         self.N  = n_divisions
#         self.R  = radius
#         self.ds = grid

#         # 自動產生輸出資料夾
#         self.folder = "LED2LAB"
#         os.makedirs(self.folder, exist_ok=True)

#         # 預先算出 N 個等角代表點
#         self._reps = self._make_representatives()

#         # 產生底圖 (RGB 影像陣列)
#         self._rgb_img = self._make_hue_disc()

#     # ---------- 內部工具 ---------- #
#     @staticmethod
#     def _lab_to_led10bit(L, a, b):
#         """Lab → 10-bit LED (0‥1023)"""
#         rgb01 = lab2rgb(np.array([[[L, a, b]]])).reshape(3)
#         return tuple(np.round(rgb01 * 1023).astype(int))

#     def _make_representatives(self):
#         reps = []
#         # 每一步的角度間隔
#         step = 360.0 / self.N
#         for k in range(self.N):
#             h = k * step
#             rad = np.deg2rad(h)
#             a = self.C * np.cos(rad)
#             b = self.C * np.sin(rad)
#             R10, G10, B10 = self._lab_to_led10bit(self.L, a, b)
#             reps.append({
#                 'h': float(h),
#                 'L': float(self.L),
#                 'a': float(a),
#                 'b': float(b),
#                 'R10': int(R10),
#                 'G10': int(G10),
#                 'B10': int(B10)
#             })

#         # —— 新增：圓心中間點 —— #
#         # a*=0, b*=0 就是純灰 (同 L* 的中間灰)
#         R10c, G10c, B10c = self._lab_to_led10bit(self.L, 0.0, 0.0)
#         reps.append({
#             'h': None,           # 中間點沒有色相
#             'L': float(self.L),
#             'a': 0.0,
#             'b': 0.0,
#             'R10': int(R10c),
#             'G10': int(G10c),
#             'B10': int(B10c)
#         })

#         return reps
    
    

#     def _make_hue_disc(self):
#         g = np.arange(-self.R, self.R + self.ds, self.ds)
#         aa, bb = np.meshgrid(g, g)
#         mask   = aa**2 + bb**2 <= self.R**2

#         lab_img        = np.zeros((*aa.shape, 3))
#         lab_img[..., 0] = self.L
#         lab_img[..., 1] = aa
#         lab_img[..., 2] = bb

#         rgb_img = lab2rgb(lab_img)
#         rgb_img[~mask] = 0      # 圓外黑
#         return rgb_img

#     # ---------- 公開介面 ---------- #
#     def representatives(self):
#         """回傳 N 色代表點 (list of dict)，每筆包含 h/L/a/b/R10/G10/B10"""
#         return self._reps

#     def plot(self,
#              label_shift: float = 1.18,
#              dot_size: int = 170,
#              save_as: str | None = None,
#              dpi: int = 300):
#         """
#         繪出 hue 圓盤並標記代表點。
#         若未提供 save_as，則自動命名為：
#           LED2LAB/Hue_Disc_L{L}_C{C}_{N}div.jpg
#         """

#         fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
#         ax.set_facecolor('black')

#         # 底圖：色域圓盤
#         im = ax.imshow(
#             self._rgb_img,
#             extent=[-self.R, self.R, -self.R, self.R],
#             origin='lower',
#             interpolation='bilinear'
#         )
#         im.set_clip_path(plt.Circle((0, 0), self.R, transform=ax.transData))

#         # 座標軸
#         ax.axhline(0, color='white', lw=.8)
#         ax.axvline(0, color='white', lw=.8)

#         # 代表點
#         for r in self._reps:
#             # 黑底 + 白框
#             ax.scatter(
#                 r['a'], r['b'],
#                 s=dot_size * 1.35,
#                 color='black',
#                 edgecolors='black',
#                 linewidths=1.4,
#                 zorder=3
#             )
#             # 疊實際顏色
#             dot_rgb = np.array([[r['R10'], r['G10'], r['B10']]], float) / 1023
#             ax.scatter(
#                 r['a'], r['b'],
#                 s=dot_size,
#                 color=dot_rgb,
#                 edgecolors='none',
#                 zorder=4
#             )
#             # 標籤文字拉外
#             if r['h'] is None:
#                 label = "C"
#             else:
#                 label = f"{int(r['h']):>3d}°"
#             ax.text(
#                 r['a'] * label_shift,
#                 r['b'] * label_shift,
#                 f"{label}°",
#                 color='black',
#                 fontsize=5,
#                 ha='center',
#                 va='center'
#             )

#         ax.set_xlim(-self.R, self.R)
#         ax.set_ylim(-self.R, self.R)
#         ax.set_aspect('equal', 'box')
#         ax.set_xlabel("a*  (–G → +R)", color='white')
#         ax.set_ylabel("b*  (–B → +Y)", color='white')
#         ax.set_title(
#             f"CIELAB Hue Disc (L*={self.L}, C*={self.C}, Divisions={self.N})",
#             color='white', pad=12
#         )
#         ax.tick_params(colors='white')
#         plt.tight_layout()

#         # 如果使用者沒有指定儲存檔名，就自動命名並放到 LED2LAB 資料夾
#         if save_as is None:
#             save_as = os.path.join(
#                 self.folder,
#                 f"Hue_Disc_L{self.L}_C{self.C}_{self.N}div.jpg"
#             )

#         plt.savefig(save_as, dpi=dpi,
#                     bbox_inches='tight', facecolor=fig.get_facecolor())
#         print(f"✔ 圖檔已儲存：{save_as}")
#         plt.close(fig)

#     def save_representatives_as_image(self,
#                                     reps: list[dict] | None = None,
#                                     out_path: str | None = None):
#         """
#         將 reps 列表（由 LabHueDisc.representatives() 回傳的）：
#         Hue, L*, a*, b*, R10, G10, B10，繪成對齊表格並存為 JPG。

#         reps     : list[dict]    # 代表點列表
#         out_path : str | None    # 輸出檔名；若 None，則預設為 LED2LAB/Hue_Rep_L{L}_C{C}_{N}div.jpg
#         """
#         import pandas as pd

#         # 確認 reps
#         if reps is None:
#             reps = self._reps

#         # 自動建立資料夾
#         os.makedirs(self.folder, exist_ok=True)

#         # 自動命名
#         if out_path is None:
#             filename = f"Hue_Rep_L{self.L:.1f}_C{self.C:.1f}_{self.N}div.jpg"
#             out_path = os.path.join(self.folder, filename)

#         # 建 DataFrame
#         rows = []
#         for r in reps:
#             hue = 'C' if r['h'] is None else f"{int(r['h'])}°"
#             rows.append({
#                 'Hue': hue,
#                 'L*': f"{r['L']:.1f}",
#                 'a*': f"{r['a']:.2f}",
#                 'b*': f"{r['b']:.2f}",
#                 'R10': str(r['R10']),
#                 'G10': str(r['G10']),
#                 'B10': str(r['B10']),
#             })
#         df = pd.DataFrame(rows, columns=['Hue','L*','a*','b*','R10','G10','B10'])

#         # 計算圖高：至少 2 吋，每行 0.6 吋
#         fig_height = max(2, 0.6 * len(df))
#         fig, ax = plt.subplots(figsize=(8, fig_height), facecolor='black')
#         ax.set_facecolor('black')
#         ax.axis('off')

#         # 畫 table
#         table = ax.table(
#             cellText=df.values,
#             colLabels=df.columns,
#             cellLoc='center',
#             colLoc='center',
#             loc='center'
#         )
#         table.auto_set_font_size(False)
#         table.set_fontsize(12)
#         table.scale(1, 1.5)

#         plt.tight_layout()
#         fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
#         plt.close(fig)
#         print(f"✔ 對齊後的代表色表格已儲存：{out_path}")



# # =========================================================
# # 示範用法
# # =========================================================
# if __name__ == "__main__":
#     # 例如：L* = 50, C* = 50, 360 分為 12 等分
#     disc = LabHueDisc(L_fixed=50, C_repr=50, n_divisions=12)

#     # 取出所有代表點
#     reps = disc.representatives()

#     # 儲存表格到 LED2LAB/Hue_Rep_L50.0_C50.0_12div.jpg
#     disc.save_representatives_as_image(reps)

#     # 繪圖並存檔到 LED2LAB/Hue_Disc_L50_C50_12div.jpg
#     disc.plot()

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.color import lab2rgb


# class LabHueDisc:
#     """
#     畫出固定 L* 截面的 CIELAB a*-b* 色域圓盤，
#     並標出等角代表色 (0°, 360/分割數, 2×360/分割數 …)。

#     參數
#     ----
#     L_fixed   : float   圓盤高度 (= 明度 L*)
#     C_repr    : float   代表點彩度 (= Chroma C*)
#     n_divisions: int    將 360° 分為幾等分 (例如 8 → 0°,45°,90°…315°)
#     radius    : int     圓盤顯示半徑 (a* / b* 範圍)
#     grid      : int     底圖取樣間隔 (1 = 最細，但耗時較久)
#     """

#     def __init__(self,
#                  L_fixed: float = 50,
#                  C_repr: float = 50,
#                  n_divisions: int = 8,
#                  radius: int = 110,
#                  grid: int = 1):
#         self.L  = L_fixed
#         self.C  = C_repr
#         self.N  = n_divisions
#         self.R  = radius
#         self.ds = grid

#         # 自動產生輸出資料夾
#         self.folder = "LED2LAB"
#         os.makedirs(self.folder, exist_ok=True)

#         # 預先算出 N 個等角代表點
#         self._reps = self._make_representatives()

#         # 產生底圖 (RGB 影像陣列)
#         self._rgb_img = self._make_hue_disc()

#     # ---------- 內部工具 ---------- #
#     @staticmethod
#     def _lab_to_led10bit(L, a, b):
#         """Lab → 10-bit LED (0‥1023)"""
#         rgb01 = lab2rgb(np.array([[[L, a, b]]])).reshape(3)
#         return tuple(np.round(rgb01 * 1023).astype(int))

#     def _make_representatives(self):
#         reps = []
#         # 每一步的角度間隔
#         step = 360.0 / self.N
#         for k in range(self.N):
#             h = k * step
#             rad = np.deg2rad(h)
#             a = self.C * np.cos(rad)
#             b = self.C * np.sin(rad)
#             R10, G10, B10 = self._lab_to_led10bit(self.L, a, b)
#             reps.append({
#                 'h': float(h),
#                 'L': float(self.L),
#                 'a': float(a),
#                 'b': float(b),
#                 'R10': int(R10),
#                 'G10': int(G10),
#                 'B10': int(B10)
#             })

#         # —— 新增：圓心中間點 —— #
#         # a*=0, b*=0 就是純灰 (同 L* 的中間灰)
#         R10c, G10c, B10c = self._lab_to_led10bit(self.L, 0.0, 0.0)
#         reps.append({
#             'h': None,           # 中間點沒有色相
#             'L': float(self.L),
#             'a': 0.0,
#             'b': 0.0,
#             'R10': int(R10c),
#             'G10': int(G10c),
#             'B10': int(B10c)
#         })

#         return reps
    
    

#     def _make_hue_disc(self):
#         g = np.arange(-self.R, self.R + self.ds, self.ds)
#         aa, bb = np.meshgrid(g, g)
#         mask   = aa**2 + bb**2 <= self.R**2

#         lab_img        = np.zeros((*aa.shape, 3))
#         lab_img[..., 0] = self.L
#         lab_img[..., 1] = aa
#         lab_img[..., 2] = bb

#         rgb_img = lab2rgb(lab_img)
#         rgb_img[~mask] = 0      # 圓外黑
#         return rgb_img

#     # ---------- 公開介面 ---------- #
#     def representatives(self):
#         """回傳 N 色代表點 (list of dict)，每筆包含 h/L/a/b/R10/G10/B10"""
#         return self._reps

    # def plot(self,
    #          label_shift: float = 1.18,
    #          dot_size: int = 170,
    #          save_as: str | None = None,
    #          dpi: int = 300):
    #     """
    #     繪出 hue 圓盤並標記代表點。
    #     若未提供 save_as，則自動命名為：
    #       LED2LAB/Hue_Disc_L{L}_C{C}_{N}div.jpg
    #     """

    #     fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    #     ax.set_facecolor('black')

    #     # 底圖：色域圓盤
    #     im = ax.imshow(
    #         self._rgb_img,
    #         extent=[-self.R, self.R, -self.R, self.R],
    #         origin='lower',
    #         interpolation='bilinear'
    #     )
    #     im.set_clip_path(plt.Circle((0, 0), self.R, transform=ax.transData))

    #     # 座標軸
    #     ax.axhline(0, color='white', lw=.8)
    #     ax.axvline(0, color='white', lw=.8)

    #     # 代表點
    #     for r in self._reps:
    #         # 黑底 + 白框
    #         ax.scatter(
    #             r['a'], r['b'],
    #             s=dot_size * 1.35,
    #             color='black',
    #             edgecolors='black',
    #             linewidths=1.4,
    #             zorder=3
    #         )
    #         # 疊實際顏色
    #         dot_rgb = np.array([[r['R10'], r['G10'], r['B10']]], float) / 1023
    #         ax.scatter(
    #             r['a'], r['b'],
    #             s=dot_size,
    #             color=dot_rgb,
    #             edgecolors='none',
    #             zorder=4
    #         )
    #         # 標籤文字拉外
    #         if r['h'] is None:
    #             label = "C"
    #         else:
    #             label = f"{int(r['h']):>3d}°"
    #         ax.text(
    #             r['a'] * label_shift,
    #             r['b'] * label_shift,
    #             f"{label}°",
    #             color='black',
    #             fontsize=5,
    #             ha='center',
    #             va='center'
    #         )

    #     ax.set_xlim(-self.R, self.R)
    #     ax.set_ylim(-self.R, self.R)
    #     ax.set_aspect('equal', 'box')
    #     ax.set_xlabel("a*  (–G → +R)", color='white')
    #     ax.set_ylabel("b*  (–B → +Y)", color='white')
    #     ax.set_title(
    #         f"CIELAB Hue Disc (L*={self.L}, C*={self.C}, Divisions={self.N})",
    #         color='white', pad=12
    #     )
    #     ax.tick_params(colors='white')
    #     plt.tight_layout()

    #     # 如果使用者沒有指定儲存檔名，就自動命名並放到 LED2LAB 資料夾
    #     if save_as is None:
    #         save_as = os.path.join(
    #             self.folder,
    #             f"Hue_Disc_L{self.L}_C{self.C}_{self.N}div.jpg"
    #         )

    #     plt.savefig(save_as, dpi=dpi,
    #                 bbox_inches='tight', facecolor=fig.get_facecolor())
    #     print(f"✔ 圖檔已儲存：{save_as}")
    #     plt.close(fig)

#     def save_representatives_as_image(self,
#                                     reps: list[dict] | None = None,
#                                     out_path: str | None = None):
#         """
#         將 reps 列表（由 LabHueDisc.representatives() 回傳的）：
#         Hue, L*, a*, b*, R10, G10, B10，繪成對齊表格並存為 JPG。

#         reps     : list[dict]    # 代表點列表
#         out_path : str | None    # 輸出檔名；若 None，則預設為 LED2LAB/Hue_Rep_L{L}_C{C}_{N}div.jpg
#         """
#         import pandas as pd

#         # 確認 reps
#         if reps is None:
#             reps = self._reps

#         # 自動建立資料夾
#         os.makedirs(self.folder, exist_ok=True)

#         # 自動命名
#         if out_path is None:
#             filename = f"Hue_Rep_L{self.L:.1f}_C{self.C:.1f}_{self.N}div.jpg"
#             out_path = os.path.join(self.folder, filename)

#         # 建 DataFrame
#         rows = []
#         for r in reps:
#             hue = 'C' if r['h'] is None else f"{int(r['h'])}°"
#             rows.append({
#                 'Hue': hue,
#                 'L*': f"{r['L']:.1f}",
#                 'a*': f"{r['a']:.2f}",
#                 'b*': f"{r['b']:.2f}",
#                 'R10': str(r['R10']),
#                 'G10': str(r['G10']),
#                 'B10': str(r['B10']),
#             })
#         df = pd.DataFrame(rows, columns=['Hue','L*','a*','b*','R10','G10','B10'])

#         # 計算圖高：至少 2 吋，每行 0.6 吋
#         fig_height = max(2, 0.6 * len(df))
#         fig, ax = plt.subplots(figsize=(8, fig_height), facecolor='black')
#         ax.set_facecolor('black')
#         ax.axis('off')

#         # 畫 table
#         table = ax.table(
#             cellText=df.values,
#             colLabels=df.columns,
#             cellLoc='center',
#             colLoc='center',
#             loc='center'
#         )
#         table.auto_set_font_size(False)
#         table.set_fontsize(12)
#         table.scale(1, 1.5)

#         plt.tight_layout()
#         fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
#         plt.close(fig)
#         print(f"✔ 對齊後的代表色表格已儲存：{out_path}")



# # =========================================================
# # 示範用法
# # =========================================================
# if __name__ == "__main__":
#     # 例如：L* = 50, C* = 50, 360 分為 12 等分
#     disc = LabHueDisc(L_fixed=50, C_repr=50, n_divisions=12)

#     # 取出所有代表點
#     reps = disc.representatives()

#     # 儲存表格到 LED2LAB/Hue_Rep_L50.0_C50.0_12div.jpg
#     disc.save_representatives_as_image(reps)

# LED2LAB.py

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab
from matplotlib.colors import hsv_to_rgb
from matplotlib.image import imread, imsave


class LabHueDisc:
    """
    畫出固定 L* 截面的 CIELAB a*-b* 色域圓盤，
    並標出等角代表色 (0°, 360/分割數, 2×360/分割數 …)。

    參數
    ----
    L_fixed   : float   圓盤高度 (= 明度 L*)
    C_repr    : float   代表點彩度 (= Chroma C*)
    n_divisions: int    將 360° 分為幾等分 (例如 8 → 0°,45°,90°…315°)
    radius    : int     圓盤顯示半徑 (a* / b* 範圍)
    grid      : int     底圖取樣間隔 (1 = 最細，但耗時較久)
    """

    BASE_FILENAME = "Hue_Disc_Base.png"

    def __init__(self,
                 L_fixed: float = 50,
                 C_repr: float = 50,
                 n_divisions: int = 8,
                 radius: int = 110,
                 grid: int = 1):
        self.L  = L_fixed
        self.C  = C_repr
        self.N  = n_divisions
        self.R  = radius
        self.ds = grid

        # 建資料夾
        self.folder = "LED2LAB"
        os.makedirs(self.folder, exist_ok=True)

        # 如果底圖不存在，就生成並存檔
        self.base_path = os.path.join(self.folder, self.BASE_FILENAME)
        if not os.path.isfile(self.base_path):
            self._generate_base_image()

        # 代表點
        self._reps = self._make_representatives()

    def _generate_base_image(self):
        """
        用 HSV 生成彩虹圓盤：
        - 中心 sat=0 → 白
        - 半徑→飽和度，緣→sat=1
        - 角度→色相
        """
        g = np.arange(-self.R, self.R + self.ds, self.ds)
        aa, bb = np.meshgrid(g, g)
        rr = np.sqrt(aa**2 + bb**2)
        mask = rr <= self.R

        # HSV：H = angle/360, S = r/R, V = 1
        H = (np.arctan2(bb, aa) / (2*np.pi) + 0.5) % 1.0
        S = np.clip(rr / self.R, 0, 1)
        V = np.ones_like(S)
        hsv = np.stack([H, S, V], axis=-1)

        rgb = hsv_to_rgb(hsv)
        rgb[~mask] = 0

        # 存 8-bit PNG
        imsave(self.base_path, (rgb * 255).astype(np.uint8))
        print(f"✔ 底圖已存：{self.base_path}")


    def _lab_to_led10bit(self, L, a, b):
        rgb01 = lab2rgb(np.array([[[L, a, b]]])).reshape(3)
        return tuple(np.round(rgb01 * 1023).astype(int))

    def _make_representatives(self):
        reps = []
        step = 360.0 / self.N
        for k in range(self.N):
            h = k * step
            rad = np.deg2rad(h)
            a = self.C * np.cos(rad)
            b = self.C * np.sin(rad)
            R10, G10, B10 = self._lab_to_led10bit(self.L, a, b)
            reps.append({'h': h, 'a': a, 'b': b,
                         'R10': R10, 'G10': G10, 'B10': B10})
        # 中心灰
        Rc, Gc, Bc = self._lab_to_led10bit(self.L, 0, 0)
        reps.append({'h': None, 'a': 0.0, 'b': 0.0,
                     'R10': Rc, 'G10': Gc, 'B10': Bc})
        return reps

    def representatives(self):
        return self._reps

    def plot(self,
            label_shift: float = 1.18,
            dot_size: int = 170,
            save_as: str | None = None,
            dpi: int = 300):
        """
        動態產生 HSV 色盤底圖（中心白、邊緣彩虹），
        再疊上 CIELAB 代表點。
        """
        # 1) 產生 HSV 底圖
        # 網格座標
        R = self.R
        grid = np.linspace(-R, R, 2*R+1)
        xx, yy = np.meshgrid(grid, grid)
        rr = np.sqrt(xx**2 + yy**2)
        mask = rr <= R

        # 角度 → Hue [0..1]
        theta = (np.arctan2(yy, xx) + 2*np.pi) % (2*np.pi)  # 0~2π
        h = theta / (2*np.pi)

        # 半徑 → Saturation [0..1]
        s = np.clip(rr / R, 0, 1)

        # Value 固定 1
        v = np.ones_like(s)

        # HSV 合併，轉 RGB
        hsv = np.stack([h, s, v], axis=-1)        # shape=(H,W,3)
        rgb = hsv_to_rgb(hsv)                     # 0..1
        rgb[~mask] = 0                            # 圓外黑

        # 2) 建圖
        fig, ax = plt.subplots(figsize=(6,6), facecolor='black')
        ax.set_facecolor('black')
        ax.axis('off')

        # 3) 畫底圖
        ax.imshow(
            rgb,
            extent=[-R, R, -R, R],
            origin='lower',
            interpolation='bilinear',
            zorder=0
        )

        # 4) 座標軸
        ax.axhline(0, color='white', lw=0.8, zorder=1)
        ax.axvline(0, color='white', lw=0.8, zorder=1)

        # 5) 疊代表點
        for r in self._reps:
            x, y = r['a'], r['b']
            # 黑底外框
            ax.scatter(x, y,
                    s=dot_size*1.3,
                    c='black',
                    edgecolors='black',
                    linewidths=1.4,
                    zorder=2)
            # 真實顏色
            rgb_dot = np.array([r['R10'], r['G10'], r['B10']]) / 1023
            ax.scatter(x, y,
                    s=dot_size,
                    c=[rgb_dot],
                    edgecolors='none',
                    zorder=3)
            # 文字
            lab = 'C' if r['h'] is None else f"{int(r['h']):d}°"
            ax.text(x*label_shift, y*label_shift,
                    lab,
                    color='black', fontsize=6,
                    ha='center', va='center',
                    zorder=4)

        # 6) 存檔
        if save_as is None:
            save_as = os.path.join(
                self.folder,
                f"Hue_Disc_L{self.L}_C{self.C}_{self.N}div.jpg"
            )
        plt.tight_layout()
        plt.savefig(save_as, dpi=dpi,
                    facecolor=fig.get_facecolor(),
                    bbox_inches='tight')
        plt.close(fig)
        print(f"✔ 圖檔已儲存：{save_as}")


if __name__ == "__main__":
    # 示範：先生成底圖（只執行一次）
    disc = LabHueDisc(L_fixed=50, C_repr=50, n_divisions=8)
    # 之後每次只要改 L_fixed 再呼叫 plot，就只重算並疊點
    disc.L = 25
    disc._reps = disc._make_representatives()
    disc.plot()



    # ---------- 3. 用 LabHueDisc 算 N 個代表色 ----------
    disc = LabHueDisc(L_fixed=L_fixed,
                      C_repr=C_repr,
                      n_divisions=n_divisions)
    reps = disc.representatives()  # list of dict（含 R10,G10,B10）

    # （一）儲存「代表色表格」到 JPEG
    #   LabHueDisc.save_representatives_as_image() 預設會儲存到 LED2LAB 資料夾
    #   並且檔名會是 Hug_Rep_L{L}_C{C}_{N}div.jpg
    disc._generate_base_image()
    # 我們可以手動組出那個路徑
    rep_filename = f"Hue_Rep_L{disc.L:.1f}_C{disc.C:.1f}_{disc.N}div.jpg"
    rep_img_path = str(Path(disc.folder) / rep_filename)

    # （二）儲存「CIELAB Hue Disc」到 JPEG
    #   LabHueDisc.plot() 如果不給 save_as，就會採預設格式：
    #   Hue_Disc_L{L}_C{C}_{N}div.jpg 放到 LED2LAB 資料夾
    disc.plot()
    disc_filename = f"Hue_Disc_L{disc.L}_C{disc.C}_{disc.N}div.jpg"
    disc_img_path = str(Path(disc.folder) / disc_filename)