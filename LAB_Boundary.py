import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans

# ============ 參數 ============
CSV_IN          = "ALL.csv"          # 原始 10-bit 資料
CSV_OUT         = "LED_representatives.csv"
HUE_BINS        = 16                  # 色相分 16 區
C_MIN           = 30                 # ★ 以 √(L²+a²+b²) 為門檻，數值可自行調
REPR_PER_BIN    = 1                   # 每區取 1 點
METHOD          = "centroid"          # 或 "kmeans"
SEED            = 0

# ============ 1. 讀檔 & 10-bit → Lab ============
df = pd.read_csv(CSV_IN)                       # R,G,B 0–1023
rgb_norm = df[['R', 'G', 'B']].to_numpy() / 1023.0
lab = rgb2lab(rgb_norm.reshape(-1, 1, 3)).reshape(-1, 3)
df[['L', 'a', 'b']] = lab

# ============ 2. 色相角 h 與「球半徑」C ============
h = np.degrees(np.arctan2(df['b'], df['a']))      # -180°~+180°
h = (h + 360) % 360                               # 0~360
C = np.sqrt(df['L']**2 + df['a']**2 + df['b']**2) # ★ 包含 L*
df['h'] = h
df['C'] = C

df_colored = df[df['C'] >= C_MIN].copy()          # 過濾低 C

# ============ 3. 依色相分組 & 抽代表點 ============
bin_w = 360 / HUE_BINS
repr_rows = []

for i in range(HUE_BINS):
    h0, h1 = i * bin_w, (i + 1) * bin_w
    sub = df_colored[(df_colored['h'] >= h0) & (df_colored['h'] < h1)]
    if sub.empty:
        continue

    if REPR_PER_BIN == 1 and METHOD == "centroid":
        center = sub[['L', 'a', 'b']].mean().to_numpy()
        idx    = ((sub[['L', 'a', 'b']] - center) ** 2).sum(1).idxmin()
        repr_rows.append(sub.loc[idx])
    else:
        k = min(REPR_PER_BIN, len(sub))
        km = KMeans(n_clusters=k, random_state=SEED).fit(sub[['L', 'a', 'b']])
        for c in km.cluster_centers_:
            idx = ((sub[['L', 'a', 'b']] - c) ** 2).sum(1).idxmin()
            repr_rows.append(sub.loc[idx])

rep = pd.DataFrame(repr_rows).reset_index(drop=True)
print(f"代表點數：{len(rep)}")

# ============ 4. 匯出 10-bit LED 代碼 ============
rep[['R', 'G', 'B']].round().astype(int).to_csv(CSV_OUT, index=False)
print(f"已輸出  {CSV_OUT}")

# ============ 5. 驗證圖 (a*–b* 投影) ============
plt.figure(figsize=(8, 8), facecolor='black')
plt.scatter(df['a'], df['b'], s=8, c='gray', alpha=.15)
plt.scatter(rep['a'], rep['b'], s=120,
            c=rep[['R', 'G', 'B']].to_numpy() / 1023.0,
            edgecolors='white')
plt.axhline(0, color='white', lw=.8); plt.axvline(0, color='white', lw=.8)
plt.xlim(-128, 128); plt.ylim(-128, 128)
plt.xlabel('a*  (−G → +R)', color='white')
plt.ylabel('b*  (−B → +Y)', color='white')
plt.title('CIELAB  a*–b*  Representatives', color='white')
plt.tick_params(colors='white')

OUT_PNG = "LAB_ab_Representatives2.png"
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='black')
plt.show()
print(f"完成！已輸出 {OUT_PNG}")
