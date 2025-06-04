#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB (10-bit)  →  CIELAB 2D a*b* 平面可視化
------------------------------------------------
1. 讀取 ALL.csv   (欄位須含 R, G, B，值域 0–1023)
2. 轉成 0–1 sRGB → CIELAB
3. 以 a* 為 X，b* 為 Y 畫散佈；顏色＝原 RGB
4. 另存 PNG 檔
------------------------------------------------
需要套件：
pip install pandas numpy matplotlib scikit-image
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab

# ---------- 讀取 CSV ----------
CSV_PATH = 'ALL.csv'
df = pd.read_csv(CSV_PATH)          # R, G, B (0–1023)

# ---------- 10-bit RGB → LAB ----------
rgb_norm = df[['R', 'G', 'B']].to_numpy() / 1023.0 # 正規化到 0–1 sRGB
lab = rgb2lab(rgb_norm.reshape(-1, 1, 3)).reshape(-1, 3)
df[['L', 'a', 'b']] = lab          # 方便後續若要用到 L*

# ---------- 繪製 a*–b* 平面 ----------
plt.figure(figsize=(8, 8), facecolor='black')
ax = plt.gca()
ax.set_facecolor('black')

# 散點：顏色 = 原 RGB (0–1)
plt.scatter(
    df['a'], df['b'],
    c=rgb_norm, s=20, edgecolors='none', alpha=0.9
)

# 座標設定：CIELAB 理論範圍 ±128
plt.xlim(-128, 128)
plt.ylim(-128, 128)
plt.xlabel('a*  (−G → +R)', color='white')
plt.ylabel('b*  (−B → +Y)', color='white')
plt.title('CIELAB  a*–b*  Scatter', color='white', pad=12)

# 原點輔助線
plt.axhline(0, color='white', lw=0.8, alpha=0.5)
plt.axvline(0, color='white', lw=0.8, alpha=0.5)

# 美化刻度與網格
ax.tick_params(colors='white')
ax.grid(color='white', linestyle='--', linewidth=0.3, alpha=0.3)

# 儲存與顯示
OUT_PNG = 'LAB_ab_plane1.png'
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='black')
plt.show()

print(f'完成！已輸出 {OUT_PNG}')
