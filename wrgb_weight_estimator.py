#!/usr/bin/env python3
"""
CSV 檔格式：
    W,R,G,B,brightness
    350,0,0,179,  67.3
    512,256,0,0,  84.6
    ...

usage:
    python wrgb_weight_estimator.py mixed.csv wrgb_coeffs.json
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from numpy.polynomial import Polynomial
from sklearn.linear_model import LinearRegression

def load_funcs(path: Path):
    data = json.loads(Path(path).read_text())
    fns  = {}
    for ch, pinfo in data.items():
        poly = Polynomial(pinfo["coeffs"])
        fns[ch] = lambda x, p=poly: p((x - 0) / 1024)   # 按 0‑1024 正規化
    return fns

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: mixed.csv coeffs.json", file=sys.stderr); sys.exit(1)

    mix_df = pd.read_csv(sys.argv[1])
    funcs  = load_funcs(sys.argv[2])

    # 依函式把強度轉成「單通道理論亮度貢獻」
    X = np.column_stack([funcs[ch](mix_df[ch].to_numpy()) for ch in ["W","R","G","B"]])
    y = mix_df["brightness"].to_numpy()

    # 不要截距 (亮度=0 時權重應為 0)
    model = LinearRegression(fit_intercept=False).fit(X, y)
    weights = model.coef_ / model.coef_.sum()   # 正規化成「相對權重」

    for ch, w in zip(["W","R","G","B"], weights):
        print(f"{ch} weight = {w:.3f}")
