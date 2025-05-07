#!/usr/bin/env python3
"""
把測試結果放在 CSV：
    channel,intensity,brightness
    W,0,   2.1
    W,128, 25.4
    ...
    B,1024,210.3
usage:
    python wrgb_response_fit.py samples.csv
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from numpy.polynomial import Polynomial

OUT_JSON = "wrgb_coeffs.json"

def fit_channel(df: pd.DataFrame, deg: int = 3) -> dict:
    x = df["intensity"].to_numpy()
    y = df["brightness"].to_numpy()
    # x 範圍 0‑1024 → 做一次 min‑max 正規化較穩定
    x_norm = (x - x.min()) / (x.max() - x.min())
    p = Polynomial.fit(x_norm, y, deg)      # least‑squares
    return {"coeffs": p.convert().coef.tolist(), "deg": deg}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("請給 CSV 檔", file=sys.stderr); sys.exit(1)

    df  = pd.read_csv(sys.argv[1])
    res = defaultdict(dict)
    for ch in ["W", "R", "G", "B"]:
        ch_df = df[df["channel"] == ch]
        if ch_df.empty: continue
        res[ch] = fit_channel(ch_df, deg=3)

    Path(OUT_JSON).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"已輸出係數 ➜ {OUT_JSON}")
