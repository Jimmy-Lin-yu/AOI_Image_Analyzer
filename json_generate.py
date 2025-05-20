import gradio as gr
import json
import tempfile
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------------- 基本設定 ----------------
MODELS = ["全色域環形100", "全色域條形30100", "全色域同軸60",
          "4層4角度全圓88", "全色域圓頂120"]
KEY_MAP = dict(zip(MODELS, ["p1", "p2", "p3", "p4", "p5"]))

BRIGHTNESS_BASE = ["1024", "512"]
COLOR_BASE      = ["1000", "500", "0"]

###################################
# 1.產生單色光 JSON
###################################
def create_single_color_json(sk_file, model, channel):
    """
    根據選擇的 model & 單色 channel，僅生成 1~1024 強度的 scenes
    回傳 JSON 檔案路徑，保留骨架中 "device" 欄位，其他模型節點刪除
    """
    # 1) 讀取骨架 JSON
    path = sk_file.name
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # 2) 保留骨架前置設定（例如 "device"）
    new_data = {}
    if 'device' in data:
        new_data['device'] = data['device']

    # 3) 計算 channel 索引及最大強度
    idx_map = {'W':0, 'R':1, 'G':2, 'B':3}
    idx = idx_map[channel]
    max_int = 1024

    # 4) 生成 scenes 列表
    scenes = []
    for i in range(1, max_int+1):
        colors = [0,0,0,0]
        colors[idx] = i
        scenes.append({
            "brightness": max_int,
            "colors": colors,
            "currentZone": 0,
            "zoneMode": 0
        })

    # 5) 從骨架 JSON 取出對應節點並加上 scenes
    key = KEY_MAP[model]
    node = data.get(key)
    if node is None:
        raise KeyError(f"模型 {model} 對應的 KEY_MAP:{key} 不存在於骨架 JSON")

    # 深拷貝節點結構，僅保留這個節點
    if isinstance(node, list):
        entries = []
        for entry in node:
            entry_copy = {k:v for k,v in entry.items() if k!='scenes'}
            entry_copy['scenes'] = scenes
            entries.append(entry_copy)
        new_data[key] = entries
    else:
        entry_copy = {k:v for k,v in node.items() if k!='scenes'}
        entry_copy['scenes'] = scenes
        new_data[key] = entry_copy

    # 6) 輸出 JSON
    out_fname = f"single_{key}_{channel}.json"
    with open(out_fname, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    return out_fname

###################################
# 2.產生 JSON 排列組合
###################################
# ---------- 排列組合：亮度 × cw × cr × cg × cb ----------
def generate_combos_for_model(brightness_list, color_list):
    b_vals = [int(b) for b in brightness_list]
    c_vals = [int(c) for c in color_list]
    combos = []
    for b in b_vals:
        for cw in c_vals:          # 白
            for cr in c_vals:      # 紅
                for cg in c_vals:  # 綠
                    for cb in c_vals:  # 藍
                        combos.append({
                            "brightness": b,
                            "colors": [cw, cr, cg, cb],
                            "currentZone": 0,
                            "zoneMode": 0
                        })
    return combos

# ---------- 產生最終 JSON ----------
def create_json(sk_path, b_cnt, c_cnt, *txts):
    data = json.load(open(sk_path, encoding="utf-8"))
    it   = iter(txts)                       # 依序 b0,c0,b1,c1,...

    for model in MODELS:
        b_list = [x.strip() for x in next(it).split(",") if x.strip()]
        c_list = [x.strip() for x in next(it).split(",") if x.strip()]

        # 長度驗證（如需嚴格限制可取消註解）
        # if len(b_list) != b_cnt: raise ValueError(f"{model} 需 {b_cnt} 個亮度")
        # if len(c_list) != c_cnt: raise ValueError(f"{model} 需 {c_cnt} 個顏色強度")

        scenes = generate_combos_for_model(b_list, c_list)
        node   = data[KEY_MAP[model]]
        if isinstance(node, list):          # p2 為 list，其餘為 dict
            for n in node: n["scenes"] = scenes
        else:
            node["scenes"] = scenes

    out = "lightPara.json"
    json.dump(data, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=4)
    return out

###################################
# 3.WRGB 等分組合 JSON 排列組合
###################################

def create_division_color_json(sk_file, model, w_max, r_max, g_max, b_max, divisions):
    """
    根據上傳的骨架 JSON、選擇的模型，
    將 W、R、G、B 四個通道的最大值各自分為指定等分，
    並對所有通道值做排列組合，生成新的 scenes JSON。
    回傳輸出檔案名稱。
    """
    import json

    # 讀取骨架 JSON
    path = sk_file.name
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    # 保留 "device" 設定
    new_data = {}
    if 'device' in data:
        new_data['device'] = data['device']

    # 取得對應的模型節點
    key = KEY_MAP[model]
    node = data.get(key)
    if node is None:
        raise KeyError(f"模型 {model} 對應的 KEY_MAP:{key} 不存在於骨架 JSON")

    # 將最大值分為等分
    def divide(max_val, parts):
        if max_val == 0:
            return [0]
        step = max_val / parts
        # 生成從 1 到 max 的 (parts+1) 個值
        return [int(round(step * i)) for i in range(1, parts + 1)]

    w_vals = divide(int(w_max), int(divisions))
    r_vals = divide(int(r_max), int(divisions))
    g_vals = divide(int(g_max), int(divisions))
    b_vals = divide(int(b_max), int(divisions))

    # 生成所有通道的排列組合
    scenes = []
    for cw in w_vals:
        for cr in r_vals:
            for cg in g_vals:
                for cb in b_vals:
                    scenes.append({
                        "brightness": 1024,       # 可根據需求調整 brightness 欄位
                        "colors": [cw, cr, cg, cb],
                        "currentZone": 0,
                        "zoneMode": 0
                    })

    # 深拷貝節點並附加新的 scenes 列表
    if isinstance(node, list):
        entries = []
        for entry in node:
            entry_copy = {k: v for k, v in entry.items() if k != 'scenes'}
            entry_copy['scenes'] = scenes
            entries.append(entry_copy)
        new_data[key] = entries
    else:
        entry_copy = {k: v for k, v in node.items() if k != 'scenes'}
        entry_copy['scenes'] = scenes
        new_data[key] = entry_copy

    # 輸出 JSON
    out_fname = f"divisions_{key}.json"
    with open(out_fname, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    return out_fname
###################################
# 4.WRGB  抽樣打光 JSON 排列組合
###################################

def train_channel_models(
    w_csv, r_csv, g_csv, b_csv,
    w_range=(0,1024), r_range=(0,1024), g_range=(0,1024), b_range=(0,1024)
):
    """
    只做一次項線性回歸。
    回傳兩份 dict：
        funcs   : {"W": f_w(x), ...}
        params  : {"W": (coef, intercept), ...}
    """
    out_f, out_p = {}, {}
    for ch, csv_path, rng in zip(
        ["W","R","G","B"], 
        [w_csv,r_csv,g_csv,b_csv],
        [w_range,r_range,g_range,b_range]
    ):
        # 缺通道
        if csv_path is None:
            out_f[ch] = (lambda x: 0.0)
            out_p[ch] = (0.0, 0.0)
            continue

        df = pd.read_csv(csv_path)
        lo, hi = rng
        df = df[(df[ch]>=lo) & (df[ch]<=hi)]
        if df.empty:
            raise ValueError(f"{csv_path} 在 {rng} 區間無資料")
        X = df[[ch]].values; y = df["Br"].values
        lin = LinearRegression().fit(X, y)
        a, b = float(lin.coef_[0]), float(lin.intercept_)
        # 避免 late-binding
        out_f[ch] = (lambda m: (lambda x: float(m.predict([[x]]))))(lin)
        out_p[ch] = (a, b)
    return out_f, out_p

# ----------------------------------------------------------------------
# 把四個一次項函式相加，並 clip 在 [y_min, y_max]
# ----------------------------------------------------------------------
def sum_brightness_func(
    params: dict[str, tuple],          
    y_min: float,
    y_max: float
):
    """
    依 4 組 (coef, intercept) 組成總亮度函式，
    並在回傳值階段 clip 至 [y_min, y_max]。
    額外回傳可讀文字公式 (含各通道截距)。
    """
    a_w, b_w = params["W"]
    a_r, b_r = params["R"]
    a_g, b_g = params["G"]
    a_b, b_b = params["B"]
    inter_total = b_w + b_r + b_g + b_b

    # 文字公式只保留非零係數
    terms_txt = []
    if a_w: terms_txt.append(f"{a_w:.4g}·W")
    if a_r: terms_txt.append(f"{a_r:.4g}·R")
    if a_g: terms_txt.append(f"{a_g:.4g}·G")
    if a_b: terms_txt.append(f"{a_b:.4g}·B")

    coef_txt = " + ".join(terms_txt) if terms_txt else "0"
    intercept_parts = []
    if b_w: intercept_parts.append(f"W:{b_w:.4g}")
    if b_r: intercept_parts.append(f"R:{b_r:.4g}")
    if b_g: intercept_parts.append(f"G:{b_g:.4g}")
    if b_b: intercept_parts.append(f"B:{b_b:.4g}")
    intercept_txt = ", ".join(intercept_parts) or "0"

    func_expr = (
        f"y = {coef_txt} + {inter_total:.4g} "
        f"({intercept_txt}) [clipped {y_min}–{y_max}]"
    )
    # 數值函式
    def _f(w, r, g, b):
        val = a_w*w + a_r*r + a_g*g + a_b*b + inter_total
        return max(y_min, min(y_max, val))

    return _f, func_expr

# ------------------------------------------------------------
#  隨機抽樣工具 -------------------------------------------
# ------------------------------------------------------------
def sample_wrbg_combos(
    params: dict[str, tuple],   # {"W": (a,b), ...}
    w_max, r_max, g_max, b_max,
    br_min, br_max,
    sample_cnt,
    rng_seed=None,
    max_attempt_factor=20,
):
    """
    隨機抽樣 WRGB；用「完整一次項公式 + 四通道截距」做篩選，
    要求 y_raw 介於 [br_min, br_max]。
    回傳 List[[W,R,G,B]]，長度 = sample_cnt
    """
    a_w, b_w = params["W"]
    a_r, b_r = params["R"]
    a_g, b_g = params["G"]
    a_b, b_b = params["B"]
    intercept_sum = b_w + b_r + b_g + b_b

    rng        = np.random.default_rng(rng_seed)
    N          = int(sample_cnt)
    combos     = []
    attempts   = 0
    max_tries  = N * max_attempt_factor

    while len(combos) < N and attempts < max_tries:
        attempts += 1
        cw = rng.integers(0, w_max + 1)
        cr = rng.integers(0, r_max + 1)
        cg = rng.integers(0, g_max + 1)
        cb = rng.integers(0, b_max + 1)

        y_raw = (
            a_w * cw + a_r * cr + a_g * cg + a_b * cb + intercept_sum
        )

        if br_min <= y_raw <= br_max:
            combos.append([int(cw), int(cr), int(cg), int(cb)])

    if len(combos) < N:
        raise ValueError(
            f"嘗試 {attempts} 次仍僅找到 {len(combos)} 組；"
            f"請放寬亮度範圍或提高通道上限"
        )
    return combos


def generate_sampling_json(
    sk_file,
    model_div,              # Gradio File 物件 (骨架 JSON)
    w_csv, r_csv, g_csv, b_csv,  # 四通道 CSV (Gradio File)
    w_max, r_max, g_max, b_max,  # 四通道強度上限 (Number)
    br_min, br_max,              # Brightness 範圍
    sample_cnt,                  # 抽樣組數
):
    """隨機抽樣 WRGB 組合並寫入骨架 JSON。

    步驟：
    1. 讀四個 CSV → 以自動階數訓練四個 brightness 函式。
    2. 將四函式加總 → comb_func(w,r,g,b)，並於呼叫時 clip 到 br_min, br_max。
    3. 在 4-維區間 (0~各自 max) 隨機抽 sample_cnt 組整數強度；
       若 comb_func 落在 [br_min,br_max] 就保留。
    4. 把保留組合塞進骨架 JSON 的第一個含 scenes 的節點，並輸出新 JSON。
    """

    # ---- 1. 一次項模型 & 亮度公式 ----------------------
    funcs, params = train_channel_models(
        w_csv.name if w_csv else None,
        r_csv.name if r_csv else None,
        g_csv.name if g_csv else None,
        b_csv.name if b_csv else None,
    )
    comb_func, func_expr = sum_brightness_func(
        params, y_min=float(br_min), y_max=float(br_max)
    )

    # 若通道缺失 → 對應 max 一律設 0，避免抽到非 0
    w_max = 0 if w_csv is None else int(w_max)
    r_max = 0 if r_csv is None else int(r_max)
    g_max = 0 if g_csv is None else int(g_max)
    b_max = 0 if b_csv is None else int(b_max)

    # ---- 2. 隨機抽樣 -----------------------------------
    try:
        combos = sample_wrbg_combos(
            params,
            w_max=int(w_max), r_max=int(r_max),
            g_max=int(g_max), b_max=int(b_max),
            br_min=float(br_min), br_max=float(br_max),
            sample_cnt=int(sample_cnt)
        )
    except ValueError as err:           # ← 只抓這一種
        func_expr += f"\n⚠️ {err}"      #   把訊息加到同一行文字
        # 直接回傳：JSON 為 None、公式文字含警示
        return None, func_expr
    
    # ---- 3. 3. 讀骨架 JSON & 寫入 scenes --------------------------------
    with open(sk_file.name, encoding="utf-8") as f:
        data = json.load(f)

    new_data = {"device": data.get("device", {})}

    # 依 model_div 從 KEY_MAP 找節點
    key = KEY_MAP.get(model_div, model_div)
    node = data.get(key)
    if node is None:
        raise KeyError(f"骨架 JSON 找不到 key={key}")

    scenes = [
        {
            "brightness": 1024,
            "colors": combo,
            "currentZone": 0,
            "zoneMode": 0,
        }
        for combo in combos
    ]

    # 深拷貝替換 scenes
    if isinstance(node, list):
        new_node = [
            {**{k: v for k, v in ent.items() if k != "scenes"}, "scenes": scenes}
            for ent in node
        ]
    else:
        new_node = {**{k: v for k, v in node.items() if k != "scenes"}, "scenes": scenes}

    new_data[key] = new_node

    # ---------- 4. 輸出 ----------------------------------
    out_json = Path(tempfile.mkdtemp()) / f"wrgb_sampling_{key}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    return str(out_json), func_expr


###################################
# 建立 Gradio 介面
###################################
with gr.Blocks(title="自動打光JSON生成器") as demo:
    with gr.Tabs():
        # 第一個頁籤：LightPara JSON 生成器
        with gr.TabItem("自動打光矩陣生成"):  
            sk_file = gr.File(label="📄 上傳骨架 JSON")
            b_cnt = gr.Dropdown([1,2,3], value=2, label="亮度數量")
            c_cnt = gr.Dropdown([1,2,3], value=3, label="顏色強度數量")
            b_boxes, c_boxes, tabs = [], [], []
            with gr.Tabs():
                for m in MODELS:
                    with gr.TabItem(m):
                        b = gr.Textbox(label="亮度列表", value=",".join(BRIGHTNESS_BASE[:2]))
                        c = gr.Textbox(label="顏色列表", value=",".join(COLOR_BASE[:3]))
                        b_boxes.append(b)
                        c_boxes.append(c)
            # 同步函式
            def _build(base, count):
                vals = base[:count] + [base[-1]] * max(0, count-len(base))
                return ",".join(vals)
            def sync_all(bn, cn):
                updates = []
                for _ in b_boxes:
                    updates.append(gr.update(value=_build(BRIGHTNESS_BASE, bn)))
                for _ in c_boxes:
                    updates.append(gr.update(value=_build(COLOR_BASE, cn)))
                return updates
            def sync_single(bn, cn, idx):
                return gr.update(value=_build(BRIGHTNESS_BASE, bn)), gr.update(value=_build(COLOR_BASE, cn))
            b_cnt.change(sync_all, [b_cnt, c_cnt], b_boxes + c_boxes)
            c_cnt.change(sync_all, [b_cnt, c_cnt], b_boxes + c_boxes)
            for i, tab in enumerate(tabs):
                tab.select(lambda bn, cn, i=i: sync_single(bn, cn, i), [b_cnt, c_cnt], [b_boxes[i], c_boxes[i]])
            demo.load(sync_all, [b_cnt, c_cnt], b_boxes + c_boxes)
            gen_btn = gr.Button("生成 JSON")
            out_file = gr.File(label="⬇️ 下載 lightPara.json")
            interleaved = [v for pair in zip(b_boxes, c_boxes) for v in pair]
            gen_btn.click(create_json, [sk_file, b_cnt, c_cnt] + interleaved, out_file)

        # 新增第二頁：單色光生成
        with gr.TabItem("單色光生成"):
            sk2 = gr.File(label="📄 上傳骨架 JSON")
            model_sel = gr.Dropdown(MODELS, label="選擇模型")
            channel_sel = gr.Dropdown(['W','R','G','B'], label="選擇光源 (W/R/G/B)")
            gen_btn = gr.Button("生成單色光 JSON")
            out_file = gr.File(label="⬇️ 下載 JSON")
            gen_btn.click(
                fn=create_single_color_json,
                inputs=[sk2, model_sel, channel_sel],
                outputs=out_file
            )

        # 新增第三頁：等分組合生成
        with gr.TabItem("WRGB 等分組合生成"):
            sk_div = gr.File(label="📄 上傳骨架 JSON")
            model_div = gr.Dropdown(MODELS, label="選擇模型")
            w_max = gr.Number(label="W(Max)")
            r_max = gr.Number(label="R(Max)")
            g_max = gr.Number(label="G(Max)")
            b_max = gr.Number(label="B(Max)")
            divisions = gr.Number(label="等分數量", value=1, precision=0)
            gen_div_btn = gr.Button("生成等分組合JSON")
            out_div_file = gr.File(label="⬇️ 下載 JSON")
            gen_div_btn.click(
                fn=create_division_color_json,
                inputs=[sk_div, model_div, w_max, r_max, g_max, b_max, divisions],
                outputs=out_div_file
            )
        # 新增第四頁：WRGB 抽樣打光矩陣生成
        with gr.TabItem("WRGB 抽樣打光矩陣"):
            # 1) 骨架 JSON
            sk_in = gr.File(label="📄 上傳骨架 JSON", file_types=[".json"])
            model_div = gr.Dropdown(MODELS, label="選擇模型")
            # 2) 四個通道的 CSV
            with gr.Row():
                w_csv = gr.File(label="W 通道 CSV", file_types=[".csv"])
                r_csv = gr.File(label="R 通道 CSV", file_types=[".csv"])
                g_csv = gr.File(label="G 通道 CSV", file_types=[".csv"])
                b_csv = gr.File(label="B 通道 CSV", file_types=[".csv"])

            # 3) 四個通道的強度上限 (0–1024，各自可不同)
            with gr.Row():
                w_max = gr.Number(label="W 上限", value=1024, precision=0)
                r_max = gr.Number(label="R 上限", value=1024, precision=0)
                g_max = gr.Number(label="G 上限", value=1024, precision=0)
                b_max = gr.Number(label="B 上限", value=1024, precision=0)

            # 4) Brightness (Br) 的允許範圍
            with gr.Row():
                br_min = gr.Number(label="Br 最小值", value=0)
                br_max = gr.Number(label="Br 最大值", value=1024)

            # 5) 4-維空間內的抽樣數量
            sample_cnt = gr.Number(label="抽樣組數 (N)", value=200, precision=0)

            # 生成 JSON 按鈕 & 下載元件
            gen_btn  = gr.Button("生成抽樣 JSON")
            out_json = gr.File(label="⬇️ 下載 JSON")
            func_box = gr.Textbox(label="合成亮度函式", lines=2)

            gen_btn.click(
                fn=generate_sampling_json,          # 你在 wrgb_sampler.py 內實作的新函式
                inputs=[
                    sk_in,model_div, w_csv, r_csv, g_csv, b_csv,
                    w_max, r_max, g_max, b_max,
                    br_min, br_max,
                    sample_cnt
                ],
                outputs=[out_json, func_box]
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)

