import gradio as gr
import json
import itertools
import os
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


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)

