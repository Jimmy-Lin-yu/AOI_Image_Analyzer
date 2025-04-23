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

# ---------- Gradio UI ----------
with gr.Blocks(title="LightPara JSON 生成器") as demo:
    sk_file = gr.File(label="📄 上傳骨架 JSON")

    # 亮度、顏色強度數量下拉
    b_cnt = gr.Dropdown([1,2,3], value=2, label="亮度數量",  type="value")
    c_cnt = gr.Dropdown([1,2,3], value=3, label="顏色強度數量", type="value")

    # 建立 Textbox 與 TabItem 引用
    b_boxes, c_boxes, tab_items = [], [], []
    with gr.Tabs():
        for m in MODELS:
            with gr.TabItem(m) as tab:
                tab_items.append(tab)       # 存 TabItem 實例
                b = gr.Textbox(label="亮度列表",
                               value=",".join(BRIGHTNESS_BASE[:2]))
                c = gr.Textbox(label="顏色列表",
                               value=",".join(COLOR_BASE[:3]))
                b_boxes.append(b); c_boxes.append(c)

    # --- 工具函式 -------------------------------------------------
    def _build(base, count):
        vals = base[:count] + [base[-1]]*(count-len(base)) \
               if count > len(base) else base[:count]
        return ",".join(vals)

    def sync_all(b_count, c_count):
        """同時更新 5×亮度與 5×顏色 textbox"""
        return ([gr.update(value=_build(BRIGHTNESS_BASE, b_count))]*len(b_boxes) +
                [gr.update(value=_build(COLOR_BASE,      c_count))]*len(c_boxes))

    def sync_single(b_count, c_count):
        """回傳單一分頁(亮度,顏色)的更新  -> outputs=[b_i, c_i]"""
        return (gr.update(value=_build(BRIGHTNESS_BASE, b_count)),
                gr.update(value=_build(COLOR_BASE,      c_count)))

    # --- 事件綁定 -------------------------------------------------
    # 1) dropdown 改變 → 已渲染的 textbox 全同步
    b_cnt.change(sync_all, inputs=[b_cnt, c_cnt], outputs=b_boxes + c_boxes)
    c_cnt.change(sync_all, inputs=[b_cnt, c_cnt], outputs=b_boxes + c_boxes)

    # 2) 進入分頁時 → 保證該分頁 textbox 再同步一次
    for i, tab in enumerate(tab_items):
        tab.select(sync_single,
                   inputs=[b_cnt, c_cnt],
                   outputs=[b_boxes[i], c_boxes[i]])

    # 3) 首次載入頁面也同步一次
    demo.load(sync_all, inputs=[b_cnt, c_cnt], outputs=b_boxes + c_boxes)

    # 生成 JSON
    gen_btn  = gr.Button("生成 JSON")
    out_file = gr.File(label="⬇️ 下載 lightPara.json")

    # 交錯 inputs: b0,c0,b1,c1,...
    interleaved = [v for pair in zip(b_boxes, c_boxes) for v in pair]
    gen_btn.click(create_json,
                  inputs=[sk_file, b_cnt, c_cnt] + interleaved,
                  outputs=out_file)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)

