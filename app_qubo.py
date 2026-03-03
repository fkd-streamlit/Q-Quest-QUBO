# -*- coding: utf-8 -*-
# app_qubo.py
# Q-Quest 量子神託（STAGE×QUOTES + 黒基調UI + 表/グラフ黒 + 緑/青パネル継承）
#
# 前提:
# - 統合Excel(pack): VOW_DICT / CHAR_TO_VOW / CHAR_MASTER / (任意) STAGE_DICT / STAGE_TO_AXIS / QUOTES
# - 画像: ./assets/images/characters/ など（CHAR_TO_VOW の IMAGE_FILE を参照）
#
# 目的:
# - UIを黒基調に統一（入力/アップロード含む）
# - 表（st.dataframe）を黒背景に
# - グラフを黒背景に（Plotlyで制御）
# - 格言(薄緑×濃緑) / QUOTES(濃青×青) の雰囲気を継承
# - 12神ギャラリーは不要（出さない）

import os
import re
import math
import random
from pathlib import Path
from collections import Counter
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go


# ============================================================
# 0) Streamlit config（必ず最初）
# ============================================================
st.set_page_config(page_title="Q-Quest 量子神託（QUBO / STAGE×QUOTES）", layout="wide")


# ============================================================
# 1) Dark UI CSS（ページ/サイドバー/入力/アップロード/表/ボタン）
# ============================================================
DARK_CSS = """
<style>
/* ---- base ---- */
:root{
  --bg0:#050814;
  --bg1:#070b1a;
  --bg2:#0b1020;
  --card:#0b1224;
  --card2:#0e1730;
  --line:rgba(255,255,255,0.10);
  --txt:#e9eef7;
  --muted:rgba(233,238,247,0.70);
  --accent:#ff4d5a;
  --accent2:#8bb7ff;
}

/* app background */
html, body, [data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 600px at 25% 0%, #0c1230 0%, var(--bg0) 55%, #030513 100%) !important;
  color: var(--txt) !important;
}

/* sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #06081a 0%, #040612 100%) !important;
  border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"] *{
  color: var(--txt) !important;
}

/* titles */
h1, h2, h3, h4, h5, h6 { color: var(--txt) !important; }
p, li, span, label, div { color: var(--txt); }
small, .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

/* cards / containers */
.block-container{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}

/* buttons */
.stButton>button{
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: linear-gradient(90deg, rgba(255,77,90,0.95), rgba(255,77,90,0.65)) !important;
  color: white !important;
  box-shadow: 0 10px 24px rgba(0,0,0,0.35) !important;
}
.stButton>button:hover{
  filter: brightness(1.05);
}

/* sliders */
[data-testid="stSlider"]{
  padding: 0.1rem 0.2rem;
}
[data-testid="stSlider"] *{
  color: var(--txt) !important;
}

/* ---- INPUTS: text input / textarea / select ---- */
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div{
  background: rgba(11,16,32,0.92) !important;
  color: var(--txt) !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  border-radius: 12px !important;
}
.stTextArea textarea::placeholder{
  color: rgba(233,238,247,0.45) !important;
}

/* select dropdown menu */
div[data-baseweb="popover"]{
  background: rgba(11,16,32,0.98) !important;
}
div[data-baseweb="menu"]{
  background: rgba(11,16,32,0.98) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
}
div[data-baseweb="menu"] *{ color: var(--txt) !important; }

/* ---- FILE UPLOADER ---- */
[data-testid="stFileUploader"] section{
  background: rgba(11,16,32,0.92) !important;
  border: 1px dashed rgba(255,255,255,0.22) !important;
  border-radius: 14px !important;
}
[data-testid="stFileUploader"] section *{
  color: var(--txt) !important;
}
[data-testid="stFileUploaderDropzone"]{
  background: rgba(11,16,32,0.92) !important;
}

/* ---- dataframe container background ---- */
[data-testid="stDataFrame"]{
  background: rgba(11,16,32,0.92) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
  overflow: hidden !important;
}

/* expander */
details{
  background: rgba(11,16,32,0.72) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
  padding: 6px 10px !important;
}
details summary{
  color: var(--txt) !important;
}

/* ---- custom panels ---- */
.panel{
  border-radius: 16px;
  padding: 14px 16px;
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 14px 34px rgba(0,0,0,0.35);
}
.panel-green{
  background: linear-gradient(180deg, rgba(175,255,205,0.22) 0%, rgba(78,196,120,0.14) 100%);
  border-color: rgba(120,255,180,0.22);
}
.panel-green .t{
  color: rgba(70,255,150,0.95);
  font-weight: 700;
}
.panel-green .b{
  color: rgba(170,255,210,0.92);
  line-height: 1.6;
}

.panel-blue{
  background: linear-gradient(180deg, rgba(80,160,255,0.18) 0%, rgba(25,70,140,0.18) 100%);
  border-color: rgba(140,190,255,0.22);
}
.panel-blue .t{
  color: rgba(140,190,255,0.96);
  font-weight: 700;
}
.panel-blue .b{
  color: rgba(175,210,255,0.92);
  line-height: 1.6;
}

/* code block */
pre, code{
  background: rgba(11,16,32,0.92) !important;
  color: var(--txt) !important;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ============================================================
# 2) Constants / helpers
# ============================================================
VOW_COLS = [f"VOW_{i:02d}" for i in range(1, 13)]
AXIS_COLS = ["AXIS_SEI", "AXIS_RYU", "AXIS_MA", "AXIS_MAKOTO"]


def _safe_str(x) -> str:
    return "" if pd.isna(x) else str(x)


def _normalize01(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    mn, mx = float(np.min(v)), float(np.max(v))
    if mx - mn < 1e-12:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn)


def _softmax(x: np.ndarray, t: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    z = np.exp(x / max(t, 1e-9))
    s = np.sum(z)
    return z / s if s > 0 else np.ones_like(z) / len(z)


def _japanese_tokens_simple(text: str) -> List[str]:
    """
    “気持ち悪い断片”が出にくい簡易抽出:
    - 日本語文字(漢字/ひら/カタカナ)の連続(>=2)のみ
    - よくある語をstop扱い
    """
    text = _safe_str(text)
    text = re.sub(r"[0-9０-９]", " ", text)
    cands = re.findall(r"[一-龥ぁ-んァ-ヴー]{2,}", text)
    stop = set([
        "したい", "した", "たい", "です", "ます", "いる", "ある", "なる",
        "こと", "もの", "ため", "よう", "から", "ので", "でも", "しかし",
        "そして", "それ", "これ", "あれ", "私", "自分", "あなた"
    ])
    out = []
    for w in cands:
        if w in stop:
            continue
        out.append(w)
    return out


def _char_ngrams(text: str, n: int = 3) -> Counter:
    s = _safe_str(text)
    s = re.sub(r"\s+", "", s)
    if len(s) < n:
        return Counter()
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def _cosine_from_counters(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    keys = set(a.keys()) | set(b.keys())
    va = np.array([a.get(k, 0.0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0.0) for k in keys], dtype=float)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def load_image(path: str):
    try:
        from PIL import Image
        if not path or not os.path.exists(path):
            return None
        return Image.open(path)
    except Exception:
        return None


def dark_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """st.dataframe に渡す黒基調のStyler"""
    return (
        df.style
        .set_properties(**{
            "background-color": "#0b1020",
            "color": "#e9eef7",
            "border-color": "rgba(255,255,255,0.10)",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#0e1730"),
                ("color", "#e9eef7"),
                ("border-color", "rgba(255,255,255,0.10)"),
                ("font-weight", "700"),
            ]},
            {"selector": "td", "props": [
                ("border-color", "rgba(255,255,255,0.08)"),
            ]},
        ])
    )


def plotly_line_vow(vow_ids: List[str], manual: np.ndarray, auto: np.ndarray, mix: np.ndarray):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vow_ids, y=auto, mode="lines+markers", name="auto"))
    fig.add_trace(go.Scatter(x=vow_ids, y=manual, mode="lines+markers", name="manual"))
    fig.add_trace(go.Scatter(x=vow_ids, y=mix, mode="lines+markers", name="mix"))
    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=20, r=20, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-90),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plotly_bar(names: List[str], values: np.ndarray, title: str, height: int = 380):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=values, name=title))
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=40, b=70),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-90),
        title=title,
    )
    return fig


# ============================================================
# 3) Excel loader
# ============================================================
@st.cache_data(show_spinner=False)
def load_pack_from_uploader(uploaded_file) -> Dict[str, pd.DataFrame]:
    import io
    bio = io.BytesIO(uploaded_file.getvalue())
    xl = pd.ExcelFile(bio)
    sheets = {}
    for name in xl.sheet_names:
        sheets[name] = pd.read_excel(bio, sheet_name=name)
        bio.seek(0)
    return sheets


# ============================================================
# 4) Sidebar
# ============================================================
st.title("🔮 Q-Quest 量子神託（QUBO / STAGE×QUOTES）")

with st.sidebar:
    st.header("📁 データ")
    pack_file = st.file_uploader("統合Excel（pack）", type=["xlsx"])

    st.header("🖼️ 画像フォルダ")
    img_dir = st.text_input("画像フォルダ（相対/絶対）", value="./assets/images/characters")

    st.header("📝 テキスト＋誓願（自動ベクトル化）")
    ngram_n = st.selectbox("n-gram", [2, 3, 4], index=1)
    alpha_text = st.slider("mix比率 α（1=テキスト寄り / 0=スライダーのみ）", 0.0, 1.0, 0.55, 0.01)

    st.header("🗣️ QUOTES神託（温度付き選択）")
    lang = st.selectbox("LANG", ["ja", "en", ""], index=0, help="空は全言語")
    quote_temp = st.slider("格言温度（高→ランダム / 低→上位固定）", 0.2, 3.0, 1.2, 0.1)

    st.header("🌙 季節×時間（Stage）")
    use_stage_auto = st.checkbox("現在時刻から自動推定", value=True)

    st.header("⚙️ QUBO設定（one-hot）")
    # ※ここはあなたの既存ロジックに合わせて「見せ方」を維持
    penalty = st.slider("one-hot ペナルティ P", 1.0, 200.0, 40.0, 1.0)
    num_reads = st.slider("サンプル数（観測分布）", 50, 600, 300, 10)
    sweeps = st.slider("SA sweeps", 50, 1200, 420, 10)
    temperature = st.slider("SA温度（大→揺らぐ）", 0.2, 3.0, 1.2, 0.1)

    run_btn = st.button("🧪 観測する（QUBOから抽出）", use_container_width=True)


# ============================================================
# 5) Load pack
# ============================================================
if pack_file is None:
    st.info("左のサイドバーから **統合Excel（pack）** をアップロードしてください。")
    st.stop()

try:
    pack = load_pack_from_uploader(pack_file)
except Exception as e:
    st.error(f"Excelの読み込みに失敗しました: {e}")
    st.stop()

required = ["VOW_DICT", "CHAR_TO_VOW", "CHAR_MASTER"]
missing = [k for k in required if k not in pack]
if missing:
    st.error(f"必須シートが不足しています: {missing}\n検出したシート: {list(pack.keys())}")
    st.stop()

df_vow = pack["VOW_DICT"].copy()
df_ctv = pack["CHAR_TO_VOW"].copy()
df_cm = pack["CHAR_MASTER"].copy()
df_stage_dict = pack.get("STAGE_DICT")
df_stage_to_axis = pack.get("STAGE_TO_AXIS")
df_quotes = pack.get("QUOTES")


# ============================================================
# 6) Validate columns / build maps
# ============================================================
if "VOW_ID" not in df_vow.columns or "TITLE" not in df_vow.columns:
    st.error("VOW_DICT に VOW_ID/TITLE 列がありません。")
    st.stop()

for c in ["CHAR_ID", "公式キャラ名"]:
    if c not in df_ctv.columns:
        st.error(f"CHAR_TO_VOW に {c} 列がありません。")
        st.stop()

for c in VOW_COLS:
    if c not in df_ctv.columns:
        st.error(f"CHAR_TO_VOW に {c} 列がありません。")
        st.stop()

img_col = "IMAGE_FILE" if "IMAGE_FILE" in df_ctv.columns else None

vow_ids = df_vow["VOW_ID"].astype(str).tolist()[:12]
vow_title_map = dict(zip(df_vow["VOW_ID"].astype(str), df_vow["TITLE"].astype(str)))

# “意味が分かる”ラベル（TITLE + SUBTITLE + LR等）
vow_label_map = {}
for i in range(min(12, len(df_vow))):
    r = df_vow.iloc[i]
    vid = str(r.get("VOW_ID", f"VOW_{i+1:02d}"))
    title = _safe_str(r.get("TITLE", ""))
    subtitle = _safe_str(r.get("SUBTITLE", ""))

    left_label = ""
    right_label = ""
    for cand in ["LEFT_LABEL", "LEFT", "LEFT_TEXT"]:
        if cand in df_vow.columns:
            left_label = _safe_str(r.get(cand, ""))
            break
    for cand in ["RIGHT_LABEL", "RIGHT", "RIGHT_TEXT"]:
        if cand in df_vow.columns:
            right_label = _safe_str(r.get(cand, ""))
            break

    if left_label or right_label:
        vow_label_map[vid] = f"{title}  ←{left_label}｜{right_label}→"
    else:
        vow_label_map[vid] = f"{title} — {subtitle}".strip(" —")

# character
char_ids = df_ctv["CHAR_ID"].astype(str).tolist()
char_names = df_ctv["公式キャラ名"].astype(str).tolist()
W_char_vow = df_ctv[VOW_COLS].to_numpy(dtype=float)

# axis from CHAR_MASTER (optional)
W_char_axis = None
if all(c in df_cm.columns for c in (["CHAR_ID"] + AXIS_COLS)):
    cm_map = df_cm.set_index("CHAR_ID")[AXIS_COLS].to_dict(orient="index")
    axis_mat = []
    for cid in char_ids:
        row = cm_map.get(cid, {})
        axis_mat.append([float(row.get(c, 0.0)) for c in AXIS_COLS])
    W_char_axis = np.array(axis_mat, dtype=float)


# ============================================================
# 7) Layout: Step1 input (left) / Step3 results (right)
# ============================================================
left, right = st.columns([1.25, 1.0], gap="large")

with left:
    st.subheader("Step 1：誓願入力（スライダー）＋テキスト（自動ベクトル化）")

    user_text = st.text_area(
        "あなたの状況を一文で（例：疲れていて決断ができない / 新しい挑戦が怖い など）",
        height=90,
        placeholder="例：迷いを断ちたいが、今は焦らず機を待つべきか悩んでいる…"
    )

    st.caption("スライダー入力はTITLEを常時表示し、テキストからの自動推定と mix します。")

    manual = np.zeros(12, dtype=float)
    for i in range(12):
        vid = vow_ids[i] if i < len(vow_ids) else f"VOW_{i+1:02d}"
        label = vow_label_map.get(vid, vid)
        manual[i] = st.slider(
            f"{vid}｜{label}",
            0.0, 5.0, 0.0, 0.5
        )
    manual = manual / 5.0  # 0..1

    # auto vector from text
    keywords = _japanese_tokens_simple(user_text)
    c_user = _char_ngrams(user_text, n=int(ngram_n))
    vow_texts = []
    for i in range(12):
        r = df_vow.iloc[i]
        vow_texts.append(" ".join([
            _safe_str(r.get("LABEL", "")),
            _safe_str(r.get("TITLE", "")),
            _safe_str(r.get("SUBTITLE", "")),
            _safe_str(r.get("DESCRIPTION_LONG", "")),
            _safe_str(r.get("UI_HINT", "")),
        ]))

    auto = np.zeros(12, dtype=float)
    for i in range(12):
        c_ref = _char_ngrams(vow_texts[i], n=int(ngram_n))
        auto[i] = _cosine_from_counters(c_user, c_ref)
    auto = _normalize01(auto)

    mix = (1.0 - alpha_text) * manual + alpha_text * auto
    mix = np.clip(mix, 0.0, 1.0)

    with st.expander("🔎 誓願ベクトル（manual / auto / mix）"):
        df_vec = pd.DataFrame({
            "VOW": vow_ids,
            "manual": np.round(manual, 3),
            "auto": np.round(auto, 3),
            "mix": np.round(mix, 3),
        })
        st.dataframe(dark_table(df_vec), use_container_width=True, hide_index=True)


# ============================================================
# 8) Stage select + axis
# ============================================================
stage_axis = np.zeros(4, dtype=float)
stage_axis_label = ""

stage_id = None
if df_stage_dict is not None and df_stage_to_axis is not None and "STAGE_ID" in df_stage_dict.columns:
    # stage label map
    stage_labels = df_stage_dict["LABEL"].astype(str).tolist() if "LABEL" in df_stage_dict.columns else df_stage_dict["STAGE_ID"].astype(str).tolist()
    stage_map = dict(zip(stage_labels, df_stage_dict["STAGE_ID"].astype(str).tolist()))
    # default
    default_label = stage_labels[0] if stage_labels else ""

    if use_stage_auto:
        # “自動推定”はあなたのExcel設計に依存するので、ここは「先頭」を仮の既定にしています。
        # 必要なら STAGE_DICT 側に "AUTO_HINT" などを作ってマッピングできます。
        stage_axis_label = default_label
    else:
        stage_axis_label = st.selectbox("STAGE_ID（手動上書き可）", stage_labels, index=0)

    stage_id = stage_map.get(stage_axis_label)

    if stage_id is not None and "STAGE_ID" in df_stage_to_axis.columns:
        row_axis = df_stage_to_axis[df_stage_to_axis["STAGE_ID"].astype(str) == str(stage_id)]
        if len(row_axis) > 0:
            r = row_axis.iloc[0]
            stage_axis = np.array([float(r.get(c, 0.0)) for c in AXIS_COLS], dtype=float)


# ============================================================
# 9) Energy (score -> energy) + sampling (existing style)
# ============================================================
# score from vows
score_vow = W_char_vow @ mix  # (12,)
score_axis = 0.0
if W_char_axis is not None and np.linalg.norm(stage_axis) > 1e-12:
    s = W_char_axis @ stage_axis
    s = _normalize01(s) * 0.25
    score_axis = s

score = score_vow + score_axis

# tiny noise
rng = np.random.default_rng(0)
score = score + rng.normal(0, 0.002, size=score.shape)

# energy smaller => better
energy = -score

# probabilities
p_char = _softmax(-energy, t=max(temperature, 1e-9))

# sample distribution (by p_char)
sample_idxs = np.random.choice(len(char_ids), size=int(num_reads), replace=True, p=p_char)

# observe single
observed_idx = int(np.random.choice(len(char_ids), size=1, p=p_char)[0])


# ============================================================
# 10) QUOTES pick (temperature)
# ============================================================
quote_choice = {}
quote_top = pd.DataFrame()

if df_quotes is not None and len(df_quotes) > 0:
    q = df_quotes.copy()
    # filter lang
    if "LANG" in q.columns and lang != "":
        q = q[q["LANG"].astype(str) == str(lang)]

    # build score:
    # - prefer observed char id
    # - prefer top vow id by mix
    top_vow_idx = int(np.argmax(mix))  # 0..11
    top_vow_id = vow_ids[top_vow_idx] if top_vow_idx < len(vow_ids) else f"VOW_{top_vow_idx+1:02d}"

    user_ng = _char_ngrams(user_text, n=int(ngram_n))

    scores = []
    for _, r in q.iterrows():
        s = 0.0
        q_char = _safe_str(r.get("CHAR_ID", ""))
        q_vow = _safe_str(r.get("VOW_ID", ""))
        txt = _safe_str(r.get("QUOTE", ""))

        if q_char and q_char == char_ids[observed_idx]:
            s += 2.5
        if q_vow and q_vow == top_vow_id:
            s += 1.2

        # weak text sim
        q_ng = _char_ngrams(txt, n=int(ngram_n))
        s += 0.6 * _cosine_from_counters(user_ng, q_ng)
        scores.append(s)

    q["SCORE"] = scores
    q = q.sort_values("SCORE", ascending=False)
    quote_top = q.copy()

    # temperature selection
    probs = _softmax(q["SCORE"].to_numpy(dtype=float), t=max(quote_temp, 1e-9))
    pick_i = int(np.random.choice(len(q), size=1, p=probs)[0])
    quote_choice = q.iloc[pick_i].to_dict()


quote_text = _safe_str(quote_choice.get("QUOTE", "")).strip()
quote_source = _safe_str(quote_choice.get("SOURCE", "")).strip()


# ============================================================
# 11) Build oracle lines (green panel)
# ============================================================
# top contributing vows
contrib = (W_char_vow[observed_idx] * mix)  # (12,)
top_idx = np.argsort(contrib)[::-1][:6]
top_vow_ids = [vow_ids[i] for i in top_idx]
top_titles_txt = "・".join([vow_title_map.get(v, v) for v in top_vow_ids[:3]])

oracle_lines = []
if user_text.strip():
    oracle_lines.append(f"「{user_text.strip()}」の奥に、**{top_titles_txt}** が見えている。")
else:
    oracle_lines.append(f"いまの波は **{top_titles_txt}** に寄っている。")

if stage_axis_label:
    oracle_lines.append(f"季節×時間の気配（Stage）は **{stage_axis_label}** を強める。")

if quote_text:
    oracle_lines.append(f"格言：『{quote_text}』")
    if quote_source:
        oracle_lines.append(f"— {quote_source}")

oracle = "<br>".join(oracle_lines)


# ============================================================
# 12) Right column outputs
# ============================================================
with right:
    st.subheader("Step 3：結果（観測された神＋理由＋QUOTES神託）")

    # Table of top 3 by energy
    rank_idx = np.argsort(energy)[:3]
    rank_df = pd.DataFrame({
        "順位": [1, 2, 3],
        "CHAR_ID": [char_ids[i] for i in rank_idx],
        "神": [char_names[i] for i in rank_idx],
        "energy（低いほど選ばれやすい）": [float(np.round(energy[i], 4)) for i in rank_idx],
        "確率p（softmax）": [float(np.round(p_char[i], 4)) for i in rank_idx],
    })
    st.dataframe(dark_table(rank_df), use_container_width=True, hide_index=True)

    observed_char_id = char_ids[observed_idx]
    observed_char_name = char_names[observed_idx]

    st.markdown(f"### 🌟 今回“観測”された神：**{observed_char_name}**（{observed_char_id}）")
    st.caption(
        "※ここは「単発の観測（1回抽選）」です。下の📊観測分布（サンプル）は「同条件で何回も観測したらどう出るか」のヒストグラムです。"
    )

    # Image
    observed_img_file = _safe_str(df_ctv.iloc[observed_idx].get(img_col, "")) if img_col else ""
    img_path = os.path.join(img_dir, observed_img_file) if observed_img_file else ""
    img = load_image(img_path)
    if img is not None:
        st.image(img, caption=f"{observed_char_name}（{observed_img_file}）", use_container_width=True)
    else:
        st.warning(f"画像が見つかりません: {img_path}")

    # Green oracle panel (継承: 薄緑×濃緑)
    st.markdown(
        f"""
        <div class="panel panel-green">
          <div class="t">🟩 格言（雰囲気：薄緑 × 濃い緑）</div>
          <div class="b" style="margin-top:8px;">{oracle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Contrib table (dark)
    char_w = W_char_vow[observed_idx]
    v_mix_map = {vow_ids[i]: float(mix[i] * 5.0) for i in range(12)}  # 以前っぽく 0..5 スケール表示
    contrib_df = pd.DataFrame({
        "VOW": top_vow_ids,
        "TITLE": [vow_title_map.get(v, v) for v in top_vow_ids],
        "mix(v)": [float(np.round(v_mix_map.get(v, 0.0), 3)) for v in top_vow_ids],
        "W(char,v)": [float(np.round(char_w[vow_ids.index(v)], 3)) for v in top_vow_ids],
        "寄与(v*w)": [float(np.round(contrib[vow_ids.index(v)], 3)) for v in top_vow_ids],
    })
    st.markdown("#### 🧩 寄与した誓願（Top）")
    st.dataframe(dark_table(contrib_df), use_container_width=True, hide_index=True)

    # QUOTES block (継承: 青×濃青文字)
    st.markdown("#### 🗣️ QUOTES神託（温度付きで選択）")
    if quote_text:
        blue_body = f"『{quote_text}』"
        if quote_source:
            blue_body += f"<br>— {quote_source}"
        st.markdown(
            f"""
            <div class="panel panel-blue">
              <div class="t">🟦 QUOTES神託（雰囲気：濃い青 × 青文字）</div>
              <div class="b" style="margin-top:8px;">{blue_body}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("🔎 格言候補Top（デバッグ）"):
            show_cols = [c for c in ["QUOTE_ID", "QUOTE", "SOURCE", "LANG", "CHAR_ID", "VOW_ID", "SCORE"] if c in quote_top.columns]
            if show_cols:
                st.dataframe(dark_table(quote_top[show_cols].head(10)), use_container_width=True, hide_index=True)
            else:
                st.caption("表示可能な列がありません。")
    else:
        st.warning("QUOTESから格言が選べませんでした（LANGフィルタやシート内容を確認してください）。")


# ============================================================
# 13) Visualizations (bottom) - dark plotly
# ============================================================
st.divider()
st.subheader("📊 可視化：テキストの影響・観測分布・エネルギー地形")

colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.markdown("### 1) テキスト→誓願 自動推定の影響（auto vs manual vs mix）")
    st.caption("auto（テキスト由来）と manual（スライダー）と mix の差が見える化されます。")
    fig1 = plotly_line_vow(vow_ids, manual * 5.0, auto * 5.0, mix * 5.0)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 2) エネルギー地形（全候補）")
    order = np.argsort(energy)
    names_sorted = [char_names[i] for i in order]
    energy_sorted = energy[order]
    fig2 = plotly_bar(names_sorted, energy_sorted, title="energy（低いほど選ばれやすい）", height=420)
    st.plotly_chart(fig2, use_container_width=True)

with colB:
    st.markdown("### 3) 観測分布（サンプル）")
    cnt = Counter(sample_idxs.tolist())
    hist_names = [char_names[i] for i in range(len(char_names))]
    hist_vals = np.array([cnt.get(i, 0) for i in range(len(char_names))], dtype=float)
    # sort by count desc for readability
    ord2 = np.argsort(-hist_vals)
    fig3 = plotly_bar([hist_names[i] for i in ord2], hist_vals[ord2], title="count（サンプル）", height=420)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 4) テキストのキーワード抽出（簡易）")
    if keywords:
        st.write(" / ".join(keywords[:32]))
    else:
        st.caption("（入力テキストが短い/空のため、キーワードが抽出できません）")

st.caption("© Q-Quest / Quantum Shintaku prototype")
