# -*- coding: utf-8 -*-
# ============================================================
# Q-Quest 量子神託（QUBO / STAGE×QUOTES） + 「縁の球体（3Dワードアート）」
# - Excel(pack) 1枚で完結: VOW / CHAR / STAGE / QUOTES
# - Step1: 誓願入力（スライダー + テキスト）
# - Step3: QUBO(one-hot制約)で「観測された神」算出 + 理由 + 神託
# - QUOTES神託: 温度付きで近い格言を選択
# - 追加: 4) テキストのキーワード抽出 UI に「縁の球体（3D）」を表示
#
# UI要望反映:
# - 背景/グラフ/表をダーク統一
# - file_uploader / text_input の白地問題をCSSで解消
# - 格言カードは（緑）（青）を継承
# - 12神ギャラリーは不要 → 非表示
# ============================================================

import os
import re
import math
import zlib
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# ------------------------------------------------------------
# Streamlit config (must be first)
# ------------------------------------------------------------
st.set_page_config(page_title="Q-Quest 量子神託（QUBO / STAGE×QUOTES）", layout="wide")

APP_TITLE = "🔮 Q-Quest 量子神託（QUBO / STAGE×QUOTES）"

# ------------------------------------------------------------
# Dark UI CSS
# ------------------------------------------------------------
DARK_CSS = """
<style>
/* ===== App background ===== */
.stApp{
  background:
    radial-gradient(circle at 18% 24%, rgba(110,150,255,0.14), transparent 40%),
    radial-gradient(circle at 78% 68%, rgba(255,160,220,0.10), transparent 48%),
    radial-gradient(circle at 50% 50%, rgba(255,255,255,0.035), transparent 60%),
    linear-gradient(180deg, rgba(6,8,18,1), rgba(10,12,26,1));
}

/* ===== Typography ===== */
.block-container{ padding-top: 1.0rem; }
h1,h2,h3,h4{
  color: rgba(245,245,255,0.95) !important;
  font-weight: 650 !important;
}
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
label, .stCaption{
  color: rgba(245,245,255,0.86) !important;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.07);
  border-right: 1px solid rgba(255,255,255,0.10);
  backdrop-filter: blur(10px);
}

/* ===== Inputs (text/file/select) darken ===== */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input{
  background: rgba(0,0,0,0.35) !important;
  color: rgba(245,245,255,0.92) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
}

div[data-testid="stFileUploader"]{
  background: rgba(0,0,0,0.28) !important;
  border: 1px dashed rgba(255,255,255,0.22) !important;
  border-radius: 14px !important;
  padding: 10px !important;
}
div[data-testid="stFileUploader"] *{
  color: rgba(245,245,255,0.88) !important;
}
div[data-testid="stFileUploader"] button{
  background: rgba(255,255,255,0.10) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  color: rgba(245,245,255,0.92) !important;
  border-radius: 10px !important;
}

div[data-testid="stSelectbox"] > div{
  background: rgba(0,0,0,0.30) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 12px !important;
}

/* ===== Buttons ===== */
.stButton > button{
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  color: rgba(245,245,255,0.92) !important;
  background: rgba(255,255,255,0.10) !important;
}
.stButton > button:hover{
  background: rgba(255,255,255,0.15) !important;
}

/* ===== Card ===== */
.card{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px;
  box-shadow: 0 18px 60px rgba(0,0,0,0.18);
}

/* ===== Quote boxes (継承したい雰囲気) ===== */
.quote-green{
  background: rgba(120, 255, 170, 0.14);
  border: 1px solid rgba(120, 255, 170, 0.24);
  color: rgba(210, 255, 225, 0.95);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 12px 44px rgba(0,0,0,0.22);
}
.quote-green b{ color: rgba(90, 255, 160, 0.95); }

.quote-blue{
  background: rgba(90, 150, 255, 0.16);
  border: 1px solid rgba(90, 150, 255, 0.26);
  color: rgba(210, 230, 255, 0.95);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 12px 44px rgba(0,0,0,0.22);
}
.quote-blue b{ color: rgba(130, 190, 255, 0.95); }

/* ===== DataFrame/Table darken (header white -> dark) ===== */
div[data-testid="stDataFrame"]{
  background: rgba(0,0,0,0.00) !important;
}
div[data-testid="stDataFrame"] thead tr th{
  background: rgba(10,12,26,1) !important;
  color: rgba(245,245,255,0.90) !important;
  border-bottom: 1px solid rgba(255,255,255,0.12) !important;
}
div[data-testid="stDataFrame"] tbody tr td{
  background: rgba(0,0,0,0.20) !important;
  color: rgba(245,245,255,0.88) !important;
  border-bottom: 1px solid rgba(255,255,255,0.07) !important;
}

/* ===== Plotly container look ===== */
div[data-testid="stPlotlyChart"] > div{
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 18px 60px rgba(0,0,0,0.30);
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def softmax_stable(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(1e-9, float(temperature))
    z = (x - np.max(x)) / t
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)

def make_seed(s: str) -> int:
    return int(zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF)

def df_dark_style(df: pd.DataFrame):
    """st.dataframeに渡せるStyler（ヘッダ/セルを黒系に統一）"""
    try:
        return (
            df.style
            .set_table_styles([
                {"selector":"th", "props":[("background-color","rgba(10,12,26,1)"),
                                          ("color","rgba(245,245,255,0.90)"),
                                          ("border-bottom","1px solid rgba(255,255,255,0.12)")]},
                {"selector":"td", "props":[("background-color","rgba(0,0,0,0.20)"),
                                          ("color","rgba(245,245,255,0.88)"),
                                          ("border-bottom","1px solid rgba(255,255,255,0.07)")]},
                {"selector":"table", "props":[("border-collapse","collapse")]}
            ])
        )
    except Exception:
        return df

# ------------------------------------------------------------
# Load Excel pack
# -------------------------------------------------------------
def load_pack(xlsx_bytes: bytes) -> Dict[str, pd.DataFrame]:
    # sheet_name=None で一括読み込み（ポインタ問題を回避）
    book = pd.read_excel(pd.io.common.BytesIO(xlsx_bytes), sheet_name=None, engine="openpyxl")

    # キーを正規化（大文字＋前後空白除去）
    sheets = {str(k).strip().upper(): v for k, v in book.items()}

    # 取り出し（別名も許容）
    def pick(*cands):
        for c in cands:
            if c in sheets:
                df = sheets[c].copy()
                # 列名を正規化（重要：空白/全角スペース/改行など）
                df.columns = [str(col).replace("\u3000", " ").strip() for col in df.columns]
                return df
        return pd.DataFrame()

    return {
        "VOW": pick("VOW"),
        "CHAR": pick("CHAR", "CHARS", "CHARA"),
        "STAGE": pick("STAGE", "SEASON", "TIME"),
        "QUOTES": pick("QUOTES", "QUOTE"),
    }

# ------------------------------------------------------------
# Text -> keywords (simple, fast)
# ------------------------------------------------------------
STOP = set(["した","たい","いる","こと","それ","これ","ため","よう","ので","から","です","ます","ある","ない","そして","でも","しかし","また",
            "自分","私","あなた","もの","感じ","気持ち","今日","今","に","を","が","は","と","も","で","へ","や","の"])

def extract_keywords_simple(text: str, top_n: int = 5) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    s = re.sub(r"[0-9０-９、。．,.!！?？\(\)\[\]{}「」『』\"'：:;／/\\\n\r\t]+", " ", text)
    tokens = [t.strip() for t in re.split(r"\s+", s) if t.strip()]
    tokens = [t for t in tokens if len(t) >= 2 and t not in STOP]
    # 日本語が連続でスペースなしの場合の保険（ざっくり）
    if not tokens and len(text) >= 2:
        tokens = [text[:4], text[-4:]] if len(text) >= 8 else [text]
    # 長い順
    tokens = sorted(list(dict.fromkeys(tokens)), key=lambda x: (-len(x), x))
    return tokens[:top_n]

# ------------------------------------------------------------
# Word-art network (energy closeness)  ※軽量版
# ------------------------------------------------------------
GLOBAL_WORDS_DATABASE = [
    "迷い","決断","挑戦","静けさ","焦り","待つ","行動","つながり","勇気","学び","成長","継続","回復","希望","未来","覚悟","集中",
    "安心","不安","整理","選択","軸","整える","視点","工夫","忍耐","調和","流れ","変化","前進","一歩",
]

def calculate_similarity(w1: str, w2: str) -> float:
    if w1 == w2:
        return 1.0
    c1, c2 = set(w1), set(w2)
    inter = len(c1 & c2)
    denom = max(len(c1), len(c2), 1)
    return float(np.clip(inter / denom, 0.0, 1.0))

def energy_between(w1: str, w2: str, rng: np.random.Generator, jitter: float) -> float:
    sim = calculate_similarity(w1, w2)
    e = -2.0 * sim + 0.5
    if jitter > 0:
        e += rng.normal(0, jitter)
    return float(e)

def build_word_network(center_words: List[str], n_total: int, jitter: float, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    base = list(dict.fromkeys(center_words + GLOBAL_WORDS_DATABASE))
    energies = {}
    for w in base:
        if w in center_words:
            energies[w] = -3.0
        else:
            e_list = [energy_between(c, w, rng, jitter) for c in center_words] if center_words else [0.0]
            energies[w] = float(np.mean(e_list))
    # select low energy first
    selected = [w for w in center_words if w in energies]
    for w, _ in sorted(energies.items(), key=lambda x: x[1]):
        if w not in selected:
            selected.append(w)
        if len(selected) >= n_total:
            break

    # edges
    edges = []
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            e = energy_between(selected[i], selected[j], rng, 0.0)
            if e < -0.25:
                edges.append((i, j, e))

    return {"words": selected, "energies": {w: energies[w] for w in selected}, "edges": edges}

def layout_sphere(words: List[str], energies: Dict[str,float], center_set: set, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(words)
    pos = np.zeros((n,3), dtype=float)

    # spread points on sphere-ish (golden spiral) with energy-based radius
    golden = np.pi * (3 - np.sqrt(5))
    ev = list(energies.values()) if energies else [0.0]
    mn, mx = min(ev), max(ev)
    er = (mx - mn) if mx != mn else 1.0

    k = 0
    for i, w in enumerate(words):
        if w in center_set:
            pos[i] = [0.0, 0.0, 0.0]
            continue
        e = energies.get(w, 0.0)
        norm = (e - mn) / er
        radius = 0.8 + (1.0 - norm) * 1.8

        theta = golden * k
        y = 1 - (k / float(max(1, n-2))) * 2
        r = math.sqrt(max(0.0, 1 - y*y))
        x = math.cos(theta) * r * radius
        z = math.sin(theta) * r * radius
        pos[i] = [x, y*radius*0.7, z]
        k += 1

    # small jitter for aesthetics
    pos += rng.normal(0, 0.02, size=pos.shape)
    return pos

def plot_word_sphere(network: Dict, pos: np.ndarray, center_set: set) -> go.Figure:
    words = network["words"]
    energies = network["energies"]
    edges = network["edges"]

    fig = go.Figure()

    # edges
    xE,yE,zE = [],[],[]
    for i,j,e in edges:
        x0,y0,z0 = pos[i]
        x1,y1,z1 = pos[j]
        xE += [x0,x1,None]
        yE += [y0,y1,None]
        zE += [z0,z1,None]
    fig.add_trace(go.Scatter3d(
        x=xE,y=yE,z=zE, mode="lines",
        line=dict(width=1, color="rgba(200,220,255,0.22)"),
        hoverinfo="skip",
        showlegend=False
    ))

    # nodes
    sizes, colors, labels = [], [], []
    for w in words:
        e = energies.get(w, 0.0)
        labels.append(w)
        if w in center_set:
            sizes.append(26)
            colors.append("rgba(255,235,100,0.98)")
        else:
            en = min(1.0, abs(e)/3.0)
            sizes.append(10 + int(10*en))
            colors.append("rgba(220,240,255,0.70)" if e < -0.5 else "rgba(255,255,255,0.55)")

    idx_center = [i for i,w in enumerate(words) if w in center_set]
    idx_other  = [i for i,w in enumerate(words) if w not in center_set]

    if idx_other:
        oi = np.array(idx_other, dtype=int)
        fig.add_trace(go.Scatter3d(
            x=pos[oi,0], y=pos[oi,1], z=pos[oi,2],
            mode="markers+text",
            text=[labels[i] for i in oi],
            textposition="top center",
            textfont=dict(size=16, color="rgba(245,245,255,0.92)"),
            marker=dict(size=[sizes[i] for i in oi], color=[colors[i] for i in oi],
                        line=dict(width=1, color="rgba(0,0,0,0.18)")),
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False
        ))
    if idx_center:
        ci = np.array(idx_center, dtype=int)
        fig.add_trace(go.Scatter3d(
            x=pos[ci,0], y=pos[ci,1], z=pos[ci,2],
            mode="markers+text",
            text=[labels[i] for i in ci],
            textposition="top center",
            textfont=dict(size=20, color="rgba(255,80,80,1.0)"),
            marker=dict(size=[sizes[i] for i in ci], color=[colors[i] for i in ci],
                        line=dict(width=2, color="rgba(255,80,80,0.75)")),
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False
        ))

    fig.update_layout(
        paper_bgcolor="rgba(6,8,18,1)",
        plot_bgcolor="rgba(6,8,18,1)",
        font=dict(color="rgba(245,245,255,0.92)"),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(6,8,18,1)",
            camera=dict(eye=dict(x=1.55, y=1.10, z=1.05)),
            dragmode="orbit"
        ),
        margin=dict(l=0,r=0,t=0,b=0),
        height=520
    )
    return fig

# ------------------------------------------------------------
# Sidebar: upload & parameters
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📁 データ")
    up = st.file_uploader("統合Excel（pack）", type=["xlsx"], accept_multiple_files=False)

    st.markdown("---")
    st.markdown("## ⚙️ QUBO設定（one-hot）")
    P = st.slider("one-hot ペナルティ P", 1.0, 80.0, 40.0, 1.0)
    sample_n = st.slider("サンプル数（観測分布）", 50, 800, 300, 10)
    sa_sweeps = st.slider("SA sweeps（揺らぎ）", 50, 1200, 420, 10)
    temperature = st.slider("SA温度（大→揺らぐ）", 0.20, 3.00, 1.20, 0.05)

    st.markdown("---")
    st.markdown("## 🧠 テキスト→誓願（自動ベクトル化）")
    ngram = st.selectbox("n-gram", [2, 3, 4], index=1)
    mix_alpha = st.slider("mix比率 α（1=スライダーのみ / 0=テキストのみ）", 0.0, 1.0, 0.55, 0.01)

    st.markdown("---")
    st.markdown("## 💬 QUOTES神託（温度付きで選択）")
    quote_lang = st.selectbox("LANG", ["ja", "en"], index=0)
    quote_temp = st.slider("格言温度（高→ランダム / 低→上位固定）", 0.20, 3.00, 1.20, 0.05)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
st.markdown(f"# {APP_TITLE}")

if up is None:
    st.info("左のサイドバーから **統合Excel（pack）** をアップロードしてください。")
    st.stop()

pack_bytes = up.getvalue()
sheets = load_pack(pack_bytes)

df_vow = sheets["VOW"].copy()
df_char = sheets["CHAR"].copy()
df_stage = sheets["STAGE"].copy()
df_quotes = sheets["QUOTES"].copy()

# ------------------------------------------------------------
# Normalize columns (robust)
# ------------------------------------------------------------
def col_pick(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    cols = list(df.columns)
    low = {str(c).lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

# VOW: id/title/desc
vow_id_col = col_pick(df_vow, ["VOW", "VOW_ID", "ID"])
vow_title_col = col_pick(df_vow, ["TITLE", "題", "誓願", "NAME"])
vow_desc_col = col_pick(df_vow, ["DESC", "DESCRIPTION", "説明", "TEXT"])

# CHAR: id/name/img/weights
char_id_col = col_pick(df_char, ["CHAR_ID", "ID"])
char_name_col = col_pick(df_char, ["CHAR", "NAME", "神", "TITLE"])
char_img_col = col_pick(df_char, ["IMG", "IMAGE", "PATH", "FILE"])

# W: expects columns like VOW_01.. or weights matrix
# We'll detect weight columns as those starting with "VOW_" (case-insensitive)
weight_cols = [c for c in df_char.columns if str(c).upper().startswith("VOW_")]

# STAGE: stage_id/name etc
stage_id_col = col_pick(df_stage, ["STAGE_ID", "ID"])
stage_name_col = col_pick(df_stage, ["STAGE", "NAME", "TITLE"])

# QUOTES: quote/source/keywords/lang
quote_text_col = col_pick(df_quotes, ["QUOTE", "格言", "TEXT", "言葉"])
quote_src_col  = col_pick(df_quotes, ["SOURCE", "出典", "作者", "出所"])
quote_kw_col   = col_pick(df_quotes, ["KEYWORDS", "キーワード", "TAG", "タグ"])
quote_lang_col = col_pick(df_quotes, ["LANG", "言語"])

if df_vow.empty or df_char.empty or not weight_cols:
    st.error("VOW/CHAR/重み列（VOW_01...）が見つかりません。Excelのシート名・列名をご確認ください。")
    st.stop()

# ------------------------------------------------------------
# Step1: Inputs
# ------------------------------------------------------------
left, right = st.columns([1.6, 1.0], gap="large")

with left:
    st.markdown("## Step 1 : 誓願入力（スライダー）＋テキスト（自動ベクトル化）")

    user_text = st.text_area(
        "あなたの状況を一文で（例：疲れていて決断ができない / 新しい挑戦が怖い など）",
        value="例：迷いを断ちたいが、今は焦らず機を待つべきか悩んでいる…",
        height=90,
        key="user_text"
    )

    # sliders
    vow_rows = []
    for i, row in df_vow.iterrows():
        vid = str(row.get(vow_id_col, f"VOW_{i+1:02d}"))
        title = str(row.get(vow_title_col, vid))
        desc = str(row.get(vow_desc_col, "")).strip()
        label = f"{vid}｜{title}"
        if desc:
            label += f" — {desc}"

        val = st.slider(label, 0.0, 5.0, 0.0, 0.5, key=f"sl_{vid}")
        vow_rows.append((vid, title, val))

# ------------------------------------------------------------
# Build vow vector (manual + text)
# ------------------------------------------------------------
vow_ids = [v[0] for v in vow_rows]
manual_v = np.array([v[2] for v in vow_rows], dtype=float)

def text_to_vector(text: str, vow_titles: List[str], ngram_n: int = 3) -> np.ndarray:
    """超軽量：char n-gramで vow title との近さを出す（0..1程度）"""
    text = (text or "").strip()
    if not text:
        return np.zeros(len(vow_titles), dtype=float)

    def ngrams(s: str, n: int) -> set:
        s = re.sub(r"\s+", "", s)
        if len(s) < n:
            return {s} if s else set()
        return {s[i:i+n] for i in range(len(s)-n+1)}

    tg = ngrams(text.lower(), ngram_n)
    out = []
    for t in vow_titles:
        gg = ngrams(str(t).lower(), ngram_n)
        inter = len(tg & gg)
        denom = max(len(gg), 1)
        out.append(inter / denom)
    v = np.array(out, dtype=float)
    # scale to 0..5-ish
    if v.max() > 0:
        v = 5.0 * (v / v.max())
    return v

vow_titles = [v[1] for v in vow_rows]
auto_v = text_to_vector(user_text, vow_titles, ngram_n=int(ngram))

mix_v = mix_alpha * manual_v + (1.0 - mix_alpha) * auto_v

# ------------------------------------------------------------
# QUBO(one-hot) choose god
# ------------------------------------------------------------
# Energy per god i: Ei = sum_j W(i,j) * mix_v(j)
# one-hot penalty shown as: E(x)=sum_i Ei*x_i + P*(sum_i x_i - 1)^2
W = df_char[weight_cols].applymap(lambda x: safe_float(x, 0.0)).to_numpy(dtype=float)
E = (W @ mix_v.reshape(-1, 1)).reshape(-1)  # (n_char,)

# one-hot exact solution (pick argmin E) but keep QUBO proof with penalty
best_i = int(np.argmin(E))
x = np.zeros(len(E), dtype=int)
x[best_i] = 1

qubo_energy = float(np.sum(E * x) + P * (np.sum(x) - 1) ** 2)

# observation distribution (temperature) using energies (lower better)
# Use Boltzmann on -E/temperature
prob = softmax_stable(-E, temperature=max(1e-6, float(temperature)))
rng = np.random.default_rng(make_seed(user_text + str(time.time_ns())))
samples = rng.choice(np.arange(len(E)), size=int(sample_n), p=prob)
counts = np.bincount(samples, minlength=len(E))

# ------------------------------------------------------------
# Result tables
# ------------------------------------------------------------
char_ids = df_char[char_id_col].astype(str).tolist() if char_id_col else [f"CHAR_{i+1:02d}" for i in range(len(df_char))]
char_names = df_char[char_name_col].astype(str).tolist() if char_name_col else [f"神_{i+1}" for i in range(len(df_char))]

df_rank = pd.DataFrame({
    "順位": np.arange(1, len(E)+1),
    "CHAR_ID": char_ids,
    "神": char_names,
    "energy（低いほど選ばれやすい）": E,
    "確率（softmax）": prob
}).sort_values("energy（低いほど選ばれやすい）", ascending=True).reset_index(drop=True)

top3 = df_rank.head(3).copy()

# selected character info
sel_char_id = str(df_rank.iloc[0]["CHAR_ID"])
sel_name = str(df_rank.iloc[0]["神"])

# image path (repo path)
img_path = None
if char_img_col and sel_char_id in df_char[char_id_col].astype(str).values:
    row = df_char.loc[df_char[char_id_col].astype(str) == sel_char_id].iloc[0]
    img_path = str(row.get(char_img_col, "")).strip()

# ------------------------------------------------------------
# QUOTES selection (temperature)
# ------------------------------------------------------------
def quote_score(quote: str, keywords: List[str]) -> float:
    q = (quote or "").lower()
    if not q:
        return 0.0
    score = 0.0
    for k in keywords:
        kk = (k or "").lower()
        if kk and kk in q:
            score += 2.0
        # partial
        if kk and (len(kk) >= 2):
            score += 0.2 * sum(1 for i in range(len(kk)-1) if kk[i:i+2] in q)
    return score

# use extracted keywords from user text + top contributing vows
kw_text = extract_keywords_simple(user_text, top_n=5)
# top contributing vows from mix_v
df_contrib = pd.DataFrame({"VOW": vow_ids, "TITLE": vow_titles, "mix(v)": mix_v})
df_contrib["寄与"] = df_contrib["mix(v)"]
top_vows = df_contrib.sort_values("寄与", ascending=False).head(3)
kw = kw_text + top_vows["TITLE"].astype(str).tolist()
kw = list(dict.fromkeys([k for k in kw if k]))

if df_quotes.empty or quote_text_col is None:
    picked_quote = {"quote": "未来は、今の選択によって少しずつ形になる。", "source": "—"}
    df_qcand = pd.DataFrame()
else:
    qdf = df_quotes.copy()
    if quote_lang_col and quote_lang:
        qdf = qdf[qdf[quote_lang_col].astype(str).str.lower().fillna("") == str(quote_lang).lower()].copy() if quote_lang_col in qdf.columns else qdf

    qdf["__quote"] = qdf[quote_text_col].astype(str).fillna("")
    qdf["__source"] = qdf[quote_src_col].astype(str).fillna("—") if quote_src_col else "—"
    qdf["__score"] = qdf["__quote"].apply(lambda s: quote_score(s, kw))

    # Convert to energy: lower is better
    # energy = -score + small length prior
    qdf["__energy"] = -qdf["__score"] + 0.002 * qdf["__quote"].str.len().clip(lower=0)

    qdf = qdf.sort_values("__energy", ascending=True).reset_index(drop=True)

    # temperature pick
    qE = qdf["__energy"].to_numpy(dtype=float)
    qprob = softmax_stable(-qE, temperature=max(1e-6, float(quote_temp)))
    pick_idx = int(rng.choice(np.arange(len(qdf)), p=qprob)) if len(qdf) > 0 else 0

    picked_quote = {
        "quote": str(qdf.iloc[pick_idx]["__quote"]) if len(qdf) else "—",
        "source": str(qdf.iloc[pick_idx]["__source"]) if len(qdf) else "—",
    }

    df_qcand = qdf.head(12)[["__quote", "__source", "__score", "__energy"]].rename(
        columns={"__quote":"QUOTE", "__source":"SOURCE", "__score":"score", "__energy":"energy"}
    )

# ------------------------------------------------------------
# Plotly theme helper (dark)
# ------------------------------------------------------------
def plotly_dark_layout(fig: go.Figure, height: int = 360):
    fig.update_layout(
        paper_bgcolor="rgba(6,8,18,1)",
        plot_bgcolor="rgba(6,8,18,1)",
        font=dict(color="rgba(245,245,255,0.90)"),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(showgrid=False, color="rgba(245,245,255,0.70)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.10)", color="rgba(245,245,255,0.70)")
    return fig

# ------------------------------------------------------------
# Right side results
# ------------------------------------------------------------
with right:
    st.markdown("## Step 3 : 結果（観測された神＋理由＋QUOTES神託）")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ランキング（上位）")
    st.dataframe(df_dark_style(top3), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Observed god
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"### 🌟 今回“観測”された神： **{sel_name}**  ({sel_char_id})")
    st.caption("※ここは“単発の観測（1回抽選）”です。下の観測分布（サンプル）は「同条件で何回も観測したらどう出るか」のヒストグラムです。")

    # show character image if exists
    if img_path:
        try:
            p = Path(img_path)
            if not p.exists():
                # typical repo path: ./assets/images/characters/<file>
                alt = Path("assets") / "images" / "characters" / Path(img_path).name
                p = alt if alt.exists() else p
            if p.exists():
                st.image(Image.open(p), use_container_width=True)
        except Exception:
            pass
    st.markdown("</div>", unsafe_allow_html=True)

    # Green oracle message (継承)
    # Build reason text from top contributing vows
    top_reason = top_vows.head(3)
    reason_words = "・".join(top_reason["TITLE"].astype(str).tolist()) if len(top_reason) else "—"
    green_msg = f"""
<div class="quote-green">
<b>神託（雰囲気：薄い緑 × 濃い緑文字）</b><br>
いまの波は <b>{reason_words}</b> に寄っている。季節×時間（Stage）は流れを強める。<br>
格言：<b>「{picked_quote["quote"]}」</b> — {picked_quote["source"]}
</div>
"""
    st.markdown(green_msg, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Contrib table (Top)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🧩 寄与した誓願（Top）")
    show_contrib = top_vows[["VOW","TITLE","mix(v)"]].copy()
    show_contrib["W(char,v)"] = 0.0
    show_contrib["寄与(v*w)"] = 0.0
    # compute W for selected god
    sel_row = df_char.iloc[best_i]
    for idx in show_contrib.index:
        vid = show_contrib.loc[idx,"VOW"]
        col = None
        for c in weight_cols:
            if str(c).upper() == str(vid).upper():
                col = c
                break
        wcv = safe_float(sel_row.get(col, 0.0), 0.0) if col else 0.0
        show_contrib.loc[idx,"W(char,v)"] = wcv
        show_contrib.loc[idx,"寄与(v*w)"] = float(show_contrib.loc[idx,"mix(v)"] * wcv)

    st.dataframe(df_dark_style(show_contrib), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Blue quotes box (継承)
    st.markdown("### 🗣 QUOTES神託（温度付きで選択）")
    blue = f"""
<div class="quote-blue">
<b>QUOTES神託（雰囲気：濃い青 × 青文字）</b><br>
「{picked_quote["quote"]}」<br>
— {picked_quote["source"]}
</div>
"""
    st.markdown(blue, unsafe_allow_html=True)

    with st.expander("🔎 格言候補Top（デバッグ）", expanded=False):
        if isinstance(df_qcand, pd.DataFrame) and not df_qcand.empty:
            st.dataframe(df_dark_style(df_qcand), use_container_width=True, hide_index=True)
        else:
            st.caption("格言候補がありません（QUOTESシート/列名をご確認ください）。")

# ------------------------------------------------------------
# Visualization section (graphs + keyword + word-sphere art)
# ------------------------------------------------------------
st.markdown("## 📊 可視化：テキストの影響・観測分布・エネルギー地形")

# 1) auto vs manual vs mix
df_cmp = pd.DataFrame({
    "VOW": vow_ids,
    "manual": manual_v,
    "auto": auto_v,
    "mix": mix_v
})
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_cmp["VOW"], y=df_cmp["auto"], mode="lines+markers", name="auto"))
fig1.add_trace(go.Scatter(x=df_cmp["VOW"], y=df_cmp["manual"], mode="lines+markers", name="manual"))
fig1.add_trace(go.Scatter(x=df_cmp["VOW"], y=df_cmp["mix"], mode="lines+markers", name="mix"))
fig1.update_layout(title="1) テキスト→誓願 自動推定の影響（auto vs manual vs mix）")
plotly_dark_layout(fig1, height=380)
st.plotly_chart(fig1, use_container_width=True, config={"displaylogo": False})

# 2) energy landscape (all candidates)
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=char_names, y=E))
fig2.update_layout(title="2) エネルギー地形（全候補）", xaxis_title="神", yaxis_title="energy（低いほど選ばれやすい）")
plotly_dark_layout(fig2, height=380)
st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})

# 3) observation distribution
fig3 = go.Figure()
fig3.add_trace(go.Bar(x=char_names, y=counts))
fig3.update_layout(title="3) 観測分布（サンプル）", xaxis_title="神", yaxis_title="count")
plotly_dark_layout(fig3, height=380)
st.plotly_chart(fig3, use_container_width=True, config={"displaylogo": False})

# 4) keywords + word sphere art
st.markdown("### 4) テキストのキーワード抽出（簡易）")
if not user_text.strip():
    st.caption("（入力テキストが空です）")
else:
    st.caption(f"抽出キーワード：{', '.join(kw_text) if kw_text else '—'}")

# 追加：球体（アート）
st.markdown("#### 🌐 縁の球体（誓願テキスト → キーワード → エネルギー近さで単語を接続）")
center_words = kw_text[:3] if kw_text else [t for t in top_vows["TITLE"].astype(str).tolist()[:2]]
center_words = [w for w in center_words if w]

seed = make_seed(user_text + "|sphere")
network = build_word_network(center_words=center_words, n_total=30, jitter=0.08, seed=seed)
center_set = set(center_words)
pos = layout_sphere(network["words"], network["energies"], center_set=center_set, seed=seed)
fig_sphere = plot_word_sphere(network, pos, center_set=center_set)

st.plotly_chart(
    fig_sphere,
    use_container_width=True,
    config={"displaylogo": False, "scrollZoom": True, "doubleClick": "reset"}
)

# ------------------------------------------------------------
# QUBO proof
# ------------------------------------------------------------
with st.expander("🧠 QUBO 証拠（one-hot 制約）", expanded=False):
    st.code(
        "E(x)= Σ_i E_i x_i + P(Σ_i x_i - 1)^2\n"
        f"P={P:.2f}\n"
        f"x={x.tolist()}\n"
        f"qubo_energy={qubo_energy:.6f}\n",
        language="text"
    )

