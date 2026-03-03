# -*- coding: utf-8 -*-
# ============================================================
# Q-Quest 量子神託 (QUBO / STAGE×QUOTES)
# - Excel統合packを読み込み
# - Step1: 誓願入力（スライダー） + テキスト（自動ベクトル化）
# - Step2: QUBO(one-hot)で「観測」(SAサンプル)
# - Step3: 結果（観測された神 + 理由 + QUOTES神託） + キャラ画像
# - Step4: テキストのキーワード抽出（簡易） + 単語の球体(アート)
# ============================================================

import os
import re
import io
import zlib
import time
import math
import glob
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ----------------------------
# Streamlit config (MUST FIRST)
# ----------------------------
st.set_page_config(page_title="Q-Quest 量子神託 (QUBO / STAGE×QUOTES)", layout="wide")

# ============================================================
# 0) THEME / CSS（白いUIを全て黒基調へ）
# ============================================================
SPACE_CSS = """
<style>
/* --- App background --- */
.stApp{
  background:
    radial-gradient(circle at 18% 24%, rgba(110,150,255,0.12), transparent 38%),
    radial-gradient(circle at 78% 68%, rgba(255,160,220,0.08), transparent 44%),
    radial-gradient(circle at 50% 50%, rgba(255,255,255,0.03), transparent 55%),
    linear-gradient(180deg, rgba(6,8,18,1), rgba(10,12,26,1));
}

/* --- header / toolbar (Share, GitHub icon area) --- */
header[data-testid="stHeader"]{
  background: rgba(6,8,18,0.90) !important;
  border-bottom: 1px solid rgba(255,255,255,0.08) !important;
}
div[data-testid="stToolbar"]{
  background: rgba(6,8,18,0.90) !important;
}
div[data-testid="stToolbar"] *{
  color: rgba(245,245,255,0.85) !important;
  fill: rgba(245,245,255,0.85) !important;
}
a, a:visited { color: rgba(170,210,255,0.95) !important; }

/* --- Base typography --- */
.block-container{ padding-top: 1.2rem; }
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li{
  font-family: "Hiragino Mincho ProN","Yu Mincho","Noto Serif JP",serif;
  letter-spacing: 0.02em;
  color: rgba(245,245,255,0.92);
}
h1,h2,h3{
  font-family: "Hiragino Mincho ProN","Yu Mincho","Noto Serif JP",serif !important;
  font-weight: 650 !important;
  color: rgba(245,245,255,0.95) !important;
  text-shadow: 0 2px 18px rgba(0,0,0,0.45);
}

/* --- Sidebar --- */
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.08);
  border-right: 1px solid rgba(255,255,255,0.10);
  backdrop-filter: blur(10px);
}
section[data-testid="stSidebar"] *{
  color: rgba(245,245,255,0.92) !important;
}

/* --- inputs: make text visible --- */
textarea, input{
  color: rgba(245,245,255,0.95) !important;
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
}
textarea::placeholder, input::placeholder{
  color: rgba(245,245,255,0.55) !important;
}

/* --- file uploader (white panel fix) --- */
div[data-testid="stFileUploader"]{
  background: rgba(0,0,0,0.30) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
  padding: 10px !important;
}
div[data-testid="stFileUploader"] *{
  color: rgba(245,245,255,0.90) !important;
}

/* --- Cards --- */
.card{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 16px 12px 16px;
  box-shadow: 0 18px 60px rgba(0,0,0,0.22);
}
.smallnote{opacity:0.80; font-size:0.92rem;}

/* --- Dataframe/Table : force dark --- */
div[data-testid="stDataFrame"]{
  border-radius: 16px !important;
  overflow: hidden !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  background: rgba(6,8,18,0.85) !important;
}
div[data-testid="stDataFrame"] *{
  color: rgba(245,245,255,0.92) !important;
}
div[data-testid="stDataFrame"] [role="columnheader"]{
  background: rgba(10,12,26,0.98) !important;
  color: rgba(245,245,255,0.95) !important;
  border-bottom: 1px solid rgba(255,255,255,0.10) !important;
}
div[data-testid="stDataFrame"] [role="gridcell"]{
  background: rgba(6,8,18,0.85) !important;
  border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}
div[data-testid="stDataFrame"] [data-testid="stTable"]{
  background: rgba(6,8,18,0.85) !important;
}

/* --- st.table fallback --- */
table{
  background: rgba(6,8,18,0.85) !important;
  color: rgba(245,245,255,0.92) !important;
}
thead tr th{
  background: rgba(10,12,26,0.98) !important;
  color: rgba(245,245,255,0.95) !important;
}
tbody tr td{
  background: rgba(6,8,18,0.85) !important;
  color: rgba(245,245,255,0.92) !important;
  border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}
</style>
"""
st.markdown(SPACE_CSS, unsafe_allow_html=True)

# ============================================================
# 1) Utils
# ============================================================
def norm_col(s: str) -> str:
    s = str(s or "")
    s = s.replace("　", " ").strip()
    s = s.upper()
    s = re.sub(r"[\s\-]+", "_", s)
    s = s.replace("＿", "_")
    return s

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def make_seed(s: str) -> int:
    return int(zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF)

def softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    x = np.array(x, dtype=float)
    tau = max(1e-6, float(tau))
    z = (x - np.max(x)) / tau
    e = np.exp(z)
    return e / np.sum(e)

# ============================================================
# 2) Excel loader (lenient)
# ============================================================
@st.cache_data(show_spinner=False)
def load_excel_pack(excel_bytes: bytes, file_hash: str) -> Dict[str, pd.DataFrame]:
    bio = io.BytesIO(excel_bytes)
    xls = pd.ExcelFile(bio)
    out = {}
    for name in xls.sheet_names:
        try:
            df = pd.read_excel(bio, sheet_name=name, engine="openpyxl")
        except Exception:
            continue
        out[name] = df
    return out

def find_sheet(sheets: Dict[str, pd.DataFrame], candidates: List[str]) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    cand_norm = [norm_col(c) for c in candidates]
    for k, df in sheets.items():
        if norm_col(k) in cand_norm:
            return k, df
    # fallback: contains match
    for k, df in sheets.items():
        nk = norm_col(k)
        for c in cand_norm:
            if c in nk:
                return k, df
    return None, None

def detect_vow_columns(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    normed = [norm_col(c) for c in cols]

    vow_cols = []
    for c, nc in zip(cols, normed):
        # accept: VOW_01, VOW01, VOW_1, VOW-01, etc.
        if re.fullmatch(r"VOW_?\d{1,2}", nc):
            vow_cols.append(c)
        elif re.fullmatch(r"VOW_?\d{1,2}\.0", nc):
            vow_cols.append(c)

    # if nothing, try any column containing "VOW" and digits
    if not vow_cols:
        for c, nc in zip(cols, normed):
            if "VOW" in nc and re.search(r"\d", nc):
                vow_cols.append(c)

    # sort by vow index if possible
    def vow_key(col):
        m = re.search(r"(\d{1,2})", norm_col(col))
        return int(m.group(1)) if m else 999

    vow_cols = sorted(list(dict.fromkeys(vow_cols)), key=vow_key)
    return vow_cols

def build_master_vows(vow_master: Optional[pd.DataFrame], vow_cols: List[str]) -> pd.DataFrame:
    """
    return df with columns: VOW_ID, TITLE
    """
    if vow_master is not None and len(vow_master) > 0:
        cols = {norm_col(c): c for c in vow_master.columns}
        vid = cols.get("VOW_ID") or cols.get("VOW") or cols.get("ID")
        ttl = cols.get("TITLE") or cols.get("NAME") or cols.get("LABEL")
        if vid and ttl:
            tmp = vow_master[[vid, ttl]].copy()
            tmp.columns = ["VOW_ID", "TITLE"]
            tmp["VOW_ID"] = tmp["VOW_ID"].astype(str).str.strip()
            tmp["TITLE"] = tmp["TITLE"].astype(str).str.strip()
            tmp = tmp[tmp["VOW_ID"].str.len() > 0]
            return tmp.reset_index(drop=True)

    # fallback from vow_cols
    rows = []
    for c in vow_cols:
        nc = norm_col(c)
        m = re.search(r"(\d{1,2})", nc)
        idx = int(m.group(1)) if m else None
        vow_id = f"VOW_{idx:02d}" if idx is not None else nc
        rows.append((vow_id, vow_id))
    return pd.DataFrame(rows, columns=["VOW_ID", "TITLE"])

def build_master_chars(char_master: Optional[pd.DataFrame], char_to_vow: pd.DataFrame) -> pd.DataFrame:
    """
    return df with columns: CHAR_ID, 神
    """
    if char_master is not None and len(char_master) > 0:
        cols = {norm_col(c): c for c in char_master.columns}
        cid = cols.get("CHAR_ID") or cols.get("CHAR") or cols.get("ID")
        god = cols.get("神") or cols.get("NAME") or cols.get("TITLE")
        if cid and god:
            tmp = char_master[[cid, god]].copy()
            tmp.columns = ["CHAR_ID", "神"]
            tmp["CHAR_ID"] = tmp["CHAR_ID"].astype(str).str.strip()
            tmp["神"] = tmp["神"].astype(str).str.strip()
            tmp = tmp[tmp["CHAR_ID"].str.len() > 0]
            return tmp.reset_index(drop=True)

    # fallback from char_to_vow
    cols = {norm_col(c): c for c in char_to_vow.columns}
    cid = cols.get("CHAR_ID") or cols.get("CHAR") or cols.get("ID")
    if cid:
        ids = char_to_vow[cid].astype(str).str.strip().tolist()
    else:
        ids = [f"CHAR_{i:02d}" for i in range(1, 13)]
    rows = [(c, c) for c in ids]
    return pd.DataFrame(rows, columns=["CHAR_ID", "神"])

# ============================================================
# 3) QUOTES
# ============================================================
def load_quotes(quotes_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if quotes_df is None or len(quotes_df) == 0:
        return pd.DataFrame(
            [
                ("Q_0001", "成功は、自分の強みを活かすことから始まる。", "ピーター・ドラッカー", "ja"),
                ("Q_0002", "困難な瞬間こそ、真の性格が現れる。", "アルフレッド・A・モンテパート", "ja"),
                ("Q_0003", "幸福とは、自分自身を探す旅の中で見つけるものだ。", "リリアン・デュ・デュヴェル", "ja"),
            ],
            columns=["QUOTE_ID", "QUOTE", "SOURCE", "LANG"],
        )

    cols = {norm_col(c): c for c in quotes_df.columns}
    qid = cols.get("QUOTE_ID") or cols.get("ID") or cols.get("Q_ID")
    qt = cols.get("QUOTE") or cols.get("格言") or cols.get("言葉") or cols.get("テキスト")
    src = cols.get("SOURCE") or cols.get("出典") or cols.get("作者") or cols.get("出所")
    lang = cols.get("LANG") or cols.get("LANGUAGE")

    use = []
    for key, col in [("QUOTE_ID", qid), ("QUOTE", qt), ("SOURCE", src), ("LANG", lang)]:
        if col:
            use.append(col)

    tmp = quotes_df[use].copy()
    rename = {}
    if qid: rename[qid] = "QUOTE_ID"
    if qt:  rename[qt]  = "QUOTE"
    if src: rename[src] = "SOURCE"
    if lang:rename[lang]= "LANG"
    tmp = tmp.rename(columns=rename)
    if "LANG" not in tmp.columns:
        tmp["LANG"] = "ja"
    tmp["QUOTE"] = tmp["QUOTE"].astype(str).str.strip()
    tmp = tmp[tmp["QUOTE"].str.len() > 0].reset_index(drop=True)
    return tmp

def pick_quotes_by_temperature(dfq: pd.DataFrame, lang: str, k: int, tau: float, seed: int) -> pd.DataFrame:
    d = dfq.copy()
    d["LANG"] = d["LANG"].astype(str).str.strip().str.lower()
    lang = (lang or "ja").strip().lower()
    pool = d[d["LANG"].str.contains(lang, na=False)]
    if len(pool) < k:
        pool = d  # fallback

    rng = np.random.default_rng(seed)
    # pseudo-score: length preference + small noise
    s = pool["QUOTE"].astype(str).str.len().values.astype(float)
    s = (s - s.mean()) / (s.std() + 1e-6)
    s = -np.abs(s) + rng.normal(0, 0.35, size=len(pool))
    p = softmax(s, tau=max(0.2, float(tau)))
    idx = rng.choice(np.arange(len(pool)), size=min(k, len(pool)), replace=False, p=p)
    out = pool.iloc[idx].copy().reset_index(drop=True)
    return out

# ============================================================
# 4) Keyword extraction (simple)
# ============================================================
STOP_TOKENS = set([
    "した","たい","いる","こと","それ","これ","ため","よう","ので","から",
    "です","ます","ある","ない","そして","でも","しかし","また",
    "自分","私","あなた","もの","感じ","気持ち","今日",
    "に","を","が","は","と","も","で","へ","や","の",
])

def extract_keywords(text: str, top_n: int = 6) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # split by punctuation/spaces
    cleaned = re.sub(r"[0-9０-９、。．,.!！?？\(\)\[\]{}「」『』\"'：:;／/\\\n\r\t]+", " ", text)
    toks = [t.strip() for t in re.split(r"\s+", cleaned) if t.strip()]
    toks = [t for t in toks if (len(t) >= 2 and t not in STOP_TOKENS)]
    if not toks:
        return []
    # prioritize longer tokens
    toks = sorted(list(dict.fromkeys(toks)), key=lambda s: (-len(s), s))
    return toks[:top_n]

# ============================================================
# 5) Word-sphere art (lighter version of your code)
# ============================================================
GLOBAL_WORDS_DATABASE = [
    "世界平和","貢献","成長","学び","挑戦","夢","希望","未来",
    "感謝","愛","幸せ","喜び","安心","充実","満足","平和",
    "努力","継続","忍耐","誠実","正直","優しさ","思いやり","共感",
    "調和","バランス","自然","美","真実","自由","正義","道",
    "絆","つながり","家族","友人","仲間","信頼","尊敬","協力",
    "今","瞬間","過程","変化","進化","発展","循環","流れ",
    "静けさ","集中","覚悟","決意","勇気","強さ","柔軟性","寛容",
]

CATEGORIES = {
    "願い": ["世界平和","貢献","成長","夢","希望","未来"],
    "感情": ["感謝","愛","幸せ","喜び","安心","満足","平和"],
    "行動": ["努力","継続","忍耐","誠実","正直","挑戦","学び"],
    "哲学": ["調和","バランス","自然","美","道","真実","自由","正義"],
    "関係": ["絆","つながり","家族","友人","仲間","信頼","尊敬","協力"],
    "内的": ["静けさ","集中","覚悟","決意","勇気","強さ","柔軟性","寛容"],
    "時間": ["今","瞬間","過程","変化","進化","発展","循環","流れ"],
}

def calculate_semantic_similarity(word1: str, word2: str) -> float:
    if word1 == word2:
        return 1.0
    common_chars = set(word1) & set(word2)
    char_sim = len(common_chars) / max(len(set(word1)), len(set(word2)), 1)

    category_sim = 0.0
    for _, ws in CATEGORIES.items():
        w1_in = word1 in ws
        w2_in = word2 in ws
        if w1_in and w2_in:
            category_sim = 1.0
            break
        elif w1_in or w2_in:
            category_sim = max(category_sim, 0.3)

    len_sim = 1.0 - abs(len(word1) - len(word2)) / max(len(word1), len(word2), 1)
    similarity = 0.4 * char_sim + 0.4 * category_sim + 0.2 * len_sim
    return float(np.clip(similarity, 0.0, 1.0))

def energy_between(word1: str, word2: str, rng: np.random.Generator, jitter: float) -> float:
    sim = calculate_semantic_similarity(word1, word2)
    e = -2.0 * sim + 0.5
    if jitter > 0:
        e += rng.normal(0, jitter)
    return float(e)

def build_word_network(center_words: List[str], n_total: int, rng: np.random.Generator, jitter: float) -> Dict:
    base = list(dict.fromkeys(center_words + GLOBAL_WORDS_DATABASE))
    energies = {}
    for w in base:
        if w in center_words:
            energies[w] = -3.0
        else:
            e_list = [energy_between(c, w, rng, jitter) for c in center_words] if center_words else [0.0]
            energies[w] = float(np.mean(e_list))
    # pick low energy words
    picked = [w for w, _ in sorted(energies.items(), key=lambda x: x[1])]
    selected = []
    for w in center_words:
        if w in picked and w not in selected:
            selected.append(w)
    for w in picked:
        if w not in selected:
            selected.append(w)
        if len(selected) >= n_total:
            break

    # edges: connect strongly related pairs
    edges = []
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            e = energy_between(selected[i], selected[j], rng, jitter=0.0)
            if e < -0.65:
                edges.append((i, j, float(e)))
    return {"words": selected, "energies": {w: energies[w] for w in selected}, "edges": edges}

def layout_sphere(words: List[str], energies: Dict[str,float], center_words: List[str], rng: np.random.Generator) -> np.ndarray:
    n = len(words)
    pos = np.zeros((n,3), dtype=float)
    # golden spiral on sphere-ish
    ga = np.pi * (3 - np.sqrt(5))
    for k in range(n):
        y = 1 - (2*k)/(max(1, n-1))
        r = np.sqrt(max(0.0, 1 - y*y))
        th = ga*k
        x = np.cos(th)*r
        z = np.sin(th)*r

        w = words[k]
        e = energies.get(w, 0.0)
        # lower energy -> closer to center
        rad = 0.55 + min(2.4, max(0.1, (e+3.0)))  # e around [-3..]
        rad = np.clip(rad, 0.45, 2.6)
        pos[k] = np.array([x,y,z]) * rad

    # pull center words closer to origin
    for i,w in enumerate(words):
        if w in set(center_words):
            pos[i] *= 0.35
    # tiny noise for aesthetics but stable by seed
    pos += rng.normal(0, 0.015, size=pos.shape)
    return pos

def plot_word_sphere(center_words: List[str], user_keywords: List[str], seed: int, star_count: int = 700) -> go.Figure:
    rng = np.random.default_rng(seed)
    center = [w for w in user_keywords if w] or center_words[:1]
    network = build_word_network(center, n_total=34, rng=rng, jitter=0.06)
    words = network["words"]
    energies = network["energies"]
    edges = network["edges"]
    pos = layout_sphere(words, energies, center, rng)

    fig = go.Figure()

    # stars
    sr = np.random.default_rng(12345)
    sx = sr.uniform(-3.2, 3.2, star_count)
    sy = sr.uniform(-2.4, 2.4, star_count)
    sz = sr.uniform(-2.0, 2.0, star_count)
    alpha = np.full(star_count, 0.20, dtype=float)
    star_size = sr.uniform(1.0, 2.2, star_count)
    star_colors = [f"rgba(255,255,255,{a})" for a in alpha]
    fig.add_trace(go.Scatter3d(
        x=sx,y=sy,z=sz, mode="markers",
        marker=dict(size=star_size, color=star_colors),
        hoverinfo="skip", showlegend=False
    ))

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
        line=dict(width=1, color="rgba(200,220,255,0.20)"),
        hoverinfo="skip", showlegend=False
    ))

    # nodes
    center_set = set(center)
    sizes, colors, labels = [], [], []
    for w in words:
        e = energies.get(w, 0.0)
        if w in center_set:
            sizes.append(26)
            colors.append("rgba(255,235,100,0.98)")
            labels.append(w)
        else:
            sizes.append(10 + int(7*min(1.0, abs(e)/3.0)))
            colors.append("rgba(220,240,255,0.70)" if e < -0.8 else "rgba(255,255,255,0.55)")
            labels.append(w)

    idx_center = np.array([i for i,w in enumerate(words) if w in center_set], dtype=int)
    idx_other  = np.array([i for i,w in enumerate(words) if w not in center_set], dtype=int)

    if len(idx_other) > 0:
        fig.add_trace(go.Scatter3d(
            x=pos[idx_other,0], y=pos[idx_other,1], z=pos[idx_other,2],
            mode="markers+text",
            text=[labels[i] for i in idx_other],
            textposition="top center",
            textfont=dict(size=14, color="rgba(245,245,255,0.92)"),
            marker=dict(size=[sizes[i] for i in idx_other], color=[colors[i] for i in idx_other],
                        line=dict(width=1, color="rgba(0,0,0,0.10)")),
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False
        ))

    if len(idx_center) > 0:
        fig.add_trace(go.Scatter3d(
            x=pos[idx_center,0], y=pos[idx_center,1], z=pos[idx_center,2],
            mode="markers+text",
            text=[labels[i] for i in idx_center],
            textposition="top center",
            textfont=dict(size=20, color="rgba(255,80,80,1.0)"),
            marker=dict(size=[sizes[i] for i in idx_center], color=[colors[i] for i in idx_center],
                        line=dict(width=2, color="rgba(255,80,80,0.85)")),
            hovertemplate="<b>%{text}</b><br>中心語<extra></extra>",
            showlegend=False
        ))

    fig.update_layout(
        paper_bgcolor="rgba(6,8,18,1)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(6,8,18,1)",
            camera=dict(eye=dict(x=1.55, y=1.10, z=1.05)),
            dragmode="orbit",
        ),
        margin=dict(l=0,r=0,t=0,b=0),
        height=520
    )
    return fig

# ============================================================
# 6) Character image resolver (robust)
# ============================================================
@st.cache_data(show_spinner=False)
def scan_character_images(folder: str) -> Dict[str, str]:
    """
    Returns dict: key like 'CHAR_01' -> filepath
    Supports names:
      - CHAR_01.png
      - CHAR_01_xxx.png
      - CHAR_p1.png / CHAR_p01.png
      - any file containing 01 and CHAR
    """
    folder = folder.strip()
    if not folder:
        return {}
    p = Path(folder)
    if not p.exists():
        return {}

    files = []
    for ext in ("*.png","*.jpg","*.jpeg","*.webp"):
        files += list(p.glob(ext))

    mapping: Dict[str,str] = {}
    for f in files:
        name = f.name
        up = name.upper()

        # direct CHAR_01
        m = re.search(r"(CHAR[_\-]?\d{1,2})", up)
        if m:
            key = m.group(1).replace("-", "_")
            mm = re.search(r"(\d{1,2})", key)
            if mm:
                key = f"CHAR_{int(mm.group(1)):02d}"
            mapping[key] = str(f)
            continue

        # CHAR_p1 / p01
        m2 = re.search(r"CHAR[_\-]?P[_\-]?(\d{1,2})", up)
        if m2:
            key = f"CHAR_{int(m2.group(1)):02d}"
            mapping[key] = str(f)
            continue

        # fallback: if contains a number 1..99 and contains CHAR somewhere
        if "CHAR" in up:
            m3 = re.search(r"(\d{1,2})", up)
            if m3:
                key = f"CHAR_{int(m3.group(1)):02d}"
                mapping.setdefault(key, str(f))

    return mapping

def get_char_image_path(char_id: str, folder: str) -> Optional[str]:
    char_id = (char_id or "").strip().upper().replace("-", "_")
    m = re.search(r"(\d{1,2})", char_id)
    if m:
        char_id = f"CHAR_{int(m.group(1)):02d}"

    mp = scan_character_images(folder)
    if char_id in mp:
        return mp[char_id]

    # last resort: try any file containing the digits
    if m:
        digits = int(m.group(1))
        p = Path(folder)
        if p.exists():
            for f in p.glob("*"):
                if f.is_file() and str(digits) in f.name:
                    return str(f)
    return None

# ============================================================
# 7) QUBO one-hot (simple SA sampler)
# ============================================================
def build_qubo_onehot(scores: np.ndarray, P: float) -> np.ndarray:
    """
    Minimize E(x) = sum_i (-score_i) x_i + P*(sum x - 1)^2
    x_i in {0,1}
    """
    n = len(scores)
    Q = np.zeros((n,n), dtype=float)

    # linear term: -score_i
    for i in range(n):
        Q[i,i] += -float(scores[i])

    # penalty: P*(sum x - 1)^2 = P*(sum x^2 + 2 sum_{i<j} x_i x_j - 2 sum x + 1)
    # since x^2=x, => P*(sum x + 2 sum_{i<j} x_i x_j - 2 sum x + 1)
    # => P*(-sum x + 2 sum_{i<j} x_i x_j) + const
    for i in range(n):
        Q[i,i] += -P
    for i in range(n):
        for j in range(i+1, n):
            Q[i,j] += 2*P
            Q[j,i] += 2*P
    return Q

def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    x = x.astype(float)
    return float(x @ Q @ x)

def sa_sample(Q: np.ndarray, sweeps: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    n = Q.shape[0]
    x = rng.integers(0,2,size=n).astype(int)
    # ensure not all zero sometimes
    if x.sum() == 0:
        x[rng.integers(0,n)] = 1

    for _ in range(max(10, int(sweeps))):
        for i in range(n):
            # delta E if flip i
            xi = x[i]
            # ΔE = (1-2xi)*(Q_ii + 2*sum_{j!=i} Q_ij x_j)
            s = Q[i,i] + 2.0*np.dot(Q[i,:], x) - 2.0*Q[i,i]*x[i]
            dE = (1 - 2*xi) * s
            if dE <= 0:
                x[i] = 1 - xi
            else:
                if rng.random() < np.exp(-beta*dE):
                    x[i] = 1 - xi
    return x

def sample_distribution(Q: np.ndarray, n_samples: int, sweeps: int, beta: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    counts = np.zeros(n, dtype=int)
    energies = np.zeros(n_samples, dtype=float)

    for k in range(n_samples):
        x = sa_sample(Q, sweeps=sweeps, beta=beta, rng=rng)
        # pick argmin among ones if multiple
        on = np.where(x==1)[0]
        if len(on) == 0:
            idx = int(rng.integers(0,n))
        else:
            # choose the best index (lowest local energy)
            local = []
            for i in on:
                xx = np.zeros(n, dtype=int); xx[i]=1
                local.append(qubo_energy(Q, xx))
            idx = int(on[int(np.argmin(local))])
        counts[idx] += 1
        energies[k] = qubo_energy(Q, x)

    prob = counts / max(1, counts.sum())
    return prob, energies

# ============================================================
# 8) App UI
# ============================================================
st.title("🔮 Q-Quest 量子神託（QUBO / STAGE×QUOTES）")

# ---- Sidebar : Excel & params ----
with st.sidebar:
    st.markdown("### 📂 データ")
    up = st.file_uploader("統合Excel（pack）", type=["xlsx"], key="pack_uploader")
    st.markdown("### 🖼 画像フォルダ（相対/絶対）")
    img_folder = st.text_input("例: ./assets/images/characters", value="./assets/images/characters", key="img_folder")

    st.markdown("---")
    st.markdown("### 🍁 季節×時間（Stage）")
    st.toggle("現在時刻から自動推定（簡易）", value=True, key="auto_stage")
    st.caption("STAGE_ID は『季節×時間の状態』です。Excel側の STAGE_TO_VOW がある場合、誓願に“季節の流れ”を混ぜます。")
    stage_id = st.selectbox("STAGE_ID（手動上書き可）", options=["ST_01","ST_02","ST_03","ST_04"], index=0, key="stage_id")

    st.markdown("---")
    st.markdown("### 🎲 揺らぎ（観測のブレ）")
    st.slider("β（大→最小エネルギー寄り / 小→多様）", 0.2, 4.0, 2.2, 0.1, key="beta")
    st.slider("微小ノイズ（エネルギーに加える）", 0.0, 0.20, 0.08, 0.01, key="eps_noise")
    st.slider("サンプル数（観測分布）", 50, 800, 300, 10, key="n_samples")
    st.slider("SA sweeps（揺らぎ）", 50, 1000, 420, 10, key="sweeps")

    st.markdown("---")
    st.markdown("### 🧩 QUBO設定（one-hot）")
    st.slider("one-hot ペナルティ P", 1.0, 80.0, 40.0, 1.0, key="P")

    st.markdown("---")
    st.markdown("### 🧠 テキスト＋誓願（自動ベクトル化）")
    st.slider("n-gram（簡易）", 1, 5, 3, 1, key="ngram")
    st.slider("mix比率 α（1=スライダー寄り / 0=テキスト寄り）", 0.0, 1.0, 0.55, 0.01, key="alpha")

    st.markdown("---")
    st.markdown("### 💬 QUOTES神託（温度付きで選択）")
    st.selectbox("LANG", options=["ja","en"], index=0, key="lang")
    st.slider("格言温度（高→ランダム / 低→上位固定）", 0.2, 2.5, 1.2, 0.1, key="quote_tau")

# ---- Load pack or fallback ----
sheets = {}
sheet_msg = "Excel未指定（デモ動作）"
if up is not None:
    b = up.getvalue()
    h = sha(b)
    try:
        sheets = load_excel_pack(b, h)
        # show primary status
        sheet_msg = f"Excel読込OK（sheets: {len(sheets)}）"
    except Exception as e:
        sheets = {}
        sheet_msg = f"Excel読込エラー: {e}"

st.success(sheet_msg)

# ---- Identify key sheets ----
# Expected (flexible):
# - VOW_MASTER
# - CHAR_MASTER
# - CHAR_TO_VOW
# - STAGE_TO_VOW (optional)
# - QUOTES
sh_char_to_vow_name, df_char_to_vow = find_sheet(sheets, ["CHAR_TO_VOW","CHAR2VOW","CHAR-VOW","CHAR_TO_VOWS"])
sh_vow_master_name, df_vow_master = find_sheet(sheets, ["VOW_MASTER","VOW","VOWS","VOW_LIST"])
sh_char_master_name, df_char_master = find_sheet(sheets, ["CHAR_MASTER","CHAR","CHAR_LIST","CHARACTERS"])
sh_stage_to_vow_name, df_stage_to_vow = find_sheet(sheets, ["STAGE_TO_VOW","STAGE2VOW","STAGE-VOW"])
sh_quotes_name, df_quotes = find_sheet(sheets, ["QUOTES","QUOTE","格言","格言一覧"])

# fallback: minimal demo char_to_vow if missing
if df_char_to_vow is None or len(df_char_to_vow)==0:
    # build demo
    vow_cols_demo = [f"VOW_{i:02d}" for i in range(1, 13)]
    rows = []
    for i in range(1, 13):
        r = {"CHAR_ID": f"CHAR_{i:02d}"}
        for j in range(1, 13):
            r[f"VOW_{j:02d}"] = 0.2 if i==j else 0.05
        rows.append(r)
    df_char_to_vow = pd.DataFrame(rows)

# detect vow columns leniently
vow_cols = detect_vow_columns(df_char_to_vow)
if len(vow_cols)==0:
    # last fallback: assume VOW_01..12 exist
    for i in range(1,13):
        c = f"VOW_{i:02d}"
        if c in df_char_to_vow.columns:
            vow_cols.append(c)

# master tables
df_vows = build_master_vows(df_vow_master, vow_cols)
df_chars = build_master_chars(df_char_master, df_char_to_vow)

# normalize char_to_vow columns
cols_map = {norm_col(c): c for c in df_char_to_vow.columns}
char_id_col = cols_map.get("CHAR_ID") or cols_map.get("CHAR") or cols_map.get("ID")
if not char_id_col:
    # if no char_id col, create from row index
    df_char_to_vow = df_char_to_vow.copy()
    df_char_to_vow.insert(0, "CHAR_ID", [f"CHAR_{i:02d}" for i in range(1, len(df_char_to_vow)+1)])
    char_id_col = "CHAR_ID"

# build char->vow weight matrix aligned
dfW = df_char_to_vow[[char_id_col] + vow_cols].copy()
dfW = dfW.rename(columns={char_id_col: "CHAR_ID"})
dfW["CHAR_ID"] = dfW["CHAR_ID"].astype(str).str.strip()

# align chars order
char_ids = df_chars["CHAR_ID"].astype(str).str.strip().tolist()
dfW = dfW.set_index("CHAR_ID").reindex(char_ids).fillna(0.0).reset_index()

# stage vector (optional)
stage_vec = np.zeros(len(vow_cols), dtype=float)
if df_stage_to_vow is not None and len(df_stage_to_vow)>0:
    scols = {norm_col(c): c for c in df_stage_to_vow.columns}
    sid = scols.get("STAGE_ID") or scols.get("STAGE") or scols.get("ID")
    if sid:
        tmp = df_stage_to_vow.copy()
        tmp[sid] = tmp[sid].astype(str).str.strip()
        row = tmp[tmp[sid]==st.session_state.get("stage_id","ST_01")]
        if len(row)>0:
            row = row.iloc[0]
            sv = []
            for c in vow_cols:
                if c in df_stage_to_vow.columns:
                    sv.append(float(row.get(c, 0.0) or 0.0))
                else:
                    sv.append(0.0)
            stage_vec = np.array(sv, dtype=float)

# quotes
dfQ = load_quotes(df_quotes)

# ============================================================
# Step1: Input
# ============================================================
left, right = st.columns([2.15, 1.0], gap="large")

with left:
    st.markdown("## Step 1：誓願入力（スライダー）＋テキスト（自動ベクトル化）")
    user_text = st.text_area(
        "あなたの状況を一文で（例：疲れていて決断ができない / 新しい挑戦が怖い など）",
        value="",
        height=90,
        key="user_text",
        placeholder="例：迷いを断ちたいが、今は待つべきか？"
    )
    st.caption("スライダー入力は TITLE を常時表示し、テキストからの自動推定と mix します。")

    # build sliders per vow
    slider_vals = []
    vow_titles = {r["VOW_ID"]: r["TITLE"] for _, r in df_vows.iterrows()}
    for c in vow_cols:
        idx = int(re.search(r"(\d{1,2})", norm_col(c)).group(1)) if re.search(r"\d", norm_col(c)) else 0
        vow_id = f"VOW_{idx:02d}" if idx>0 else norm_col(c)
        title = vow_titles.get(vow_id, vow_id)
        v = st.slider(f"{vow_id}｜{title}", 0.0, 4.0, 0.0, 0.5, key=f"sl_{vow_id}")
        slider_vals.append(v)
    slider_vec = np.array(slider_vals, dtype=float)

# ============================================================
# text -> vow vector (simple n-gram-ish match against titles)
# ============================================================
def text_to_vow_vec(text: str, vows_df: pd.DataFrame, vow_cols: List[str], ngram: int) -> np.ndarray:
    text = (text or "").strip()
    if not text:
        return np.zeros(len(vow_cols), dtype=float)

    # build tokens by char ngram
    t = re.sub(r"\s+", "", text)
    grams = []
    n = max(1, int(ngram))
    if len(t) <= n:
        grams = [t]
    else:
        grams = [t[i:i+n] for i in range(len(t)-n+1)]

    # score vow title hits
    scores = np.zeros(len(vow_cols), dtype=float)
    titles = vows_df["TITLE"].astype(str).tolist()
    for i, title in enumerate(titles[:len(vow_cols)]):
        tt = re.sub(r"\s+", "", str(title))
        hit = 0
        for g in grams:
            if g and g in tt:
                hit += 1
        scores[i] = hit

    # normalize to 0..4 scale
    if scores.max() > 0:
        scores = 4.0 * (scores / scores.max())
    return scores

text_vec = text_to_vow_vec(user_text, df_vows, vow_cols, st.session_state.get("ngram",3))

alpha = float(st.session_state.get("alpha",0.55))
mix_vec = alpha*slider_vec + (1.0-alpha)*text_vec

# blend stage (small)
mix_vec2 = mix_vec + 0.25*stage_vec

# ============================================================
# Step3 (right): QUBO observe
# ============================================================
with right:
    st.markdown("## Step 3：結果（観測された神＋理由＋QUOTES神託）")

    # score per character = dot(mix_vec2, W_char) + noise
    W = dfW[vow_cols].values.astype(float)  # shape (n_char, n_vow)
    base_scores = (W @ mix_vec2.reshape(-1,1)).reshape(-1)

    rng = np.random.default_rng(make_seed(user_text + "|" + str(mix_vec2.sum())))
    eps = float(st.session_state.get("eps_noise",0.08))
    noisy_scores = base_scores + rng.normal(0, eps, size=len(base_scores))

    # energies: lower is better
    energies = -noisy_scores

    # QUBO one-hot
    P = float(st.session_state.get("P", 40.0))
    Q = build_qubo_onehot(scores=noisy_scores, P=P)

    prob, sampleE = sample_distribution(
        Q,
        n_samples=int(st.session_state.get("n_samples",300)),
        sweeps=int(st.session_state.get("sweeps",420)),
        beta=float(st.session_state.get("beta",2.2)),
        seed=make_seed(user_text + "|qubo"),
    )

    # ranking table
    df_rank = pd.DataFrame({
        "順位": np.arange(1, len(char_ids)+1),
        "CHAR_ID": char_ids,
        "神": df_chars["神"].astype(str).tolist(),
        "energy（低いほど選ばれやすい）": energies,
        "確率（sample）": prob
    }).sort_values("energy（低いほど選ばれやすい）", ascending=True).reset_index(drop=True)
    df_rank["順位"] = np.arange(1, len(df_rank)+1)

    st.dataframe(df_rank.head(10), use_container_width=True, hide_index=True)

    # observed: sample from prob (single observation)
    obs_idx = int(np.argmax(prob)) if prob.sum() > 0 else int(np.argmin(energies))
    obs_char = char_ids[obs_idx]
    obs_god = str(df_chars.loc[df_chars["CHAR_ID"]==obs_char, "神"].values[0]) if (df_chars["CHAR_ID"]==obs_char).any() else obs_char

    st.markdown(f"### 🌟 今回“観測”された神：{obs_god}（{obs_char}）")

    # character image
    img_path = get_char_image_path(obs_char, st.session_state.get("img_folder","./assets/images/characters"))
    if img_path and Path(img_path).exists():
        st.image(img_path, use_container_width=True, caption=f"{obs_god}（{Path(img_path).name}）")
    else:
        st.warning(
            f"キャラクター画像が見つかりません（探索フォルダ: {st.session_state.get('img_folder')} / CHAR_ID: {obs_char}）\n"
            f"※ assets/images/characters 配下のファイル名に CHAR_01 か 1 などが含まれるようにしてください。"
        )

    # reason (Top VOW contributions)
    contrib = mix_vec2 * W[obs_idx]  # elementwise
    df_top = pd.DataFrame({
        "VOW": [f"VOW_{int(re.search(r'(\\d{1,2})', norm_col(c)).group(1)):02d}" if re.search(r'\d', norm_col(c)) else norm_col(c) for c in vow_cols],
        "TITLE": [df_vows.iloc[i]["TITLE"] if i < len(df_vows) else "" for i in range(len(vow_cols))],
        "mix(v)": mix_vec2,
        "W(char,v)": W[obs_idx],
        "寄与(v*w)": contrib
    }).sort_values("寄与(v*w)", ascending=False).reset_index(drop=True)

    st.markdown("### 🧩 寄与した誓願（Top）")
    st.dataframe(df_top.head(6), use_container_width=True, hide_index=True)

    # message line
    top_titles = df_top.head(4)["TITLE"].astype(str).tolist()
    stage_label = st.session_state.get("stage_id","ST_01")
    st.markdown(
        f"<div class='card' style='background:rgba(40,120,80,0.25); border-color: rgba(80,200,140,0.25)'>"
        f"いまの波：<b>{'・'.join([t for t in top_titles if t])}</b> に寄っている。<br/>"
        f"季節×時間（Stage）: <b>{stage_label}</b> は流れを強める。</div>",
        unsafe_allow_html=True
    )

    # QUOTES (temperature)
    st.markdown("### 🗣 QUOTES神託（温度付きで選択）")
    qpick = pick_quotes_by_temperature(
        dfQ,
        lang=st.session_state.get("lang","ja"),
        k=3,
        tau=float(st.session_state.get("quote_tau",1.2)),
        seed=make_seed(user_text + "|quotes"),
    )
    for i in range(len(qpick)):
        qt = str(qpick.loc[i,"QUOTE"])
        src = str(qpick.loc[i,"SOURCE"]) if "SOURCE" in qpick.columns else "—"
        st.markdown(
            f"<div class='card' style='background:rgba(40,90,160,0.18)'>"
            f"<b>神託{i+1}</b><br/>"
            f"『{qt}』<br/>"
            f"<span class='smallnote'>— {src}</span></div>",
            unsafe_allow_html=True
        )

# ============================================================
# Step4 : keyword extract + word sphere art
# ============================================================
st.markdown("## 4）テキストのキーワード抽出（簡易）")

kw = extract_keywords(user_text, top_n=6)
colA, colB = st.columns([1.0, 1.6], gap="large")
with colA:
    st.markdown("### 抽出キーワード")
    if kw:
        st.markdown("**" + " / ".join(kw) + "**")
        st.caption("※簡易抽出です（形態素解析なし）。短文だと少なくなることがあります。")
    else:
        st.info("入力が短い/空のため、キーワードが抽出できません。")

with colB:
    st.markdown("### 🌐 単語の球体（誓願→キーワード→縁のネットワーク）")
    seed = make_seed(user_text + "|sphere")
    fig = plot_word_sphere(center_words=GLOBAL_WORDS_DATABASE, user_keywords=kw, seed=seed, star_count=850)
    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": True,
        "scrollZoom": True,
        "displaylogo": False,
        "doubleClick": "reset",
    })

# ============================================================
# Debug helper (optional)
# ============================================================
with st.expander("🔧 Excel検出デバッグ（シート名・列名）", expanded=False):
    st.write("検出シート:")
    st.write({
        "CHAR_TO_VOW": sh_char_to_vow_name,
        "VOW_MASTER": sh_vow_master_name,
        "CHAR_MASTER": sh_char_master_name,
        "STAGE_TO_VOW": sh_stage_to_vow_name,
        "QUOTES": sh_quotes_name,
    })
    st.write("CHAR_TO_VOW columns:")
    st.write(list(df_char_to_vow.columns))
    st.write("検出された VOW列:", vow_cols)
    st.write("画像フォルダ:", st.session_state.get("img_folder"))
    st.write("スキャン結果（一部）:", dict(list(scan_character_images(st.session_state.get("img_folder")).items())[:10]))
