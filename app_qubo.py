# -*- coding: utf-8 -*-
# ============================================================
# Q-Quest 量子神託 (QUBO / STAGE×QUOTES)  完全版
# - 黒基調UI（アップロード/入力/表/グラフも黒）
# - Step3で「選ばれた神」画像を必ず表示（globで自動探索）
# - 表は pandas Styler でヘッダ含め黒化
# - Step4に「キーワード抽出」+「単語の球体（縁のネットワーク）」を表示
# - Excelの「VOW_01...列」検出を緩く（正規表現 + 大小/記号揺れ吸収）
# ============================================================

import os
import re
import io
import math
import time
import zlib
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ----------------------------
# 0) Page config（最初のStreamlitコマンド）
# ----------------------------
st.set_page_config(
    page_title="Q-Quest 量子神託（QUBO / STAGE×QUOTES）",
    layout="wide",
)

# ----------------------------
# 1) CSS（黒テーマを全領域に強制）
# ----------------------------
APP_CSS = r"""
<style>
/* 全体背景 */
.stApp{
  background:
    radial-gradient(circle at 18% 22%, rgba(110,150,255,0.12), transparent 38%),
    radial-gradient(circle at 78% 68%, rgba(255,160,220,0.08), transparent 44%),
    radial-gradient(circle at 50% 50%, rgba(255,255,255,0.03), transparent 55%),
    linear-gradient(180deg, rgba(6,8,18,1), rgba(10,12,26,1));
}

/* 余白 */
.block-container{ padding-top: 1.0rem; padding-bottom: 2rem; }

/* 文字 */
html, body, [class*="css"]  { color: rgba(245,245,255,0.92); }
h1,h2,h3,h4{
  font-family: "Hiragino Mincho ProN", "Yu Mincho", "Noto Serif JP", serif !important;
  font-weight: 650 !important;
  color: rgba(245,245,255,0.96) !important;
  text-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

/* サイドバー */
section[data-testid="stSidebar"]{
  background: rgba(10,12,26,0.90);
  border-right: 1px solid rgba(255,255,255,0.10);
  backdrop-filter: blur(10px);
}
section[data-testid="stSidebar"] *{ color: rgba(245,245,255,0.92) !important; }

/* 入力（text_input / text_area / select / number etc） */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stSelectbox"] div[role="combobox"],
div[data-testid="stNumberInput"] input{
  background: rgba(255,255,255,0.06) !important;
  color: rgba(245,245,255,0.95) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 12px !important;
}

/* ファイルアップローダ（白地対策） */
div[data-testid="stFileUploader"] section{
  background: rgba(255,255,255,0.06) !important;
  border: 1px dashed rgba(255,255,255,0.22) !important;
  border-radius: 14px !important;
}
div[data-testid="stFileUploader"] *{
  color: rgba(245,245,255,0.92) !important;
}
div[data-testid="stFileUploader"] button{
  background: rgba(255,255,255,0.10) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  color: rgba(245,245,255,0.95) !important;
  border-radius: 10px !important;
}

/* ボタン */
.stButton>button{
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.20) !important;
  background: linear-gradient(90deg, rgba(90,80,255,0.65), rgba(160,90,255,0.55)) !important;
  color: rgba(255,255,255,0.96) !important;
  box-shadow: 0 16px 50px rgba(0,0,0,0.25);
}
.stButton>button:hover{
  filter: brightness(1.08);
}

/* カード */
.card{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 14px 14px 12px 14px;
  box-shadow: 0 18px 60px rgba(0,0,0,0.20);
}
.smallnote{ opacity:0.80; font-size:0.92rem; }

/* Plotlyの外枠 */
div[data-testid="stPlotlyChart"] > div{
  position: relative;
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 18px 60px rgba(0,0,0,0.30);
}
div[data-testid="stPlotlyChart"] > div::after{
  content:"";
  position:absolute;
  inset:0;
  background:
    radial-gradient(circle at 30% 25%, rgba(120,160,255,0.10), transparent 45%),
    radial-gradient(circle at 70% 65%, rgba(255,180,220,0.06), transparent 52%),
    radial-gradient(circle at 50% 50%, rgba(0,0,0,0.00), rgba(0,0,0,0.40));
  pointer-events:none;
}

/* Streamlitのalert背景を少し暗く */
div[data-testid="stAlert"]{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
}

/* テーブル系（念押し） */
table{
  color: rgba(245,245,255,0.94) !important;
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# ----------------------------
# 2) ユーティリティ
# ----------------------------
def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def norm_col(s: str) -> str:
    """列名ゆらぎ吸収：全角/半角/記号/空白を寄せる"""
    s = str(s)
    s = s.strip()
    s = s.replace("　", " ")
    s = re.sub(r"\s+", "", s)
    s = s.replace("-", "_").replace("—", "_")
    s = s.upper()
    return s

VOW_COL_RE = re.compile(r"^VOW[_]?\d{1,2}$", re.IGNORECASE)

def find_vow_columns(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    mapping = {c: norm_col(c) for c in cols}
    vow_cols = [c for c in cols if VOW_COL_RE.match(mapping[c])]
    # 例: VOW01 みたいなパターンも拾う
    vow_cols2 = [c for c in cols if re.match(r"^VOW\d{1,2}$", mapping[c])]
    vow_cols = list(dict.fromkeys(vow_cols + vow_cols2))
    return vow_cols

def dark_df(df: pd.DataFrame, precision: int = 6):
    """pandas Stylerでテーブルを黒化（ヘッダも黒）"""
    if df is None:
        return None
    d2 = df.copy()
    # 数値丸め（見やすさ）
    for c in d2.columns:
        if pd.api.types.is_numeric_dtype(d2[c]):
            d2[c] = d2[c].astype(float).round(precision)

    styles = [
        dict(selector="th",
             props=[
                 ("background-color", "rgba(10,12,26,0.95)"),
                 ("color", "rgba(245,245,255,0.95)"),
                 ("border", "1px solid rgba(255,255,255,0.14)"),
                 ("font-weight", "700"),
             ]),
        dict(selector="td",
             props=[
                 ("background-color", "rgba(255,255,255,0.04)"),
                 ("color", "rgba(245,245,255,0.92)"),
                 ("border", "1px solid rgba(255,255,255,0.10)"),
             ]),
        dict(selector="table",
             props=[
                 ("border-collapse", "separate"),
                 ("border-spacing", "0"),
                 ("border", "1px solid rgba(255,255,255,0.12)"),
                 ("border-radius", "14px"),
                 ("overflow", "hidden"),
             ]),
    ]
    return d2.style.set_table_styles(styles)

def read_excel_sheets(excel_bytes: bytes) -> Dict[str, pd.DataFrame]:
    bio = io.BytesIO(excel_bytes)
    xls = pd.ExcelFile(bio, engine="openpyxl")
    out = {}
    for name in xls.sheet_names:
        try:
            out[name] = pd.read_excel(bio, sheet_name=name, engine="openpyxl")
        except Exception:
            # 1枚壊れてても全体は止めない
            pass
    return out

def pick_sheet(sheets: Dict[str, pd.DataFrame], prefer_keywords: List[str]) -> Optional[Tuple[str, pd.DataFrame]]:
    """sheet名にキーワードを含むもの優先で選ぶ"""
    if not sheets:
        return None
    items = list(sheets.items())
    # まず完全一致/部分一致
    for kw in prefer_keywords:
        for name, df in items:
            if kw.upper() in name.upper():
                return name, df
    # 次に VOW列を持つシート
    for name, df in items:
        if len(find_vow_columns(df)) >= 6:
            return name, df
    # 最後は先頭
    name, df = items[0]
    return name, df

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

# ----------------------------
# 3) キーワード抽出（簡易）
# ----------------------------
STOP_TOKENS = set([
    "した","たい","いる","こと","それ","これ","ため","よう","ので","から",
    "です","ます","ある","ない","そして","でも","しかし","また",
    "自分","私","あなた","もの","感じ","気持ち","今日",
    "に","を","が","は","と","も","で","へ","や","の"
])

def extract_keywords_simple(text: str, top_n: int = 5) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    text_clean = re.sub(r"[0-9０-９、。．,.!！?？\(\)\[\]{}「」『』\"'：:;／/\\\n\r\t]+", " ", text)
    tokens = [t.strip() for t in re.split(r"\s+", text_clean) if t.strip()]
    tokens = [t for t in tokens if (len(t) >= 2 and t not in STOP_TOKENS)]
    # 長い順 + 出現順のバランス
    tokens = sorted(list(dict.fromkeys(tokens)), key=lambda s: (-len(s), s))
    return tokens[:top_n]

# ----------------------------
# 4) 単語球体（あなたの球体ロジックを簡略統合）
# ----------------------------
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
    "行動": ["努力","継続","忍耐","誠実","正直","挑戦"],
    "哲学": ["調和","バランス","自然","美","道","真実","自由","正義"],
    "関係": ["絆","つながり","家族","友人","仲間","信頼","尊敬","協力"],
    "内的": ["静けさ","集中","覚悟","決意","勇気","強さ","柔軟性","寛容"],
    "時間": ["今","瞬間","過程","変化","進化","発展","循環","流れ"],
}

def calc_sim(w1: str, w2: str) -> float:
    if w1 == w2:
        return 1.0
    common = set(w1) & set(w2)
    char_sim = len(common) / max(len(set(w1)), len(set(w2)), 1)
    category_sim = 0.0
    for _, ws in CATEGORIES.items():
        a = w1 in ws
        b = w2 in ws
        if a and b:
            category_sim = 1.0
            break
        elif a or b:
            category_sim = max(category_sim, 0.3)
    len_sim = 1.0 - abs(len(w1) - len(w2)) / max(len(w1), len(w2), 1)
    return float(np.clip(0.4*char_sim + 0.4*category_sim + 0.2*len_sim, 0, 1))

def energy_between(w1: str, w2: str, rng: np.random.Generator, jitter: float) -> float:
    s = calc_sim(w1, w2)
    e = -2.0 * s + 0.5
    if set(w1) & set(w2):
        e -= 0.15
    for _, ws in CATEGORIES.items():
        if (w1 in ws) and (w2 in ws):
            e -= 0.45
            break
    if jitter > 0:
        e += rng.normal(0, jitter)
    return float(e)

def build_network(center_words: List[str], n_total: int, rng: np.random.Generator, jitter: float) -> Dict:
    all_words = list(dict.fromkeys(center_words + GLOBAL_WORDS_DATABASE))
    energies = {}
    for w in all_words:
        if w in center_words:
            energies[w] = -3.0
        else:
            e_list = [energy_between(c, w, rng, jitter) for c in center_words] if center_words else [0.0]
            energies[w] = float(np.mean(e_list))

    sorted_words = sorted(energies.items(), key=lambda x: x[1])
    selected = []
    for w, _ in sorted_words:
        if w in center_words and w not in selected:
            selected.append(w)
    for w, _ in sorted_words:
        if w not in selected:
            selected.append(w)
        if len(selected) >= n_total:
            break

    n = len(selected)
    Q = np.zeros((n, n), dtype=float)
    np.fill_diagonal(Q, -0.5)
    for i in range(n):
        for j in range(i+1, n):
            e = energy_between(selected[i], selected[j], rng, jitter)
            Q[i, j] = e
            Q[j, i] = e

    center_indices = [i for i, w in enumerate(selected) if w in center_words]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if Q[i, j] < -0.25:
                edges.append((i, j, float(Q[i, j])))

    return {
        "words": selected,
        "energies": {w: energies[w] for w in selected},
        "Q": Q,
        "edges": edges,
        "center_indices": center_indices,
    }

def solve_positions(Q: np.ndarray, words: List[str], center_indices: List[int],
                    energies: Dict[str, float], rng: np.random.Generator,
                    n_iterations: int = 80) -> np.ndarray:
    n = len(words)
    pos = rng.normal(0, 1, size=(n, 3)) * 0.6

    # centers at origin
    for ci in center_indices:
        pos[ci] = np.array([0.0, 0.0, 0.0])

    ev = list(energies.values()) if energies else [-3, 3]
    mn, mx = float(min(ev)), float(max(ev))
    er = (mx - mn) if mx != mn else 1.0

    # simple relax
    for _ in range(n_iterations):
        for i in range(n):
            if i in center_indices:
                continue
            force = np.zeros(3, dtype=float)

            # pull to center with target radius by energy
            w = words[i]
            e = energies.get(w, 0.0)
            norm = (e - mn) / er
            target = 0.5 + (1.0 - norm) * 2.0
            for ci in center_indices[:1] if center_indices else []:
                vec = pos[ci] - pos[i]
                d = np.linalg.norm(vec) + 1e-6
                if d < target*0.9:
                    force -= vec/d * 0.04
                elif d > target*1.1:
                    force += vec/d * 0.08

            # pairwise based on Q
            for j in range(n):
                if i == j:
                    continue
                vec = pos[j] - pos[i]
                d = np.linalg.norm(vec) + 1e-6
                eij = Q[i, j]
                if eij < -0.25:   # attract
                    force += vec/d * (abs(eij)*0.035)
                elif eij > 0.20:  # repel
                    force -= vec/d * (abs(eij)*0.015)

            pos[i] += force * 0.30

    return pos

def plot_word_sphere(center_words: List[str], seed_key: str,
                     n_total: int = 28, jitter: float = 0.10,
                     iters: int = 80, noise: float = 0.06) -> go.Figure:
    seed = int(zlib.adler32(seed_key.encode("utf-8")) & 0xFFFFFFFF)
    rng = np.random.default_rng(seed)

    center_words = [w for w in center_words if w]
    network = build_network(center_words, n_total=n_total, rng=rng, jitter=jitter)
    pos = solve_positions(network["Q"], network["words"], network["center_indices"], network["energies"], rng=rng, n_iterations=iters)
    if noise > 0:
        pos = pos + rng.normal(0, noise, size=pos.shape)

    words = network["words"]
    energies = network["energies"]
    center_set = set(center_words)
    edges = network["edges"]
    center_indices = network["center_indices"]

    fig = go.Figure()

    # stars (fixed)
    star_rng = np.random.default_rng(12345)
    star_count = 650
    sx = star_rng.uniform(-3.2, 3.2, star_count)
    sy = star_rng.uniform(-2.4, 2.4, star_count)
    sz = star_rng.uniform(-2.0, 2.0, star_count)
    fig.add_trace(go.Scatter3d(
        x=sx, y=sy, z=sz, mode="markers",
        marker=dict(size=star_rng.uniform(1.0, 2.2, star_count),
                    color=[f"rgba(255,255,255,{a})" for a in np.full(star_count, 0.20)]),
        hoverinfo="skip", showlegend=False
    ))

    # edges
    xE, yE, zE = [], [], []
    for i, j, _e in edges:
        x0, y0, z0 = pos[i]
        x1, y1, z1 = pos[j]
        xE += [x0, x1, None]
        yE += [y0, y1, None]
        zE += [z0, z1, None]
    fig.add_trace(go.Scatter3d(
        x=xE, y=yE, z=zE, mode="lines",
        line=dict(width=1, color="rgba(200,220,255,0.22)"),
        hoverinfo="skip", showlegend=False
    ))

    # nodes
    sizes, colors, labels = [], [], []
    for w in words:
        e = energies.get(w, 0.0)
        if w in center_set:
            sizes.append(26); colors.append("rgba(255,235,100,0.98)"); labels.append(w)
        else:
            en = min(1.0, abs(e)/3.0)
            sizes.append(10 + int(10*en))
            colors.append("rgba(220,240,255,0.70)" if e < -0.7 else "rgba(255,255,255,0.55)")
            labels.append(w)

    center_idx = [i for i, w in enumerate(labels) if w in center_set]
    other_idx  = [i for i, w in enumerate(labels) if w not in center_set]

    if other_idx:
        oi = np.array(other_idx, dtype=int)
        fig.add_trace(go.Scatter3d(
            x=pos[oi,0], y=pos[oi,1], z=pos[oi,2],
            mode="markers+text",
            text=[labels[i] for i in oi],
            textposition="top center",
            textfont=dict(size=16, color="rgba(255,255,255,1.0)"),
            marker=dict(size=[sizes[i] for i in oi], color=[colors[i] for i in oi],
                        line=dict(width=1, color="rgba(0,0,0,0.12)")),
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False
        ))

    if center_idx:
        ci = np.array(center_idx, dtype=int)
        fig.add_trace(go.Scatter3d(
            x=pos[ci,0], y=pos[ci,1], z=pos[ci,2],
            mode="markers+text",
            text=[labels[i] for i in ci],
            textposition="top center",
            textfont=dict(size=22, color="rgba(255,80,80,1.0)"),
            marker=dict(size=[sizes[i] for i in ci], color=[colors[i] for i in ci],
                        line=dict(width=2, color="rgba(255,80,80,0.8)")),
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False
        ))

    fig.update_layout(
        paper_bgcolor="rgba(6,8,18,1)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(6,8,18,1)",
            camera=dict(eye=dict(x=1.6, y=1.15, z=1.05)),
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig

# ----------------------------
# 5) キャラ画像探索（最重要：globで確実に見つける）
# ----------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")

def find_character_image(char_id: str, folder: str) -> Optional[Path]:
    if not char_id:
        return None
    base = char_id.upper().strip()
    folder_path = Path(folder)
    if not folder_path.exists():
        return None

    # 例: CHAR_01, char_01, CHAR-01 など揺れを許容
    candidates = []
    # 完全一致優先
    for p in folder_path.glob(f"{base}*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            candidates.append(p)
    # さらに緩く（アンダースコア無しも）
    if not candidates:
        key = base.replace("_", "")
        for p in folder_path.glob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                name_key = p.stem.upper().replace("_", "").replace("-", "")
                if key in name_key:
                    candidates.append(p)

    # p1優先など軽い優先度
    def score(p: Path) -> Tuple[int, int]:
        s = p.stem.upper()
        return (0 if "P1" in s else 1, len(s))
    candidates = sorted(candidates, key=score)
    return candidates[0] if candidates else None

# ----------------------------
# 6) Excel → 必要データを“緩く”抽出
# ----------------------------
def build_data_from_excel(excel_bytes: bytes) -> Dict:
    sheets = read_excel_sheets(excel_bytes)

    # 典型名を優先（あなたのpackに合わせて）
    char_to_vow = pick_sheet(sheets, ["CHAR_TO_VOW", "CHAR2VOW", "CHAR-VOW", "CHARVOW"])
    vow_master  = pick_sheet(sheets, ["VOW", "VOWS", "VOW_MASTER", "VOW_LIST"])
    stage_to_vow = pick_sheet(sheets, ["STAGE_TO_VOW", "STAGE2VOW", "STAGE-VOW", "STAGE"])
    quotes_sheet = pick_sheet(sheets, ["QUOTES", "QUOTE"])

    out = {
        "sheets": list(sheets.keys()),
        "char_to_vow_sheet": char_to_vow[0] if char_to_vow else None,
        "vow_master_sheet": vow_master[0] if vow_master else None,
        "stage_to_vow_sheet": stage_to_vow[0] if stage_to_vow else None,
        "quotes_sheet": quotes_sheet[0] if quotes_sheet else None,
        "char_to_vow": char_to_vow[1] if char_to_vow else None,
        "vow_master": vow_master[1] if vow_master else None,
        "stage_to_vow": stage_to_vow[1] if stage_to_vow else None,
        "quotes": quotes_sheet[1] if quotes_sheet else None,
    }
    return out

def ensure_vow_id_list(vow_master_df: Optional[pd.DataFrame], fallback_n: int = 12) -> List[str]:
    if vow_master_df is not None and len(vow_master_df.columns) > 0:
        cols = [norm_col(c) for c in vow_master_df.columns]
        # VOW_ID列っぽいのを探す
        id_col = None
        for cand in ["VOW_ID", "VOWID", "ID"]:
            if cand in cols:
                id_col = vow_master_df.columns[cols.index(cand)]
                break
        if id_col is not None:
            ids = [str(x).strip() for x in vow_master_df[id_col].tolist() if str(x).strip() and str(x).lower() != "nan"]
            # VOW_01形式に寄せる
            fixed = []
            for v in ids:
                v2 = v.upper().replace("-", "_")
                if re.match(r"^VOW_\d{1,2}$", v2) or re.match(r"^VOW\d{1,2}$", v2):
                    if not v2.startswith("VOW_"):
                        v2 = "VOW_" + v2.replace("VOW", "")
                    n = int(re.findall(r"\d+", v2)[0])
                    fixed.append(f"VOW_{n:02d}")
            fixed = list(dict.fromkeys(fixed))
            if fixed:
                return fixed

    # fallback: VOW_01..VOW_12
    return [f"VOW_{i:02d}" for i in range(1, fallback_n+1)]

def get_vow_titles(vow_master_df: Optional[pd.DataFrame], vow_ids: List[str]) -> Dict[str, str]:
    titles = {vid: vid for vid in vow_ids}
    if vow_master_df is None:
        return titles
    cols_norm = [norm_col(c) for c in vow_master_df.columns]

    # title列候補
    title_col = None
    for cand in ["TITLE", "名称", "名前", "LABEL"]:
        if cand.upper() in cols_norm:
            title_col = vow_master_df.columns[cols_norm.index(cand.upper())]
            break

    # id列候補
    id_col = None
    for cand in ["VOW_ID", "VOWID", "ID"]:
        if cand.upper() in cols_norm:
            id_col = vow_master_df.columns[cols_norm.index(cand.upper())]
            break

    if id_col is None or title_col is None:
        return titles

    for _, r in vow_master_df.iterrows():
        vid_raw = str(r.get(id_col, "")).strip().upper().replace("-", "_")
        if not vid_raw:
            continue
        if not vid_raw.startswith("VOW_"):
            if re.match(r"^VOW\d{1,2}$", vid_raw):
                n = int(re.findall(r"\d+", vid_raw)[0])
                vid_raw = f"VOW_{n:02d}"
        if vid_raw in titles:
            t = str(r.get(title_col, "")).strip()
            if t and t.lower() != "nan":
                titles[vid_raw] = t
    return titles

def build_char_list(char_to_vow_df: Optional[pd.DataFrame]) -> List[str]:
    if char_to_vow_df is None:
        return []
    cols_norm = [norm_col(c) for c in char_to_vow_df.columns]
    id_col = None
    for cand in ["CHAR_ID", "CHARID", "神", "ID"]:
        if cand.upper() in cols_norm:
            id_col = char_to_vow_df.columns[cols_norm.index(cand.upper())]
            break
    if id_col is None:
        # 先頭列をid扱い
        id_col = char_to_vow_df.columns[0]
    ids = [str(x).strip() for x in char_to_vow_df[id_col].tolist() if str(x).strip() and str(x).lower() != "nan"]
    # CHAR_01形式へ寄せる
    out = []
    for c in ids:
        c2 = c.upper().replace("-", "_")
        if re.match(r"^CHAR_\d{1,2}$", c2) or re.match(r"^CHAR\d{1,2}$", c2):
            if not c2.startswith("CHAR_"):
                c2 = "CHAR_" + c2.replace("CHAR", "")
            n = int(re.findall(r"\d+", c2)[0])
            out.append(f"CHAR_{n:02d}")
        else:
            out.append(c2)
    return list(dict.fromkeys(out))

def build_char_vow_weight_matrix(char_to_vow_df: Optional[pd.DataFrame], vow_ids: List[str]) -> pd.DataFrame:
    """
    CHAR_TO_VOW から
    - 行: CHAR_ID
    - 列: VOW_01.. の重み
    を作る（列検出を“緩く”）
    """
    if char_to_vow_df is None:
        return pd.DataFrame()

    cols = list(char_to_vow_df.columns)
    cols_norm = [norm_col(c) for c in cols]

    # char id col
    id_col = None
    for cand in ["CHAR_ID", "CHARID", "神", "ID"]:
        if cand.upper() in cols_norm:
            id_col = cols[cols_norm.index(cand.upper())]
            break
    if id_col is None:
        id_col = cols[0]

    # vow weight cols（緩い検出）
    vow_cols = find_vow_columns(char_to_vow_df)
    # vow_ids に一致する列へ寄せる（例: VOW01 -> VOW_01）
    rename_map = {}
    for c in vow_cols:
        key = norm_col(c)
        # VOW01 / VOW_1 / VOW_01 -> VOW_01
        m = re.findall(r"\d{1,2}", key)
        if not m:
            continue
        n = int(m[0])
        rename_map[c] = f"VOW_{n:02d}"

    df = char_to_vow_df[[id_col] + vow_cols].copy()
    df = df.rename(columns=rename_map)
    # 欠けてるVOW列は0埋め
    for vid in vow_ids:
        if vid not in df.columns:
            df[vid] = 0.0

    # index = char
    df[id_col] = df[id_col].astype(str).str.strip().str.upper().str.replace("-", "_")
    df = df.set_index(id_col)

    # 数値化
    for vid in vow_ids:
        df[vid] = pd.to_numeric(df[vid], errors="coerce").fillna(0.0)

    # CHAR_01形式へ寄せたindexにする
    new_index = []
    for c in df.index.tolist():
        c2 = str(c).upper().replace("-", "_")
        if re.match(r"^CHAR\d{1,2}$", c2):
            n = int(re.findall(r"\d+", c2)[0])
            c2 = f"CHAR_{n:02d}"
        elif re.match(r"^CHAR_\d{1,2}$", c2):
            n = int(re.findall(r"\d+", c2)[0])
            c2 = f"CHAR_{n:02d}"
        new_index.append(c2)
    df.index = new_index

    return df[vow_ids]

def build_stage_bias(stage_to_vow_df: Optional[pd.DataFrame], vow_ids: List[str]) -> pd.DataFrame:
    """
    STAGE_TO_VOW:
    - 行: STAGE_ID
    - 列: VOW_01.. のバイアス
    """
    if stage_to_vow_df is None:
        return pd.DataFrame()

    cols = list(stage_to_vow_df.columns)
    cols_norm = [norm_col(c) for c in cols]

    id_col = None
    for cand in ["STAGE_ID", "STAGEID", "STAGE", "ID"]:
        if cand.upper() in cols_norm:
            id_col = cols[cols_norm.index(cand.upper())]
            break
    if id_col is None:
        id_col = cols[0]

    vow_cols = find_vow_columns(stage_to_vow_df)
    rename_map = {}
    for c in vow_cols:
        key = norm_col(c)
        m = re.findall(r"\d{1,2}", key)
        if not m:
            continue
        n = int(m[0])
        rename_map[c] = f"VOW_{n:02d}"

    df = stage_to_vow_df[[id_col] + vow_cols].copy().rename(columns=rename_map)
    df[id_col] = df[id_col].astype(str).str.strip().str.upper().str.replace("-", "_")
    df = df.set_index(id_col)

    for vid in vow_ids:
        if vid not in df.columns:
            df[vid] = 0.0
        df[vid] = pd.to_numeric(df[vid], errors="coerce").fillna(0.0)

    return df[vow_ids]

def pick_quotes(quotes_df: Optional[pd.DataFrame], lang: str = "ja") -> pd.DataFrame:
    if quotes_df is None or quotes_df.empty:
        return pd.DataFrame(columns=["QUOTE_ID","QUOTE","SOURCE"])

    cols = list(quotes_df.columns)
    cols_norm = [norm_col(c) for c in cols]

    def pick(*cands):
        for cand in cands:
            if cand.upper() in cols_norm:
                return cols[cols_norm.index(cand.upper())]
        return None

    qid = pick("QUOTE_ID","ID","Q_ID")
    qt  = pick("QUOTE","格言","言葉","テキスト")
    src = pick("SOURCE","出典","作者","出所")

    if qt is None:
        # 2列目を格言扱いにするなど最低限で救う
        qt = cols[0]

    out = pd.DataFrame()
    out["QUOTE_ID"] = quotes_df[qid] if qid else range(1, len(quotes_df)+1)
    out["QUOTE"]    = quotes_df[qt].astype(str)
    out["SOURCE"]   = quotes_df[src].astype(str) if src else ""
    out = out.replace("nan", "", regex=False)
    out = out[out["QUOTE"].str.strip() != ""]
    return out.head(2000)

# ----------------------------
# 7) QUBO（one-hot：1神を選ぶ）
# ----------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    return ex / s if s > 0 else np.ones_like(x)/len(x)

def solve_one_hot_by_sa(energies: np.ndarray, penalty_p: float = 40.0,
                        sweeps: int = 420, rng: Optional[np.random.Generator] = None) -> int:
    """
    energies: shape (N,) 低いほど選ばれやすい
    one-hot: sum x = 1
    E = sum_i e_i x_i + P (sum_i x_i - 1)^2
    SAで0/1を更新し最小を探す（簡易）
    """
    rng = rng or np.random.default_rng(0)
    n = energies.shape[0]

    # 初期：ランダムに1つだけ1
    x = np.zeros(n, dtype=int)
    x[rng.integers(0, n)] = 1

    def cost(xv):
        s = int(np.sum(xv))
        return float(np.dot(energies, xv) + penalty_p * (s - 1)**2)

    best_x = x.copy()
    best_c = cost(x)
    cur_c = best_c

    # 温度スケジュール
    T0, T1 = 2.5, 0.05
    for t in range(sweeps):
        T = T0 * ((T1/T0) ** (t / max(1, sweeps-1)))

        # 1ビット反転
        i = int(rng.integers(0, n))
        x2 = x.copy()
        x2[i] = 1 - x2[i]
        c2 = cost(x2)
        dc = c2 - cur_c
        if dc <= 0 or rng.random() < math.exp(-dc / max(1e-9, T)):
            x = x2
            cur_c = c2
            if cur_c < best_c:
                best_c = cur_c
                best_x = x.copy()

    # one-hotになっていなければ最小エネルギー1点に強制
    if np.sum(best_x) != 1:
        idx = int(np.argmin(energies))
        best_x = np.zeros(n, dtype=int)
        best_x[idx] = 1

    return int(np.argmax(best_x))

# ----------------------------
# 8) UI（サイドバー）
# ----------------------------
st.sidebar.markdown("## 📁 データ")

uploaded = st.sidebar.file_uploader("統合Excel（pack）", type=["xlsx"])

st.sidebar.markdown("### 🖼 画像フォルダ（相対/絶対）")
img_folder = st.sidebar.text_input("例: ./assets/images/characters", value="./assets/images/characters")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🌸 季節×時間（Stage）")
auto_stage = st.sidebar.toggle("現在時刻から自動推定（簡易）", value=True)

# STAGE_IDはExcelに合わせる前提だが、無い場合でも表示自体は止めない
stage_id = st.sidebar.text_input("STAGE_ID（手動上書き可）", value="ST_01")

if auto_stage:
    # 超簡易：時間帯だけで振る（あなたのStage設計に合わせて後で調整OK）
    hour = time.localtime().tm_hour
    if 5 <= hour < 11:
        stage_id = "ST_01"
    elif 11 <= hour < 17:
        stage_id = "ST_02"
    elif 17 <= hour < 23:
        stage_id = "ST_03"
    else:
        stage_id = "ST_04"

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙ QUBO設定（one-hot）")
penalty_p = st.sidebar.slider("one-hot ペナルティP", 1.0, 120.0, 40.0, 1.0)
sample_n  = st.sidebar.slider("サンプル数（観測分布）", 50, 2000, 300, 50)
sa_sweeps = st.sidebar.slider("SA sweeps（揺らぎ）", 50, 1200, 420, 10)
sa_temp   = st.sidebar.slider("SA温度（大→揺らぐ）", 0.2, 3.0, 1.2, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🧩 テキスト＋誓願（自動ベクトル化）")
ngram = st.sidebar.number_input("n-gram（簡易）", min_value=1, max_value=5, value=3, step=1)
mix_a = st.sidebar.slider("mix比率 a（1=スライダー寄り / 0=テキスト寄り）", 0.0, 1.0, 0.55, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🟦 QUOTES神託（温度付きで選択）")
lang = st.sidebar.selectbox("LANG", ["ja", "en"], index=0)
quote_temp = st.sidebar.slider("格言温度（高=ランダム/低=上位固定）", 0.1, 3.0, 1.2, 0.05)

# ----------------------------
# 9) メイン（Excel読み込み）
# ----------------------------
st.markdown("## 🔮 Q-Quest 量子神託（QUBO / STAGE×QUOTES）")

if uploaded is None:
    st.info("左の「統合Excel（pack）」をアップロードしてください。")
    st.stop()

excel_bytes = uploaded.getvalue()
data = build_data_from_excel(excel_bytes)

# ここで “シート検出が厳しすぎる” を潰す：検出結果を見せる
with st.expander("🧪 Excel検出デバッグ（シート名・検出結果）", expanded=False):
    st.write("Sheets:", data["sheets"])
    st.write("CHAR_TO_VOW:", data["char_to_vow_sheet"])
    st.write("VOW_MASTER :", data["vow_master_sheet"])
    st.write("STAGE_TO_VOW:", data["stage_to_vow_sheet"])
    st.write("QUOTES:", data["quotes_sheet"])

vow_ids = ensure_vow_id_list(data["vow_master"], fallback_n=12)
vow_titles = get_vow_titles(data["vow_master"], vow_ids)

W_char_vow = build_char_vow_weight_matrix(data["char_to_vow"], vow_ids)   # rows=CHAR, cols=VOW
B_stage_vow = build_stage_bias(data["stage_to_vow"], vow_ids)             # rows=STAGE, cols=VOW
Q_quotes = pick_quotes(data["quotes"], lang=lang)

if W_char_vow.empty:
    st.error("Excelから CHAR_TO_VOW（キャラ×誓願重み）が作れませんでした。シート/列名のどれかが想定外です。")
    st.stop()

char_ids = list(W_char_vow.index)

# ステージバイアス（無ければゼロ）
stage_bias = np.zeros(len(vow_ids), dtype=float)
if (B_stage_vow is not None) and (not B_stage_vow.empty) and (stage_id in B_stage_vow.index):
    stage_bias = B_stage_vow.loc[stage_id].to_numpy(dtype=float)

# ----------------------------
# 10) Step1：誓願入力（スライダー＋テキスト）
# ----------------------------
left, right = st.columns([2.2, 1.0], gap="large")

with left:
    st.markdown("### Step 1：誓願入力（スライダー）＋テキスト（自動ベクトル化）")
    user_text = st.text_area(
        "あなたの状況を一文で（例：疲れていて決断ができない／新しい挑戦が怖い など）",
        value="迷いを断ちたいが、今は待つべきか？",
        height=90
    )

    st.caption("スライダー入力はTITLEを常時表示し、テキストからの自動推定と mix します。")

    # スライダー（0〜5）
    manual = {}
    for vid in vow_ids:
        t = vow_titles.get(vid, vid)
        manual[vid] = st.slider(f"{vid}｜{t}", 0.0, 5.0, 0.0, 0.5)

    manual_vec = np.array([manual[v] for v in vow_ids], dtype=float)

    # テキスト簡易ベクトル（超簡易：VOWタイトルとの文字一致 / n-gram）
    # ※あなたの既存ロジックに差し替え可能。UI確認優先で軽量化。
    def text_to_vow_vec(text: str, vow_titles_map: Dict[str, str], n: int = 3) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(len(vow_ids), dtype=float)
        text_u = text
        grams = set([text_u[i:i+n] for i in range(max(0, len(text_u)-n+1))]) if len(text_u) >= n else set([text_u])
        out = np.zeros(len(vow_ids), dtype=float)
        for i, vid in enumerate(vow_ids):
            title = vow_titles_map.get(vid, vid)
            t = title
            if not t:
                continue
            tgrams = set([t[j:j+n] for j in range(max(0, len(t)-n+1))]) if len(t) >= n else set([t])
            inter = len(grams & tgrams)
            out[i] = inter
        # 0〜5にゆるく正規化
        if np.max(out) > 0:
            out = 5.0 * (out / np.max(out))
        return out

    auto_vec = text_to_vow_vec(user_text, vow_titles, n=int(ngram))

    mix_vec = mix_a * manual_vec + (1.0 - mix_a) * auto_vec
    mix_df = pd.DataFrame({
        "VOW": vow_ids,
        "TITLE": [vow_titles.get(v, v) for v in vow_ids],
        "manual(0-5)": manual_vec,
        "auto(0-5)": auto_vec,
        "mix(0-5)": mix_vec
    })
    with st.expander("🧷 誓願ベクトル（manual / auto / mix）", expanded=False):
        st.dataframe(dark_df(mix_df, precision=3), use_container_width=True, hide_index=True)

# ----------------------------
# 11) エネルギー計算（CHAR候補）
# ----------------------------
# energy = - (mix_vec • W_char_vow[char]) - (stage_bias • W_char_vow[char])  ※低いほど選ばれやすい
W = W_char_vow.to_numpy(dtype=float)  # shape (C, V)
mix = mix_vec.astype(float)
bias = stage_bias.astype(float)

energies = - (W @ mix) - (W @ bias)

# softmax（見せる用）
probs = softmax(-energies)  # energy低いほど確率高い

# ----------------------------
# 12) QUBO one-hot 観測（分布）
# ----------------------------
# 温度をSAの揺らぎに反映（簡易）
rng0 = np.random.default_rng(int(zlib.adler32((user_text + stage_id).encode("utf-8")) & 0xFFFFFFFF))

obs = []
for k in range(int(sample_n)):
    # SA温度に応じてエネルギーへ微小ノイズ（=観測揺らぎ）
    noisy = energies + rng0.normal(0, float(sa_temp)*0.15, size=energies.shape)
    idx = solve_one_hot_by_sa(noisy, penalty_p=float(penalty_p), sweeps=int(sa_sweeps), rng=rng0)
    obs.append(idx)

obs = np.array(obs, dtype=int)
counts = np.bincount(obs, minlength=len(char_ids)).astype(int)
obs_prob = counts / max(1, counts.sum())

best_idx = int(np.argmax(obs_prob))
best_char = char_ids[best_idx]
best_energy = float(energies[best_idx])

# Step3用：ランキング表
rank_df = pd.DataFrame({
    "順位": np.arange(1, min(10, len(char_ids))+1),
    "CHAR_ID": [char_ids[i] for i in np.argsort(energies)[:min(10, len(char_ids))]],
    "energy（低いほど選ばれやすい）": [float(energies[i]) for i in np.argsort(energies)[:min(10, len(char_ids))]],
    "確率（softmax）": [float(probs[i]) for i in np.argsort(energies)[:min(10, len(char_ids))]],
})

# ----------------------------
# 13) Step3：結果表示（あなたの理想UI寄せ）
# ----------------------------
with right:
    st.markdown("### Step 3：結果（観測された神＋理由＋QUOTES神託）")

    # 上部ランキング（黒テーブル）
    st.dataframe(dark_df(rank_df, precision=6), use_container_width=True, hide_index=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"#### 🌟 今回“観測”された神：**{best_char}**")
    st.markdown(f"<div class='smallnote'>※ここは「単発の観測（1回抽選）」ではなく、{int(sample_n)}回の観測分布の最頻値です。</div>", unsafe_allow_html=True)

    # 画像表示（ここが今回の最重要改善）
    img_path = find_character_image(best_char, img_folder)
    if img_path and img_path.exists():
        st.image(str(img_path), use_container_width=True, caption=f"{best_char}（{img_path.name}）")
    else:
        st.warning(f"※キャラクター画像が見つかりません（探索フォルダ: {img_folder} / CHAR_ID: {best_char}）")

        # デバッグ：フォルダ内の先頭数個を出す（視認用）
        folder_path = Path(img_folder)
        if folder_path.exists():
            files = [p.name for p in folder_path.glob("*") if p.suffix.lower() in IMG_EXTS]
            st.caption(f"フォルダ内画像（先頭20件）: {files[:20]}")
        else:
            st.caption("画像フォルダ自体が存在しません。パスを ./assets/images/characters のように指定してください。")

    # “いまの波”（以前の薄緑×濃緑の雰囲気を継承）
    top_vows = np.argsort(-mix_vec)[:4]
    vow_phrase = "・".join([vow_titles.get(vow_ids[i], vow_ids[i]) for i in top_vows])
    st.markdown(
        f"""
        <div style="
          background: rgba(40,120,80,0.35);
          border: 1px solid rgba(90,220,160,0.28);
          color: rgba(210,255,230,0.95);
          padding: 12px 12px;
          border-radius: 14px;
          box-shadow: 0 18px 60px rgba(0,0,0,0.16);
          ">
          <b>いまの波：</b> {vow_phrase} に寄っている。<br/>
          <span style="opacity:0.85;">季節×時間（Stage）: {stage_id}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 寄与した誓願（Top）も黒テーブルで
    contrib_df = pd.DataFrame({
        "VOW": [vow_ids[i] for i in top_vows],
        "TITLE": [vow_titles.get(vow_ids[i], vow_ids[i]) for i in top_vows],
        "mix(v)": [float(mix_vec[i]) for i in top_vows],
    })
    st.markdown("#### 🧩 寄与した誓願（Top）")
    st.dataframe(dark_df(contrib_df, precision=3), use_container_width=True, hide_index=True)

    # QUOTES神託（以前の「濃い青×青文字」継承）
    st.markdown("#### 🟦 QUOTES神託（温度付きで選択）")

    def sample_quote(dfq: pd.DataFrame, temperature: float, rng: np.random.Generator) -> Tuple[str, str]:
        if dfq is None or dfq.empty:
            return ("（格言データがありません）", "")
        # 温度：低いほど上位固定、高いほどランダム
        # ここでは「全文長いほど上位」みたいな仮スコア（あなたの既存ロジックに差し替えOK）
        lens = dfq["QUOTE"].fillna("").astype(str).apply(len).to_numpy(dtype=float)
        score = lens
        # 温度で確率を平坦化
        t = max(0.05, float(temperature))
        p = softmax(score / t)
        idx = int(rng.choice(len(dfq), p=p))
        q = str(dfq.iloc[idx]["QUOTE"])
        s = str(dfq.iloc[idx].get("SOURCE", ""))
        return (q, s)

    rngq = np.random.default_rng(int(zlib.adler32((best_char + user_text).encode("utf-8")) & 0xFFFFFFFF))
    q1, s1 = sample_quote(Q_quotes, quote_temp, rngq)
    q2, s2 = sample_quote(Q_quotes, quote_temp, rngq)
    q3, s3 = sample_quote(Q_quotes, quote_temp, rngq)

    def quote_card(title, q, src):
        st.markdown(
            f"""
            <div style="
              background: rgba(35,70,130,0.38);
              border: 1px solid rgba(120,170,255,0.22);
              color: rgba(210,235,255,0.96);
              padding: 12px 12px;
              border-radius: 14px;
              margin-bottom: 10px;
              ">
              <b>{title}</b><br/>
              「{q}」<br/>
              <span style="opacity:0.85;">— {src}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    quote_card("神託1", q1, s1)
    quote_card("神託2", q2, s2)
    quote_card("神託3", q3, s3)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# 14) 2)〜4) グラフ群（黒背景）
# ----------------------------
st.markdown("---")
st.markdown("### 2) エネルギー地形（全候補）")

# bar chart（Plotly：黒背景）
order = np.argsort(energies)
show_n = min(12, len(char_ids))
xlabels = [char_ids[i] for i in order[:show_n]]
yvals = [float(energies[i]) for i in order[:show_n]]

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(x=xlabels, y=yvals))
fig_bar.update_layout(
    paper_bgcolor="rgba(6,8,18,1)",
    plot_bgcolor="rgba(6,8,18,1)",
    font=dict(color="rgba(245,245,255,0.92)"),
    margin=dict(l=10, r=10, t=40, b=40),
    xaxis=dict(tickangle=-90, gridcolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.10)"),
    title="energy（低いほど選ばれやすい）",
)
st.plotly_chart(fig_bar, use_container_width=True, config={"displaylogo": False})

st.markdown("### 4) テキストのキーワード抽出（簡易）＋ 単語の球体")
kw = extract_keywords_simple(user_text, top_n=5)

c1, c2 = st.columns([1.0, 1.4], gap="large")
with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 抽出キーワード")
    if kw:
        st.write(" / ".join(kw))
        st.caption("※ここを起点に、エネルギーが近い単語を空間に配置し、縁（線）で結びます。")
    else:
        st.info("入力テキストが短すぎる/抽出できませんでした。もう少し文章を増やしてください。")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 🌐 単語の球体（誓願→キーワード→縁のネットワーク）")
    if kw:
        fig_sphere = plot_word_sphere(
            center_words=kw[:2],  # 中心語は2つ程度が見やすい
            seed_key=user_text + "|" + stage_id,
            n_total=28,
            jitter=0.10,
            iters=80,
            noise=0.06,
        )
        st.plotly_chart(fig_sphere, use_container_width=True, config={
            "displayModeBar": True,
            "scrollZoom": True,
            "displaylogo": False,
            "doubleClick": "reset",
        })
    else:
        st.info("キーワードが無いので球体表示できません。")
    st.markdown("</div>", unsafe_allow_html=True)
