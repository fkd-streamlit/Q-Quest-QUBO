# -*- coding: utf-8 -*-
# ============================================================
# Q-Quest 量子神託（QUBO / STAGE×QUOTES）
# - ダークUI（入力/アップロード/表/グラフ背景を黒系に統一）
# - Excel列名検出を「ゆるく」して VOW_01... が無くても拾う
# - Step3 で「選ばれたキャラクター画像」を必ず表示（あれば）
# - Step4 で「キーワード抽出」＋「単語の球体（縁のネットワーク）」を表示
# ============================================================

import os
import re
import io
import math
import time
import zlib
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# MUST be first Streamlit command
# -----------------------------
st.set_page_config(
    page_title="Q-Quest 量子神託（QUBO / STAGE×QUOTES）",
    page_icon="🔮",
    layout="wide",
)

# ============================================================
# 0) Dark UI CSS（入力/アップロード/表/グラフを黒）
# ============================================================
DARK_CSS = """
<style>
/* ---- App background ---- */
.stApp{
  background:
    radial-gradient(circle at 18% 24%, rgba(110,150,255,0.12), transparent 38%),
    radial-gradient(circle at 78% 68%, rgba(255,160,220,0.08), transparent 44%),
    radial-gradient(circle at 50% 50%, rgba(255,255,255,0.03), transparent 55%),
    linear-gradient(180deg, rgba(6,8,18,1), rgba(10,12,26,1));
  color: rgba(245,245,255,0.92);
}
.block-container{ padding-top: 1.2rem; }

/* ---- Typography ---- */
h1,h2,h3{
  font-family: "Hiragino Mincho ProN","Yu Mincho","Noto Serif JP",serif !important;
  font-weight: 650 !important;
  color: rgba(245,245,255,0.95) !important;
  text-shadow: 0 2px 10px rgba(0,0,0,0.35);
}
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li{
  font-family: "Hiragino Mincho ProN","Yu Mincho","Noto Serif JP",serif;
  letter-spacing: 0.02em;
  color: rgba(245,245,255,0.92);
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.06);
  border-right: 1px solid rgba(255,255,255,0.10);
  backdrop-filter: blur(12px);
}
section[data-testid="stSidebar"] *{
  color: rgba(245,245,255,0.92);
}

/* ---- Inputs (text/number/select) ---- */
textarea, input, .stTextInput input, .stTextArea textarea{
  background: rgba(12,14,30,0.92) !important;
  color: rgba(245,245,255,0.95) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}
textarea::placeholder, input::placeholder{
  color: rgba(220,220,240,0.55) !important;
}

/* ---- Select / multiselect container ---- */
div[data-baseweb="select"] > div{
  background: rgba(12,14,30,0.92) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}
div[data-baseweb="select"] *{
  color: rgba(245,245,255,0.92) !important;
}

/* ---- File uploader (the "white on white" fix) ---- */
div[data-testid="stFileUploader"]{
  background: rgba(12,14,30,0.92) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 14px !important;
  padding: 10px 12px !important;
}
div[data-testid="stFileUploader"] *{
  color: rgba(245,245,255,0.92) !important;
}
div[data-testid="stFileUploaderDropzone"]{
  background: rgba(12,14,30,0.92) !important;
  border: 1px dashed rgba(255,255,255,0.22) !important;
  border-radius: 12px !important;
}
div[data-testid="stFileUploaderDropzone"] *{
  color: rgba(245,245,255,0.85) !important;
}

/* ---- Buttons ---- */
.stButton>button{
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(18,22,44,0.72);
  color: rgba(245,245,255,0.92);
}
.stButton>button:hover{
  background: rgba(28,34,64,0.82);
}

/* ---- "Card" blocks ---- */
.card{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 16px 16px 12px 16px;
  box-shadow: 0 18px 60px rgba(0,0,0,0.18);
}

/* ---- Green vibe (旧コードの薄緑/濃緑) ---- */
.quote-green{
  background: rgba(16, 92, 58, 0.45);
  border: 1px solid rgba(80, 220, 150, 0.28);
  color: rgba(210, 255, 230, 0.96);
  border-radius: 14px;
  padding: 12px 14px;
}

/* ---- Blue vibe (QUOTES神託：薄青/濃青文字) ---- */
.quote-blue{
  background: rgba(22, 70, 140, 0.40);
  border: 1px solid rgba(120, 180, 255, 0.24);
  color: rgba(210, 235, 255, 0.96);
  border-radius: 14px;
  padding: 12px 14px;
}

/* ---- DataFrame (st.dataframe) dark theme ---- */
div[data-testid="stDataFrame"]{
  background: rgba(12,14,30,0.92) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 14px !important;
  overflow: hidden !important;
}
div[data-testid="stDataFrame"] *{
  color: rgba(245,245,255,0.92) !important;
}
div[data-testid="stDataFrame"] thead tr th{
  background: rgba(10,12,26,0.98) !important;
}
div[data-testid="stDataFrame"] tbody tr{
  background: rgba(12,14,30,0.92) !important;
}

/* ---- Plotly container ---- */
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
    radial-gradient(circle at 50% 50%, rgba(0,0,0,0.00), rgba(0,0,0,0.38));
  pointer-events:none;
}

/* ---- Minor text ---- */
.smallnote{opacity:0.78; font-size:0.92rem;}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ============================================================
# 1) Helpers
# ============================================================

VOW_N = 12

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def make_seed(s: str) -> int:
    return int(zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF)

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    t = max(1e-6, float(temp))
    y = x / t
    y = y - np.max(y)
    e = np.exp(y)
    return e / np.sum(e)

# ============================================================
# 2) Excel loader (ゆるい列名検出)
# ============================================================

@dataclass
class PackData:
    char_to_vow: pd.DataFrame        # rows: char_id, cols: vow_01..12
    vow_master: pd.DataFrame         # rows: vow_id, title
    quotes: pd.DataFrame             # quotes table
    stage_master: pd.DataFrame       # stage table (optional)
    meta: Dict[str, str]

def _normalize_col_key(s: str) -> str:
    s = str(s).strip()
    s = s.replace("　", " ")
    s = re.sub(r"\s+", "_", s)
    return s

def detect_vow_columns(df: pd.DataFrame) -> List[str]:
    """
    厳密に VOW_01 が無くても拾う。
    例: vow_1, VOW1, VOW-01, VOW 01, v01, v_01, 01, 1 などを許容しないと危険なので
    「VOWっぽい列」を優先し、無ければ列名一覧を出してエラー。
    """
    cols = [str(c) for c in df.columns]
    up = [c.upper() for c in cols]

    idx_map: Dict[int, str] = {}

    # 1) "VOW" を含む列を最優先
    for c, cu in zip(cols, up):
        m = re.search(r"(VOW|VOY|V)\s*[_\-\s]*0*([1-9]|1[0-2])\b", cu)
        if m:
            k = int(m.group(2))
            idx_map[k] = c

    # 2) それでも足りない場合、"01".. "12" 単独に近い列名も拾う（危険なので控えめ）
    if len(idx_map) < VOW_N:
        for c, cu in zip(cols, up):
            # 例: "01" "1" "V01" が取りこぼされる場合の救済
            m = re.fullmatch(r"0*([1-9]|1[0-2])", cu)
            if m:
                k = int(m.group(1))
                if k not in idx_map:
                    idx_map[k] = c

    missing = [i for i in range(1, VOW_N + 1) if i not in idx_map]
    if missing:
        # 列候補提示（ユーザーがExcelを直せるように）
        sample_cols = ", ".join(cols[:40])
        raise ValueError(
            "VOWの重み列を十分に検出できませんでした。\n"
            f"不足: {missing}\n"
            "期待: VOW_01..VOW_12（または vow_1, VOW1, VOW-01 など）\n"
            f"検出した列名（一部）: {sample_cols}"
        )

    # i=1..12 の順
    return [idx_map[i] for i in range(1, VOW_N + 1)]

def find_sheet_name(xls: pd.ExcelFile, candidates: List[str]) -> Optional[str]:
    names = list(xls.sheet_names)
    up = {n.upper(): n for n in names}
    for c in candidates:
        if c.upper() in up:
            return up[c.upper()]
    # fallback: partial match
    for n in names:
        for c in candidates:
            if c.upper() in n.upper():
                return n
    return None

@st.cache_data(show_spinner=False)
def load_pack_from_excel_bytes(excel_bytes: bytes, file_hash: str) -> PackData:
    bio = io.BytesIO(excel_bytes)
    xls = pd.ExcelFile(bio, engine="openpyxl")

    # 想定シート名（多少のブレを許容）
    s_char = find_sheet_name(xls, ["CHAR_TO_VOW", "CHAR2VOW", "CHAR-VOW", "CHAR_TO_VOWS"])
    s_vow  = find_sheet_name(xls, ["VOW", "VOW_MASTER", "VOWS"])
    s_q    = find_sheet_name(xls, ["QUOTES", "QUOTE", "格言"])
    s_stg  = find_sheet_name(xls, ["STAGE", "STAGES", "季節", "STAGE_MASTER"])

    if not s_char:
        raise ValueError(f"必須シートが見つかりません: CHAR_TO_VOW（類似名も可） / シート一覧: {xls.sheet_names}")

    df_char = pd.read_excel(xls, sheet_name=s_char).copy()
    df_char.columns = [_normalize_col_key(c) for c in df_char.columns]

    # char_id列を推定
    id_candidates = ["CHAR_ID", "CHAR", "ID", "キャラID", "神_ID", "神ID"]
    char_id_col = None
    for c in df_char.columns:
        if c.upper() in [x.upper() for x in id_candidates]:
            char_id_col = c
            break
    if char_id_col is None:
        # 先頭列をID扱い
        char_id_col = df_char.columns[0]

    vow_cols_raw = detect_vow_columns(df_char)

    # 統一列名 VOW_01..12 を作る
    vow_cols_std = [f"VOW_{i:02d}" for i in range(1, VOW_N + 1)]
    df_ctv = df_char[[char_id_col] + vow_cols_raw].copy()
    df_ctv = df_ctv.rename(columns={char_id_col: "CHAR_ID", **{vow_cols_raw[i]: vow_cols_std[i] for i in range(VOW_N)}})
    df_ctv["CHAR_ID"] = df_ctv["CHAR_ID"].astype(str).str.strip()
    for c in vow_cols_std:
        df_ctv[c] = pd.to_numeric(df_ctv[c], errors="coerce").fillna(0.0)

    # VOW master（タイトル）
    if s_vow:
        df_vow = pd.read_excel(xls, sheet_name=s_vow).copy()
        df_vow.columns = [_normalize_col_key(c) for c in df_vow.columns]
    else:
        df_vow = pd.DataFrame({
            "VOW_ID": vow_cols_std,
            "TITLE": [f"VOW_{i:02d}" for i in range(1, VOW_N + 1)]
        })

    # vow id/title列を推定
    vow_id_col = None
    vow_title_col = None
    for c in df_vow.columns:
        if c.upper() in ["VOW_ID", "VOW", "ID"]:
            vow_id_col = c
        if c.upper() in ["TITLE", "名前", "誓願", "誓願名", "タイトル"]:
            vow_title_col = c
    if vow_id_col is None:
        vow_id_col = df_vow.columns[0]
    if vow_title_col is None:
        vow_title_col = df_vow.columns[1] if len(df_vow.columns) > 1 else df_vow.columns[0]

    df_vow2 = df_vow[[vow_id_col, vow_title_col]].copy()
    df_vow2 = df_vow2.rename(columns={vow_id_col: "VOW_ID", vow_title_col: "TITLE"})
    df_vow2["VOW_ID"] = df_vow2["VOW_ID"].astype(str).str.strip()
    df_vow2["TITLE"] = df_vow2["TITLE"].astype(str).str.strip()

    # QUOTES
    if s_q:
        df_q = pd.read_excel(xls, sheet_name=s_q).copy()
        df_q.columns = [_normalize_col_key(c) for c in df_q.columns]
    else:
        df_q = pd.DataFrame(columns=["QUOTE_ID", "QUOTE", "SOURCE", "KEYWORDS"])

    # STAGE（任意）
    if s_stg:
        df_st = pd.read_excel(xls, sheet_name=s_stg).copy()
        df_st.columns = [_normalize_col_key(c) for c in df_st.columns]
    else:
        df_st = pd.DataFrame(columns=["STAGE_ID", "TITLE"])

    meta = {
        "sheet_char_to_vow": s_char,
        "sheet_vow": s_vow or "(none)",
        "sheet_quotes": s_q or "(none)",
        "sheet_stage": s_stg or "(none)",
    }
    return PackData(char_to_vow=df_ctv, vow_master=df_vow2, quotes=df_q, stage_master=df_st, meta=meta)

# ============================================================
# 3) Text -> VOW auto estimation (簡易)
# ============================================================

STOP_TOKENS = set([
    "した","たい","いる","こと","それ","これ","ため","よう","ので","から",
    "です","ます","ある","ない","そして","でも","しかし","また",
    "自分","私","あなた","もの","感じ","気持ち","今日",
    "に","を","が","は","と","も","で","へ","や","の","な","だ"
])

def extract_keywords_simple(text: str, max_n: int = 8) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # まずは日本語の「連続文字」を雑に拾う（形態素なし）
    t = re.sub(r"[0-9０-９、。．,.!！?？\(\)\[\]{}「」『』\"'：:;／/\\\n\r\t]+", " ", text)
    toks = [x.strip() for x in re.split(r"\s+", t) if x.strip()]
    toks = [x for x in toks if len(x) >= 2 and x not in STOP_TOKENS]
    # 長い順
    toks = sorted(list(dict.fromkeys(toks)), key=lambda s: (-len(s), s))
    return toks[:max_n]

def build_vow_title_map(vow_master: pd.DataFrame) -> Dict[str, str]:
    m = {}
    for _, r in vow_master.iterrows():
        vid = str(r.get("VOW_ID","")).strip()
        ttl = str(r.get("TITLE","")).strip()
        if vid:
            m[vid] = ttl
    return m

def auto_vow_from_text(text: str, vow_titles: Dict[str, str]) -> np.ndarray:
    """
    超簡易:
    - 入力文に VOWタイトルの部分一致があれば加点
    - キーワードが「挑戦/迷い/静けさ...」などタイトルに含まれれば加点
    """
    v = np.zeros(VOW_N, dtype=float)
    t = (text or "").strip()
    if not t:
        return v

    kws = extract_keywords_simple(t, max_n=10)
    vow_ids = [f"VOW_{i:02d}" for i in range(1, VOW_N + 1)]

    for i, vid in enumerate(vow_ids, start=0):
        title = vow_titles.get(vid, vid)
        score = 0.0

        # タイトルの直ヒット
        if title and title != vid and title in t:
            score += 4.0

        # キーワードがタイトルに含まれる
        for k in kws:
            if title and k in title:
                score += 1.8
            if k in t and (title and k in title):
                score += 0.7

        # 弱いヒューリスティック（よく出る語）
        if "迷" in t and ("迷" in title or "断つ" in title):
            score += 1.2
        if ("挑戦" in t or "チャレンジ" in t) and ("挑戦" in title or "踏み出" in title):
            score += 1.2
        if ("焦" in t or "待" in t) and ("待" in title or "静" in title):
            score += 1.0

        v[i] = score

    # 0ばかり回避：少しだけ分散ノイズ
    if float(np.max(v)) <= 1e-9:
        v += 0.2

    # 0..5へ正規化（見た目合わせ）
    v = 5.0 * (v / (np.max(v) + 1e-9))
    return v

# ============================================================
# 4) QUBO one-hot solve (SA)
# ============================================================

def build_qubo_onehot(linear: np.ndarray, P: float) -> np.ndarray:
    """
    minimize: sum_i linear[i]*x_i + P*(sum_i x_i - 1)^2
    """
    n = len(linear)
    Q = np.zeros((n, n), dtype=float)
    # linear
    for i in range(n):
        Q[i, i] += linear[i]
    # penalty: P*(sum x -1)^2 = P*(sum x + 2 sum_{i<j} x_i x_j -2 sum x +1)
    # = P*(-sum x + 2 sum_{i<j} x_i x_j) + const
    for i in range(n):
        Q[i, i] += -P
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2.0 * P
            Q[j, i] += 2.0 * P
    return Q

def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    return float(x @ Q @ x)

def sa_solve_qubo(Q: np.ndarray, n_sweeps: int = 400, temp0: float = 3.0, temp1: float = 0.05, rng=None) -> Tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng(0)
    n = Q.shape[0]
    # start random
    x = rng.integers(0, 2, size=n).astype(float)
    e = qubo_energy(Q, x)
    best_x = x.copy()
    best_e = e

    for s in range(max(10, int(n_sweeps))):
        t = temp0 * ((temp1 / temp0) ** (s / max(1, n_sweeps - 1)))
        # one sweep: try flipping each var once (random order)
        for i in rng.permutation(n):
            x2 = x.copy()
            x2[i] = 1.0 - x2[i]
            e2 = qubo_energy(Q, x2)
            de = e2 - e
            if de <= 0 or rng.random() < math.exp(-de / max(1e-9, t)):
                x, e = x2, e2
                if e < best_e:
                    best_e = e
                    best_x = x.copy()

    return best_x, best_e

def decode_onehot(x: np.ndarray) -> int:
    ones = np.where(x > 0.5)[0]
    if len(ones) == 0:
        return int(np.argmax(x))
    if len(ones) == 1:
        return int(ones[0])
    # multiple: choose strongest
    return int(ones[0])

# ============================================================
# 5) Character image resolver
# ============================================================

def resolve_char_image(char_id: str, image_dir: str, pack_df: Optional[pd.DataFrame] = None) -> Optional[str]:
    """
    いろんな命名に対応:
    - CHAR_01.png / char_01.png
    - CHAR_01.jpg / .webp
    - Excel側にIMG/FILE/PNG列があればそれも優先（ある場合のみ）
    """
    char_id = (char_id or "").strip()
    if not char_id:
        return None

    pdir = Path(image_dir)
    if not pdir.exists():
        return None

    # Excel内のファイル名列があれば先に見る（任意）
    if pack_df is not None:
        for col in ["IMG", "IMAGE", "FILE", "FILENAME", "PNG", "PATH"]:
            if col in pack_df.columns:
                row = pack_df[pack_df["CHAR_ID"].astype(str).str.strip() == char_id]
                if len(row) > 0:
                    fn = str(row.iloc[0].get(col, "")).strip()
                    if fn:
                        cand = pdir / fn
                        if cand.exists():
                            return str(cand)

    stems = [
        char_id,
        char_id.lower(),
        char_id.upper(),
        char_id.replace("-", "_"),
        char_id.replace("_", "-"),
    ]
    exts = [".png", ".jpg", ".jpeg", ".webp"]

    for s in stems:
        for e in exts:
            f = pdir / f"{s}{e}"
            if f.exists():
                return str(f)

    # fallback: prefix search (例: CHAR_p1.png など)
    # "CHAR_01" -> "CHAR" and "01" を含むファイルを探す
    m = re.search(r"(\d+)", char_id)
    num = m.group(1) if m else None
    files = list(pdir.glob("*"))
    for f in files:
        name = f.name.lower()
        if ("char" in name) and (num and num in name) and f.suffix.lower() in exts:
            return str(f)

    return None

# ============================================================
# 6) Word-sphere (簡易) for Step4
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

def calc_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    denom = max(1, len(sa | sb))
    return inter / denom

def build_word_graph(center: List[str], rng: np.random.Generator, n_total: int = 30) -> Dict:
    center = [c for c in center if c]
    if not center:
        center = ["静けさ"]

    # candidate words = center + db
    all_words = list(dict.fromkeys(center + GLOBAL_WORDS_DATABASE))
    # energy: close to center => lower
    energies = {}
    for w in all_words:
        sims = [calc_similarity(w, c) for c in center]
        sim = float(np.mean(sims)) if sims else 0.0
        e = -2.0 * sim + rng.normal(0, 0.08)
        if w in center:
            e -= 1.2
        energies[w] = e

    # pick lowest energies
    selected = [w for w, _ in sorted(energies.items(), key=lambda x: x[1])][:max(10, int(n_total))]
    sel_set = set(selected)
    # edges: connect pairs with sim high
    edges = []
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            s = calc_similarity(selected[i], selected[j])
            if s >= 0.28:
                edges.append((i, j, -s))
    return {"words": selected, "energies": {w: energies[w] for w in selected}, "edges": edges, "center_set": set(center) & sel_set}

def layout_word_graph(words: List[str], energies: Dict[str, float], center_set: set, rng: np.random.Generator) -> np.ndarray:
    n = len(words)
    pos = rng.normal(0, 1, size=(n, 3))
    # pull center close to origin, others by energy radius
    e_vals = [energies.get(w, 0.0) for w in words]
    mn, mx = float(np.min(e_vals)), float(np.max(e_vals))
    rng_e = (mx - mn) if mx != mn else 1.0
    for i, w in enumerate(words):
        e = energies.get(w, 0.0)
        norm = (e - mn) / rng_e
        r = 0.3 + (1.0 - norm) * 2.2
        if w in center_set:
            r = 0.25
        v = pos[i]
        v = v / (np.linalg.norm(v) + 1e-9) * r
        pos[i] = v
    return pos

def plot_word_sphere(graph: Dict, seed: int) -> go.Figure:
    rng = np.random.default_rng(seed)
    words = graph["words"]
    energies = graph["energies"]
    edges = graph["edges"]
    center_set = graph["center_set"]
    pos = layout_word_graph(words, energies, center_set, rng)

    fig = go.Figure()

    # stars (fixed)
    star_rng = np.random.default_rng(12345)
    star_n = 600
    sx = star_rng.uniform(-3.2, 3.2, star_n)
    sy = star_rng.uniform(-2.4, 2.4, star_n)
    sz = star_rng.uniform(-2.0, 2.0, star_n)
    fig.add_trace(go.Scatter3d(
        x=sx, y=sy, z=sz,
        mode="markers",
        marker=dict(size=star_rng.uniform(1.0, 2.2, star_n), color="rgba(255,255,255,0.20)"),
        hoverinfo="skip",
        showlegend=False
    ))

    # edges
    xE, yE, zE = [], [], []
    for i, j, _ in edges:
        x0, y0, z0 = pos[i]
        x1, y1, z1 = pos[j]
        xE += [x0, x1, None]
        yE += [y0, y1, None]
        zE += [z0, z1, None]
    fig.add_trace(go.Scatter3d(
        x=xE, y=yE, z=zE,
        mode="lines",
        line=dict(width=1, color="rgba(200,220,255,0.22)"),
        hoverinfo="skip",
        showlegend=False
    ))

    # nodes
    sizes, colors, labels = [], [], []
    for w in words:
        e = energies.get(w, 0.0)
        if w in center_set:
            sizes.append(26)
            colors.append("rgba(255,235,100,0.98)")
        else:
            sizes.append(12 + int(6 * min(1.0, abs(e) / 2.0)))
            colors.append("rgba(220,240,255,0.72)" if e < -0.5 else "rgba(255,255,255,0.58)")
        labels.append(w)

    # split center / others for text color
    ci = [i for i, w in enumerate(words) if w in center_set]
    oi = [i for i, w in enumerate(words) if w not in center_set]

    if oi:
        oi = np.array(oi, dtype=int)
        fig.add_trace(go.Scatter3d(
            x=pos[oi,0], y=pos[oi,1], z=pos[oi,2],
            mode="markers+text",
            text=[labels[i] for i in oi],
            textposition="top center",
            textfont=dict(size=14, color="rgba(245,245,255,0.95)"),
            marker=dict(size=[sizes[i] for i in oi], color=[colors[i] for i in oi], line=dict(width=1, color="rgba(0,0,0,0.12)")),
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False
        ))
    if ci:
        ci = np.array(ci, dtype=int)
        fig.add_trace(go.Scatter3d(
            x=pos[ci,0], y=pos[ci,1], z=pos[ci,2],
            mode="markers+text",
            text=[labels[i] for i in ci],
            textposition="top center",
            textfont=dict(size=18, color="rgba(255,80,80,1.0)"),
            marker=dict(size=[sizes[i] for i in ci], color=[colors[i] for i in ci], line=dict(width=2, color="rgba(255,80,80,0.80)")),
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
            camera=dict(eye=dict(x=1.55, y=1.15, z=1.05)),
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
    )
    return fig

# ============================================================
# 7) Sidebar (Excel / paths / params)
# ============================================================

def init_state():
    defaults = {
        "mix_alpha": 0.55,            # 1=slider寄り, 0=text寄り（UI文言に合わせるなら反転でもOK）
        "ngram_dummy": 3,
        "quote_temp": 1.20,
        "P_onehot": 40.0,
        "sa_sweeps": 420,
        "sample_n": 300,
        "stage_id": "ST_01",
        "use_stage_auto": True,
        "image_dir": "./assets/images/characters",
        "user_text": "新しい事にチャレンジする。",
        "manual_vow": {f"VOW_{i:02d}": 0.0 for i in range(1, VOW_N+1)},
        "last_run_sig": "",
        "last_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

with st.sidebar:
    st.markdown("## 📁 データ")

    uploaded = st.file_uploader("統合Excel（pack）", type=["xlsx"], key="pack_uploader")

    st.markdown("### 🖼 画像フォルダ（相対/絶対）")
    st.text_input("例: ./assets/images/characters", key="image_dir")

    st.markdown("---")
    st.markdown("## 🍁 季節×時間（Stage）")
    st.toggle("現在時刻から自動推定（簡易）", value=st.session_state["use_stage_auto"], key="use_stage_auto")
    st.text_input("STAGE_ID（手動上書き可）", key="stage_id")

    st.markdown("---")
    st.markdown("## 🎲 揺らぎ（観測のブレ）")
    st.slider("サンプル数（観測分布）", 50, 1000, int(st.session_state["sample_n"]), 10, key="sample_n")
    st.slider("SA sweeps（揺らぎ）", 80, 1200, int(st.session_state["sa_sweeps"]), 10, key="sa_sweeps")
    st.slider("SA温度（大揺らぎ）", 0.40, 2.50, float(st.session_state["quote_temp"]), 0.05, key="quote_temp")

    st.markdown("---")
    st.markdown("## ⚙️ QUBO設定（one-hot）")
    st.slider("one-hot ペナルティ P", 5.0, 120.0, float(st.session_state["P_onehot"]), 1.0, key="P_onehot")

    st.markdown("---")
    st.markdown("## 🧠 テキスト＋誓願（自動ベクトル化）")
    st.slider("mix比率 α（1=スライダー寄り / 0=テキスト寄り）", 0.0, 1.0, float(st.session_state["mix_alpha"]), 0.01, key="mix_alpha")
    st.number_input("n-gram（簡易）", min_value=1, max_value=6, value=int(st.session_state["ngram_dummy"]), step=1, key="ngram_dummy")

    st.markdown("---")
    if st.button("🔮 観測する（QUBOから抽出）", use_container_width=True):
        st.session_state["last_run_sig"] = ""  # force rerun compute

# ============================================================
# 8) Load Excel (or stop with friendly message)
# ============================================================

pack: Optional[PackData] = None
pack_err: Optional[str] = None

if uploaded is None:
    st.warning("左のサイドバーから **統合Excel（pack）** をアップロードしてください。")
    st.stop()

try:
    b = uploaded.getvalue()
    pack = load_pack_from_excel_bytes(b, file_hash=_hash_bytes(b))
    st.success(f"Excel読み込みOK（sheet: {pack.meta['sheet_char_to_vow']}）")
except Exception as e:
    st.error("Excel読み込みでエラーが発生しました。")
    st.code(str(e))
    st.stop()

df_ctv = pack.char_to_vow
df_vow = pack.vow_master
df_quotes = pack.quotes

vow_titles = build_vow_title_map(df_vow)
vow_ids = [f"VOW_{i:02d}" for i in range(1, VOW_N + 1)]
vow_title_list = [vow_titles.get(vid, vid) for vid in vow_ids]

# ============================================================
# 9) Main UI
# ============================================================

st.title("🔮 Q-Quest 量子神託（QUBO / STAGE×QUOTES）")

# ----------------------------
# Step1
# ----------------------------
left, right = st.columns([2.2, 1.0], gap="large")

with left:
    st.markdown("## Step 1：誓願入力（スライダー）＋テキスト（自動ベクトル化）")

    st.caption("あなたの状況を一文で（例：疲れていて決断ができない / 新しい挑戦が怖い など）")
    st.text_area("", height=90, key="user_text", placeholder="例：迷いを断ちたいが、今は焦らず機を待つべきか悩んでいる…")

    st.markdown("<div class='smallnote'>スライダー入力はTITLEを常時表示し、テキストからの自動推定と mix します。</div>", unsafe_allow_html=True)
    for i, vid in enumerate(vow_ids, start=0):
        ttl = vow_title_list[i]
        # 初期値が残るように key を固定
        k = f"manual_{vid}"
        if k not in st.session_state:
            st.session_state[k] = float(st.session_state["manual_vow"].get(vid, 0.0))
        st.slider(f"{vid}｜{ttl}", 0.0, 5.0, float(st.session_state[k]), 0.5, key=k)

with right:
    st.markdown("## Step 3：結果（観測された神＋理由＋QUOTES神託）")
    st.markdown("<div class='smallnote'>右側に「観測結果」「寄与した誓願」「格言（雰囲気）」を表示します。</div>", unsafe_allow_html=True)

# ============================================================
# 10) Compute vectors & QUBO
# ============================================================

def compute_once() -> Dict:
    text = st.session_state.get("user_text", "")
    alpha = float(st.session_state.get("mix_alpha", 0.55))
    P = float(st.session_state.get("P_onehot", 40.0))
    sweeps = int(st.session_state.get("sa_sweeps", 420))
    temp = float(st.session_state.get("quote_temp", 1.2))

    # manual vector
    v_manual = np.array([float(st.session_state.get(f"manual_{vid}", 0.0)) for vid in vow_ids], dtype=float)

    # auto vector
    v_auto = auto_vow_from_text(text, vow_titles)

    # mix (alpha=slider寄り)
    v_mix = alpha * v_manual + (1.0 - alpha) * v_auto

    # score each character: dot(char_to_vow, v_mix)
    W = df_ctv[vow_ids].to_numpy(dtype=float)  # shape (n_char, 12)
    scores = W @ v_mix                         # larger is better
    # for QUBO minimize, linear energy should be negative of score
    linear = -scores

    # build QUBO and solve
    Q = build_qubo_onehot(linear=linear, P=P)

    # seed depends on text + v_mix coarse + params
    sig = f"{text}|{alpha:.3f}|{P:.2f}|{sweeps}|{temp:.2f}|{np.round(v_mix,3).tolist()}"
    seed = make_seed(sig)
    rng = np.random.default_rng(seed)

    x_best, e_best = sa_solve_qubo(Q, n_sweeps=sweeps, temp0=3.0*temp, temp1=0.05, rng=rng)
    pick = decode_onehot(x_best)

    # probabilities (softmax over scores with temperature)
    probs = softmax(scores.astype(float), temp=max(0.2, temp))

    chosen_char = str(df_ctv.iloc[pick]["CHAR_ID"]).strip()

    # top3
    topk = np.argsort(-probs)[:3]
    df_top = pd.DataFrame({
        "順位": np.arange(1, len(topk)+1),
        "CHAR_ID": [str(df_ctv.iloc[i]["CHAR_ID"]).strip() for i in topk],
        "energy（低いほど選ばれやすい）": [float(linear[i]) for i in topk],
        "確率（softmax）": [float(probs[i]) for i in topk],
    })

    # contribution per vow (mix*vow_weight)
    vow_contrib = v_mix.copy()
    # show top
    contrib_df = pd.DataFrame({
        "VOW": vow_ids,
        "TITLE": vow_title_list,
        "mix(v)": np.round(v_mix, 6),
        "W(char,v)": np.round(W[pick, :], 6),
        "寄与(v*w)": np.round(v_mix * W[pick, :], 6),
    })
    contrib_df = contrib_df.sort_values("寄与(v*w)", ascending=False).head(6).reset_index(drop=True)

    # sample distribution (観測分布)
    sample_n = int(st.session_state.get("sample_n", 300))
    picks = []
    for s in range(sample_n):
        rrng = np.random.default_rng(seed + 1000 + s)
        xb, _ = sa_solve_qubo(Q, n_sweeps=max(120, sweeps//2), temp0=3.0*temp, temp1=0.07, rng=rrng)
        picks.append(decode_onehot(xb))
    picks = np.array(picks, dtype=int)
    counts = np.bincount(picks, minlength=len(df_ctv))
    # top 12 in distribution
    top_dist = np.argsort(-counts)[:min(12, len(counts))]
    dist_df = pd.DataFrame({
        "CHAR_ID": [str(df_ctv.iloc[i]["CHAR_ID"]).strip() for i in top_dist],
        "count": [int(counts[i]) for i in top_dist],
    })

    return {
        "sig": sig,
        "seed": seed,
        "v_manual": v_manual,
        "v_auto": v_auto,
        "v_mix": v_mix,
        "scores": scores,
        "probs": probs,
        "chosen_index": int(pick),
        "chosen_char": chosen_char,
        "df_top": df_top,
        "contrib_df": contrib_df,
        "dist_df": dist_df,
        "Q": Q,
        "x_best": x_best,
        "e_best": e_best,
        "counts": counts,
    }

run_sig = f"{st.session_state.get('user_text','')}|{st.session_state.get('mix_alpha',0.55)}|{st.session_state.get('P_onehot',40.0)}|{st.session_state.get('sa_sweeps',420)}|{st.session_state.get('quote_temp',1.2)}|" \
          f"{[st.session_state.get(f'manual_{vid}',0.0) for vid in vow_ids]}"

if st.session_state.get("last_run_sig", "") != run_sig:
    st.session_state["last_result"] = compute_once()
    st.session_state["last_run_sig"] = run_sig

res = st.session_state["last_result"]

# ============================================================
# 11) Step2: graphs (dark background)
# ============================================================

st.markdown("---")
st.markdown("## 2) エネルギー地形（全候補）")

# energy bar plot (use plotly)
energies = -res["scores"]  # because linear = -score; low is better
order = np.argsort(energies)[:min(12, len(energies))]
labels = [str(df_ctv.iloc[i]["CHAR_ID"]).strip() for i in order]
vals = [float(energies[i]) for i in order]

fig_energy = go.Figure(go.Bar(x=labels, y=vals))
fig_energy.update_layout(
    paper_bgcolor="rgba(6,8,18,1)",
    plot_bgcolor="rgba(6,8,18,1)",
    font=dict(color="rgba(245,245,255,0.92)"),
    margin=dict(l=10, r=10, t=30, b=40),
    height=360,
    title="energy（低いほど選ばれやすい）上位",
)
fig_energy.update_xaxes(tickangle=-90, gridcolor="rgba(255,255,255,0.08)")
fig_energy.update_yaxes(gridcolor="rgba(255,255,255,0.10)")
st.plotly_chart(fig_energy, use_container_width=True)

# ============================================================
# 12) Step3: right panel (image + tables + quotes)
# ============================================================

def pick_quotes_for_char(char_id: str, n: int = 3) -> pd.DataFrame:
    if df_quotes is None or df_quotes.empty:
        return pd.DataFrame([
            {"QUOTE_ID": "Q_0001", "QUOTE": "未来は私たちの選択にかかっている。", "SOURCE": "—"},
            {"QUOTE_ID": "Q_0002", "QUOTE": "挑戦は小さな一歩から始まる。", "SOURCE": "—"},
            {"QUOTE_ID": "Q_0003", "QUOTE": "焦らず、機が熟すのを待て。", "SOURCE": "—"},
        ])
    # column guess
    qcol = None
    scol = None
    idcol = None
    for c in df_quotes.columns:
        cu = c.upper()
        if cu in ["QUOTE","格言","テキスト","文","言葉"]:
            qcol = c
        if cu in ["SOURCE","出典","作者","典拠","出所"]:
            scol = c
        if cu in ["QUOTE_ID","ID"]:
            idcol = c
    if qcol is None:
        qcol = df_quotes.columns[0]
    if scol is None:
        scol = df_quotes.columns[1] if len(df_quotes.columns) > 1 else qcol
    if idcol is None:
        idcol = df_quotes.columns[0]

    # score by keyword overlap (very simple)
    text = st.session_state.get("user_text","")
    kws = set(extract_keywords_simple(text, max_n=10))
    def score_row(r):
        q = str(r.get(qcol,""))
        s = 0
        for k in kws:
            if k and k in q:
                s += 2
        return s

    tmp = df_quotes.copy()
    tmp["_score"] = tmp.apply(score_row, axis=1)
    tmp = tmp.sort_values(["_score"], ascending=False).head(max(n, 3)).reset_index(drop=True)

    out = pd.DataFrame({
        "QUOTE_ID": tmp[idcol].astype(str) if idcol in tmp.columns else [f"Q_{i:04d}" for i in range(len(tmp))],
        "QUOTE": tmp[qcol].astype(str),
        "SOURCE": tmp[scol].astype(str) if scol in tmp.columns else "—",
    })
    return out.head(n)

with right:
    # top table
    st.dataframe(res["df_top"], use_container_width=True, hide_index=True)

    chosen_char = res["chosen_char"]
    st.markdown(f"### 🌟 今回“観測”された神：**{chosen_char}**")
    st.markdown("<div class='smallnote'>ここは“単発の観測（1回抽選）”です。下の観測分布（サンプル）は同条件で何回も観測したヒストグラムです。</div>", unsafe_allow_html=True)

    # character image
    img_path = resolve_char_image(chosen_char, st.session_state.get("image_dir","./assets/images/characters"), pack_df=df_ctv)
    if img_path and os.path.exists(img_path):
        st.image(img_path, use_container_width=True, caption=f"{chosen_char}（{Path(img_path).name}）")
    else:
        st.info("※キャラクター画像が見つかりません（画像フォルダパス/ファイル名をご確認ください）")

    # green vibe advice box (継承)
    # ここは「雰囲気の理由文」を生成（寄与上位VOWで文章化）
    top_vows = res["contrib_df"]["TITLE"].tolist()
    stage_txt = f"季節×時間（Stage）は流れを強める。"
    reason = "いまの波は " + "・".join([str(x) for x in top_vows[:4]]) + " に寄っている。 " + stage_txt
    st.markdown(f"<div class='quote-green'><b>いまの波</b>：{reason}</div>", unsafe_allow_html=True)

    st.markdown("### 🧩 寄与した誓願（Top）")
    st.dataframe(res["contrib_df"], use_container_width=True, hide_index=True)

    # blue vibe QUOTES
    st.markdown("### 🗣 QUOTES神託（温度付きで選択）")
    qdf = pick_quotes_for_char(chosen_char, n=3)
    for i in range(len(qdf)):
        qt = str(qdf.iloc[i]["QUOTE"])
        src = str(qdf.iloc[i].get("SOURCE","—"))
        st.markdown(
            f"<div class='quote-blue'><b>神託{i+1}</b><br>「{qt}」<br><span class='smallnote'>— {src}</span></div>",
            unsafe_allow_html=True
        )
        st.write("")

    with st.expander("🔎 格言候補Top（デバッグ）", expanded=False):
        st.dataframe(qdf, use_container_width=True, hide_index=True)

# ============================================================
# 13) Step4: keyword + word sphere (必ず表示)
# ============================================================

st.markdown("---")
st.markdown("## 4) テキストのキーワード抽出（簡易）")

kws = extract_keywords_simple(st.session_state.get("user_text",""), max_n=10)
if not kws:
    st.info("入力テキストが短い/抽出できないため、キーワードが空になりました。もう少し具体的に書くと安定します。")
else:
    kcol1, kcol2 = st.columns([1.0, 1.6], gap="large")
    with kcol1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 抽出キーワード")
        st.write(", ".join(kws[:8]))
        st.markdown("<div class='smallnote'>※ここを起点に、エネルギーが近い単語を空間に配置し、縁（線）で結びます。</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with kcol2:
        seed = res["seed"] + 777
        graph = build_word_graph(center=kws[:3], rng=np.random.default_rng(seed), n_total=32)
        fig_ws = plot_word_sphere(graph, seed=seed)
        fig_ws.update_layout(title="🌐 単語の球体（誓願→キーワード→縁のネットワーク）")
        st.plotly_chart(fig_ws, use_container_width=True, config={"displaylogo": False})

# ============================================================
# 14) Extra: show vow vectors (manual/auto/mix) (optional)
# ============================================================

with st.expander("📈 誓願ベクトル（manual / auto / mix）", expanded=False):
    df_vec = pd.DataFrame({
        "VOW_ID": vow_ids,
        "TITLE": vow_title_list,
        "manual(0-5)": np.round(res["v_manual"], 4),
        "auto(0-5)": np.round(res["v_auto"], 4),
        "mix(0-5)": np.round(res["v_mix"], 4),
    })
    st.dataframe(df_vec, use_container_width=True, hide_index=True)

with st.expander("🧠 QUBO証拠（one-hot）", expanded=False):
    x = res["x_best"]
    st.write(f"P = {float(st.session_state.get('P_onehot',40.0)):.2f}")
    st.write(f"x（選択ベクトル） = {x.astype(int).tolist()} / sum={int(np.sum(x>0.5))}")
    st.write(f"E(x) = {res['e_best']:.6f}")
    st.latex(r"E(\mathbf{x})=\sum_i Q_{ii}x_i + \sum_{i<j} Q_{ij}x_i x_j + P(\sum_i x_i - 1)^2")
