# -*- coding: utf-8 -*-
# ============================================================
# Q-Quest 量子神託（QUBO / STAGE×QUOTES）
# - ダークUI完全対応（入力/アップロード/表/グラフ）
# - Excel列名検出を緩く（VOW_01... が多少崩れても吸収）
# - Step3はExcel不備でも落とさず表示（fallback）
# - 4) テキストキーワード抽出エリアに「縁の球体（単語ネットワーク）」を表示
# - 12神キャラクター(ギャラリー)は削除
# ============================================================

import os
import re
import io
import zlib
import math
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# pandas（Excel読み込み / 表のStyler）
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    pd = None
    PANDAS_AVAILABLE = False


# ------------------------------------------------------------
# ★ 必ず最初のStreamlitコマンド
# ------------------------------------------------------------
st.set_page_config(page_title="Q-Quest 量子神託（QUBO / STAGE×QUOTES）", layout="wide")


# ============================================================
# 0) CSS（全面ダーク化：file_uploader / text_area / selectbox / dataframe header など）
# ============================================================
DARK_CSS = """
<style>
/* --- App background --- */
.stApp{
  background:
    radial-gradient(circle at 18% 24%, rgba(110,150,255,0.12), transparent 38%),
    radial-gradient(circle at 78% 68%, rgba(255,160,220,0.08), transparent 44%),
    radial-gradient(circle at 50% 50%, rgba(255,255,255,0.03), transparent 55%),
    linear-gradient(180deg, rgba(6,8,18,1), rgba(10,12,26,1));
}

/* --- Base text --- */
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
  color: rgba(245,245,255,0.96);
}

/* --- Sidebar --- */
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.06);
  border-right: 1px solid rgba(255,255,255,0.10);
  backdrop-filter: blur(10px);
}
section[data-testid="stSidebar"] *{
  color: rgba(245,245,255,0.92);
}

/* --- Inputs: text_area / text_input / selectbox --- */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] div[role="combobox"],
div[data-testid="stNumberInput"] input{
  background: rgba(10,14,30,0.85) !important;
  color: rgba(245,245,255,0.95) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 12px !important;
}
div[data-testid="stTextArea"] textarea::placeholder,
div[data-testid="stTextInput"] input::placeholder{
  color: rgba(220,230,255,0.45) !important;
}

/* --- File uploader: dropzone + button --- */
div[data-testid="stFileUploader"] section{
  background: rgba(10,14,30,0.70) !important;
  border: 1px dashed rgba(255,255,255,0.22) !important;
  border-radius: 14px !important;
}
div[data-testid="stFileUploader"] *{
  color: rgba(245,245,255,0.92) !important;
}
div[data-testid="stFileUploader"] button{
  background: rgba(255,255,255,0.10) !important;
  color: rgba(245,245,255,0.95) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 12px !important;
}

/* --- Sliders --- */
div[data-testid="stSlider"] div[role="slider"]{
  background: rgba(255,255,255,0.10) !important;
}

/* --- Buttons --- */
.stButton>button{
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  color: rgba(245,245,255,0.95) !important;
  background: linear-gradient(90deg, rgba(120,110,255,0.65), rgba(170,120,255,0.55)) !important;
}
.stButton>button:hover{
  filter: brightness(1.05);
}

/* --- Expanders / containers --- */
div[data-testid="stExpander"]{
  border-radius: 16px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  background: rgba(255,255,255,0.04) !important;
}

/* --- Plotly wrapper --- */
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

/* --- Dataframe/Table : make surrounding dark (base) --- */
div[data-testid="stDataFrame"]{
  background: rgba(8,12,26,0.85) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 16px !important;
  padding: 6px !important;
}

/* --- Success/Info/Warning boxes to match atmosphere --- */
div[data-testid="stAlert"]{
  border-radius: 14px !important;
}

/* --- Quote boxes (継承) --- */
.quote-green{
  background: rgba(40,120,60,0.34);
  border: 1px solid rgba(90,255,150,0.26);
  border-radius: 14px;
  padding: 12px 14px;
  color: rgba(220,255,235,0.95);
}
.quote-green b{ color: rgba(210,255,230,1.0); }

.quote-blue{
  background: rgba(30,80,150,0.36);
  border: 1px solid rgba(130,190,255,0.26);
  border-radius: 14px;
  padding: 12px 14px;
  color: rgba(225,245,255,0.95);
}
.quote-blue b{ color: rgba(235,250,255,1.0); }

/* --- Card --- */
.card{
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 16px 12px 16px;
  box-shadow: 0 18px 60px rgba(0,0,0,0.18);
}

/* --- Small note --- */
.smallnote{opacity:0.80; font-size:0.92rem;}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ============================================================
# 1) Utility
# ============================================================
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _adler_seed(s: str) -> int:
    return int(zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF)

def _norm_colname(s: str) -> str:
    """列名を緩く正規化（全角/空白/記号/大小を吸収）"""
    if s is None:
        return ""
    s = str(s)
    s = s.strip()
    # 全角アンダーラインなども吸収
    s = s.replace("＿", "_")
    s = s.replace("－", "-").replace("ー", "-")
    s = s.replace("　", " ")
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    return s

def _detect_vow_columns(columns: List[str], vow_n: int = 12) -> Dict[int, str]:
    """
    2) 「重み列（VOW_01...）」検出が厳しすぎる問題を解決：
    - VOW_01 / VOW01 / vow-1 / VOW 1 / ＶＯＷ＿０１ などを吸収して検出
    """
    colmap: Dict[int, str] = {}
    for c in columns:
        raw = str(c)
        nc = _norm_colname(raw)

        # 例: vow_01 / vow01 / vow-1 / vow1 / vow０１
        # 全角数字 → 半角へ（ざっくり）
        nc = nc.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

        m = re.match(r"^vow[_\-]*0*([1-9]|1[0-2])$", nc)
        if not m:
            # 例: vow01_weight のようなケースも少し許容
            m = re.match(r"^vow[_\-]*0*([1-9]|1[0-2]).*$", nc)

        if m:
            idx = int(m.group(1))
            if 1 <= idx <= vow_n and idx not in colmap:
                colmap[idx] = raw
    return colmap

def style_df_dark(df: "pd.DataFrame") -> "pd.io.formats.style.Styler":
    """
    表のヘッダー白問題の決定打：
    - st.dataframe(df.style...) でヘッダー(th)も含め黒基調に統一
    """
    if not PANDAS_AVAILABLE:
        return df  # type: ignore

    base_bg = "rgba(8,12,26,0.92)"
    head_bg = "rgba(12,18,38,0.98)"
    grid = "rgba(255,255,255,0.10)"
    txt = "rgba(245,245,255,0.92)"

    sty = df.style
    sty = sty.set_table_styles([
        {"selector": "table", "props": [("background-color", base_bg), ("border-collapse", "collapse"), ("width", "100%")]},
        {"selector": "th", "props": [("background-color", head_bg), ("color", txt), ("border", f"1px solid {grid}"), ("font-weight", "700")]},
        {"selector": "td", "props": [("background-color", base_bg), ("color", txt), ("border", f"1px solid {grid}")]},
        {"selector": "tr:hover td", "props": [("background-color", "rgba(255,255,255,0.06)")]},
    ])
    sty = sty.set_properties(**{
        "font-family": '"Hiragino Mincho ProN","Yu Mincho","Noto Serif JP",serif',
        "font-size": "13px",
    })
    return sty


# ============================================================
# 2) Excel Loader（ゆるく読む / ダメでも落とさない）
# ============================================================
EXCEL_DEFAULT_CANDIDATES = [
    "quantum_shintaku_pack.xlsx",
    "quantum_shintaku_pack_v3.xlsx",
    "quantum_shintaku_pack_v3_with_sense_20260213_oposite_modify.xlsx",
    "quantum_shintaku_pack_v3_with_sense_20260213_oposite_modify (2).xlsx",
    "quantum_shintaku_pack_v3_with_sense_20260213_oposite_modify_with_lr022101.xlsx",
]

REQUIRED_SHEETS_HINT = ["CHAR_TO_VOW", "VOW_MASTER", "QUOTES", "STAGE_MASTER"]

@st.cache_data(show_spinner=False)
def load_excel_pack(excel_bytes: bytes, file_hash: str) -> Dict:
    """
    できるだけ頑強に読みます：
    - シート名が多少違っても候補探索
    - VOW列名が崩れてても検出
    - 読めない部分は fallback
    """
    out = {
        "ok": False,
        "errors": [],
        "sheet_names": [],
        "char_to_vow": None,   # pd.DataFrame
        "vow_master": None,    # pd.DataFrame
        "quotes": None,        # pd.DataFrame
        "stage_master": None,  # pd.DataFrame
    }

    if not PANDAS_AVAILABLE:
        out["errors"].append("pandas が利用できないためExcel読み込みが無効です。requirements.txt に pandas と openpyxl を入れてください。")
        return out

    bio = io.BytesIO(excel_bytes)
    try:
        xls = pd.ExcelFile(bio, engine="openpyxl")
        out["sheet_names"] = list(xls.sheet_names)
    except Exception as e:
        out["errors"].append(f"Excelの読み込みに失敗: {e}")
        return out

    # シート名候補（ゆるく）
    def pick_sheet(candidates: List[str]) -> Optional[str]:
        names = out["sheet_names"]
        norm_names = {_norm_colname(n): n for n in names}
        for c in candidates:
            key = _norm_colname(c)
            if key in norm_names:
                return norm_names[key]
        # 部分一致（最後の保険）
        for n in names:
            nn = _norm_colname(n)
            for c in candidates:
                if _norm_colname(c) in nn:
                    return n
        return None

    s_char_to_vow = pick_sheet(["CHAR_TO_VOW", "CHAR2VOW", "CHAR_TO_VOWS"])
    s_vow_master  = pick_sheet(["VOW_MASTER", "VOWS", "VOW"])
    s_quotes      = pick_sheet(["QUOTES", "QUOTE", "KAKUGEN"])
    s_stage       = pick_sheet(["STAGE_MASTER", "STAGE", "SEASON", "JIKAN"])

    # 読む（失敗しても落とさない）
    def safe_read(sheet: Optional[str]) -> Optional["pd.DataFrame"]:
        if not sheet:
            return None
        try:
            return pd.read_excel(bio, sheet_name=sheet, engine="openpyxl")
        except Exception as e:
            out["errors"].append(f"シート '{sheet}' 読み込み失敗: {e}")
            return None

    # BytesIO は読み進むので読み直す
    def read_sheet(sheet: Optional[str]) -> Optional["pd.DataFrame"]:
        if not sheet:
            return None
        try:
            return pd.read_excel(io.BytesIO(excel_bytes), sheet_name=sheet, engine="openpyxl")
        except Exception as e:
            out["errors"].append(f"シート '{sheet}' 読み込み失敗: {e}")
            return None

    out["char_to_vow"] = read_sheet(s_char_to_vow)
    out["vow_master"]  = read_sheet(s_vow_master)
    out["quotes"]      = read_sheet(s_quotes)
    out["stage_master"]= read_sheet(s_stage)

    # 重要：CHAR_TO_VOW の VOW列検出を「厳密一致」しない
    if out["char_to_vow"] is not None:
        df = out["char_to_vow"]
        vmap = _detect_vow_columns(list(df.columns), vow_n=12)
        if len(vmap) < 8:
            out["errors"].append(
                "VOW/CHAR/重み列の検出が十分できませんでした（VOW_01..12相当の列が少ない）。"
                " 列名が 'VOW_01' 形式に近いか確認してください（例: VOW01 / vow-1 などは吸収します）。"
            )
        # 標準化した列を作る（足りない分は0）
        for i in range(1, 13):
            src = vmap.get(i)
            if src is None:
                df[f"VOW_{i:02d}"] = 0.0
            else:
                df[f"VOW_{i:02d}"] = pd.to_numeric(df[src], errors="coerce").fillna(0.0)

        # char_id 列もゆるく探す
        cols_norm = {_norm_colname(c): c for c in df.columns}
        char_id_col = None
        for key in ["char_id", "charid", "id", "chr_id", "character_id"]:
            if key in cols_norm:
                char_id_col = cols_norm[key]
                break
        if char_id_col is None:
            # 先頭列をID扱い（最後の保険）
            char_id_col = df.columns[0]
            out["errors"].append(f"CHAR_ID列が見つからないため、先頭列 '{char_id_col}' をIDとして扱います。")

        df["CHAR_ID__STD"] = df[char_id_col].astype(str)
        out["char_to_vow"] = df

    out["ok"] = True
    return out


# ============================================================
# 3) 基本データ（fallback）
# ============================================================
DEFAULT_VOW_TITLES = {
    "VOW_01": "迷いを断つ",
    "VOW_02": "静けさを保ち、焦らず待つ",
    "VOW_03": "内面の洞察を深める",
    "VOW_04": "行動に踏み出す",
    "VOW_05": "つながりを育む",
    "VOW_06": "挑戦を選ぶ",
    "VOW_07": "焦りと迷いを手放す",
    "VOW_08": "小さく動く",
    "VOW_09": "つながりと挑戦",
    "VOW_10": "判断を急がず、機を待つ",
    "VOW_11": "静けさとつながり",
    "VOW_12": "行動と挑戦を加速",
}

DEFAULT_QUOTES = [
    {"quote":"途中であきらめてはいけない。途中であきらめてしまったら、得るものより失うものの方が、ずっと多くなってしまう。", "source":"ルイ・アームストロング（ミュージシャン）"},
    {"quote":"人生の最良の瞬間は、今この瞬間だ。", "source":"ヒンディーのことわざ"},
    {"quote":"過去は過去であり、未来は未来である。今この瞬間こそがプレゼントだ。", "source":"エリオット・D・アディ"},
    {"quote":"成功は、自分の強みを活かすことから始まる。", "source":"ピーター・ドラッカー"},
    {"quote":"挫折は成功への階段である。", "source":"ウィンストン・チャーチル"},
]


# ============================================================
# 4) テキスト→キーワード抽出（簡易）
# ============================================================
STOP_TOKENS = set([
    "した","たい","いる","こと","それ","これ","ため","よう","ので","から",
    "です","ます","ある","ない","そして","でも","しかし","また",
    "自分","私","あなた","もの","感じ","気持ち","今日",
    "に","を","が","は","と","も","で","へ","や","の"
])

def extract_keywords_simple(text: str, top_n: int = 5) -> List[str]:
    text = (text or "").strip()
    if not text:
        return ["静けさ","迷い"]

    text_clean = re.sub(r"[0-9０-９、。．,.!！?？\(\)\[\]{}「」『』\"'：:;／/\\\n\r\t]+", " ", text)
    tokens = [t.strip() for t in re.split(r"\s+", text_clean) if t.strip()]
    tokens = [t for t in tokens if (len(t) >= 2 and t not in STOP_TOKENS)]
    if not tokens:
        return ["静けさ","迷い"]
    tokens = sorted(tokens, key=lambda s: (-len(s), s))
    return tokens[:top_n]


# ============================================================
# 5) 単語ネットワーク（縁の球体）— 提示されたコードを整理して埋め込み
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
    "行動": ["努力","継続","忍耐","誠実","正直","挑戦"],
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

def calculate_energy_between_words(word1: str, word2: str, rng: np.random.Generator, jitter: float) -> float:
    similarity = calculate_semantic_similarity(word1, word2)
    energy = -2.0 * similarity + 0.5

    common = set(word1) & set(word2)
    if common:
        energy -= 0.20 * len(common) / max(len(word1), len(word2), 1)

    for _, ws in CATEGORIES.items():
        if (word1 in ws) and (word2 in ws):
            energy -= 0.60
            break

    if jitter > 0:
        energy += rng.normal(0, jitter)
    return float(energy)

def build_qubo_matrix_for_words(words: List[str], rng: np.random.Generator, jitter: float) -> np.ndarray:
    n = len(words)
    Q = np.zeros((n, n), dtype=float)
    np.fill_diagonal(Q, -0.5)
    for i in range(n):
        for j in range(i + 1, n):
            e = calculate_energy_between_words(words[i], words[j], rng, jitter)
            Q[i, j] = e
            Q[j, i] = e
    return Q

def solve_qubo_placement(
    Q: np.ndarray,
    words: List[str],
    center_indices: List[int],
    energies: Dict[str, float],
    rng: np.random.Generator,
    n_iterations: int = 100,
) -> np.ndarray:
    n = len(words)
    pos = np.zeros((n, 3), dtype=float)
    for idx in center_indices:
        if idx < n:
            pos[idx] = [0.0, 0.0, 0.0]

    ev = list(energies.values()) if energies else []
    if ev:
        mn, mx = min(ev), max(ev)
        er = (mx - mn) if mx != mn else 1.0
    else:
        mn, er = -3.0, 3.0

    golden_angle = np.pi * (3 - np.sqrt(5))
    k = 0

    # 初期配置
    for i in range(n):
        if i in center_indices:
            continue
        w = words[i]
        e = energies.get(w, 0.0)
        norm = (e - mn) / er if er > 0 else 0.5
        dist = 0.3 + (1.0 - norm) * 2.2

        theta = golden_angle * k
        y = 1 - (k / float(max(1, n - len(center_indices) - 1))) * 2
        r = np.sqrt(max(0.0, 1 - y * y))
        x = np.cos(theta) * r * dist
        z = np.sin(theta) * r * dist
        pos[i] = [x, y * dist * 0.6, z]
        k += 1

    # 疑似最適化
    for _ in range(n_iterations):
        for i in range(n):
            if i in center_indices:
                continue
            force = np.zeros(3, dtype=float)

            # 中心との距離制御
            for cidx in center_indices:
                vec = pos[cidx] - pos[i]
                d = np.linalg.norm(vec)
                if d > 0.01:
                    w = words[i]
                    e = energies.get(w, 0.0)
                    norm = (e - mn) / er if er > 0 else 0.5
                    target = 0.3 + (1.0 - norm) * 2.2

                    if d < target * 0.9:
                        force -= vec / d * 0.05
                    elif d > target * 1.1:
                        force += vec / d * 0.10

            # 単語間相互作用
            for j in range(n):
                if i == j or j in center_indices:
                    continue
                eij = Q[i, j]
                if eij < -0.3:  # 引力
                    vec = pos[j] - pos[i]
                    d = np.linalg.norm(vec)
                    if d > 0.01:
                        force += vec / d * (abs(eij) * 0.08)
                elif eij > 0.2:  # 斥力
                    vec = pos[i] - pos[j]
                    d = np.linalg.norm(vec)
                    if d > 0.01:
                        force += vec / d * (abs(eij) * 0.03)

            pos[i] += force * 0.15

    return pos

def build_word_network(center_words: List[str], database: List[str], n_total: int,
                       rng: np.random.Generator, jitter: float) -> Dict:
    all_words = list(dict.fromkeys(center_words + database))
    energies: Dict[str, float] = {}

    for w in all_words:
        if w in center_words:
            energies[w] = -3.0
        else:
            e_list = [calculate_energy_between_words(c, w, rng, jitter) for c in center_words]
            energies[w] = float(np.mean(e_list))

    sorted_words = sorted(energies.items(), key=lambda x: x[1])
    selected: List[str] = []
    for w, _ in sorted_words:
        if w in center_words and w not in selected:
            selected.append(w)
    for w, _ in sorted_words:
        if w not in selected:
            selected.append(w)
        if len(selected) >= n_total:
            break

    Q = build_qubo_matrix_for_words(selected, rng, jitter)
    center_indices = [i for i, w in enumerate(selected) if w in center_words]

    edges: List[Tuple[int, int, float]] = []
    n = len(selected)
    for i in range(n):
        for j in range(i + 1, n):
            e = Q[i, j]
            if e < -0.25:
                edges.append((i, j, float(e)))

    return {
        "words": selected,
        "energies": {w: energies[w] for w in selected},
        "edges": edges,
        "Q": Q,
        "center_indices": center_indices,
    }

def render_word_sphere(user_text: str, seed_key: str, height: int = 520):
    """
    4) テキストキーワード抽出エリアに表示する「縁の球体」。
    - 点滅しない（固定seed）
    - 背景黒
    """
    kw = extract_keywords_simple(user_text, top_n=5)
    seed = _adler_seed(seed_key + "|" + user_text + "|" + ",".join(kw))
    rng = np.random.default_rng(seed)

    network = build_word_network(kw, GLOBAL_WORDS_DATABASE, n_total=34, rng=rng, jitter=0.10)
    pos = solve_qubo_placement(network["Q"], network["words"], network["center_indices"], network["energies"], rng=rng, n_iterations=80)

    words = network["words"]
    energies = network["energies"]
    edges = network["edges"]
    center_indices = network["center_indices"]
    center_set = set([words[i] for i in center_indices])

    fig = go.Figure()

    # 星屑固定
    star_rng = np.random.default_rng(12345)
    star_count = 900
    sx = star_rng.uniform(-3.2, 3.2, star_count)
    sy = star_rng.uniform(-2.4, 2.4, star_count)
    sz = star_rng.uniform(-2.0, 2.0, star_count)
    alpha = np.full(star_count, 0.22, dtype=float)
    star_size = star_rng.uniform(1.0, 2.4, star_count)
    star_colors = [f"rgba(255,255,255,{a})" for a in alpha]

    fig.add_trace(go.Scatter3d(
        x=sx, y=sy, z=sz,
        mode="markers",
        marker=dict(size=star_size, color=star_colors),
        hoverinfo="skip",
        showlegend=False
    ))

    # エッジ
    xE, yE, zE = [], [], []
    for i, j, e in edges:
        x0, y0, z0 = pos[i]
        x1, y1, z1 = pos[j]
        xE += [x0, x1, None]
        yE += [y0, y1, None]
        zE += [z0, z1, None]

    fig.add_trace(go.Scatter3d(
        x=xE, y=yE, z=zE,
        mode="lines",
        line=dict(width=1, color="rgba(200,220,255,0.20)"),
        hoverinfo="skip",
        showlegend=False
    ))

    # ノード（中心語は赤文字）
    sizes, colors, labels = [], [], []
    for w in words:
        e = energies.get(w, 0.0)
        if w in center_set:
            sizes.append(28)
            colors.append("rgba(255,235,100,0.98)")
            labels.append(w)
        else:
            en = min(1.0, abs(e) / 3.0)
            sizes.append(12 + int(8 * en))
            if e < -1.5:
                colors.append("rgba(180,220,255,0.85)")
            elif e < -0.5:
                colors.append("rgba(220,240,255,0.75)")
            else:
                colors.append("rgba(255,255,255,0.58)")
            labels.append(w)

    center_idx = [i for i, w in enumerate(labels) if w in center_set]
    other_idx  = [i for i, w in enumerate(labels) if w not in center_set]

    if other_idx:
        oi = np.array(other_idx, dtype=int)
        fig.add_trace(go.Scatter3d(
            x=pos[oi, 0], y=pos[oi, 1], z=pos[oi, 2],
            mode="markers+text",
            text=[labels[i] for i in oi],
            textposition="top center",
            textfont=dict(size=16, color="rgba(245,245,255,0.95)"),
            marker=dict(size=[sizes[i] for i in oi], color=[colors[i] for i in oi],
                        line=dict(width=1, color="rgba(0,0,0,0.10)")),
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False
        ))

    if center_idx:
        ci = np.array(center_idx, dtype=int)
        fig.add_trace(go.Scatter3d(
            x=pos[ci, 0], y=pos[ci, 1], z=pos[ci, 2],
            mode="markers+text",
            text=[labels[i] for i in ci],
            textposition="top center",
            textfont=dict(size=22, color="rgba(255,80,80,1.0)"),
            marker=dict(size=[sizes[i] for i in ci], color=[colors[i] for i in ci],
                        line=dict(width=2, color="rgba(255,80,80,0.8)")),
            hovertemplate="<b>%{text}</b><br>中心語<extra></extra>",
            showlegend=False
        ))

    fig.update_layout(
        height=height,
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
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})


# ============================================================
# 6) QUBO（one-hot）— “観測された神” を決める
#   ※ここは「動く」こと重視で、実装の骨格を頑強に
# ============================================================
def one_hot_qubo(Q_diag: np.ndarray, P: float) -> Tuple[np.ndarray, float]:
    """
    E(x)=sum_i Qii*xi + P*(sum_i xi -1)^2
    解は全探索（12キャラ程度なら十分軽い）
    """
    n = len(Q_diag)
    best_x = None
    best_e = float("inf")
    for k in range(n):
        x = np.zeros(n, dtype=int)
        x[k] = 1
        e = float(np.dot(Q_diag, x) + P * (np.sum(x) - 1) ** 2)
        if e < best_e:
            best_e = e
            best_x = x
    return best_x, best_e


# ============================================================
# 7) UI（Sidebar）
# ============================================================
st.title("🔮 Q-Quest 量子神託（QUBO / STAGE×QUOTES）")

if not PANDAS_AVAILABLE:
    st.warning("pandas が無いため Excel読み込みと表の完全ダーク化（ヘッダー黒）が制限されます。requirements.txt に pandas と openpyxl を追加してください。")

with st.sidebar:
    st.markdown("## 📁 データ")
    up = st.file_uploader("統合Excel（pack）", type=["xlsx"], key="pack_uploader")
    excel_bytes: Optional[bytes] = None
    source_label = "未指定（fallback）"

    if up is not None:
        excel_bytes = up.getvalue()
        source_label = "アップロード"
    else:
        # 同梱候補を探す
        for cand in EXCEL_DEFAULT_CANDIDATES:
            if os.path.exists(cand):
                try:
                    excel_bytes = Path(cand).read_bytes()
                    source_label = f"同梱: {cand}"
                    break
                except Exception:
                    pass

    st.caption(f"読み込み元: {source_label}")

    st.markdown("---")
    st.markdown("## ⚙️ QUBO設定（one-hot）")
    P = st.slider("one-hot ペナルティ P", 1.0, 120.0, 40.0, 1.0)
    n_samples = st.slider("サンプル数（観測分布）", 50, 800, 300, 10)
    sa_sweeps = st.slider("SA sweeps（揺らぎ）", 50, 900, 420, 10)
    sa_temp = st.slider("SA温度（大揺らぎ）", 0.2, 3.0, 1.2, 0.1)

    st.markdown("---")
    st.markdown("## 🧠 テキスト＋誓願（自動ベクトル化）")
    ngram_n = st.slider("n-gram（簡易）", 1, 6, 3, 1)
    alpha = st.slider("mix比率 α（1=スライダー寄り / 0=テキスト寄り）", 0.0, 1.0, 0.55, 0.01)

    st.markdown("---")
    st.markdown("## 💬 QUOTES神託（温度付きで選択）")
    lang = st.selectbox("LANG", ["ja", "en"], index=0)
    quote_temp = st.slider("格言温度（高→ランダム / 低→上位固定）", 0.2, 2.5, 1.2, 0.1)


# ============================================================
# 8) Excel解析（失敗しても落とさない）
# ============================================================
pack = None
excel_errors = []
sheet_names = []

if excel_bytes is not None and PANDAS_AVAILABLE:
    file_hash = _hash_bytes(excel_bytes)
    pack = load_excel_pack(excel_bytes, file_hash=file_hash)
    excel_errors = pack.get("errors", [])
    sheet_names = pack.get("sheet_names", [])

if excel_bytes is None:
    st.info("Excel未指定のため、内部fallbackデータで動作します。")

if excel_errors:
    # Step3が出ない原因になりがちだったので、ここでは stop() しない
    with st.expander("⚠ Excel読み込みの注意（止めずに継続します）", expanded=False):
        for e in excel_errors:
            st.warning(e)
        if sheet_names:
            st.caption("検出シート: " + ", ".join(sheet_names))


# ============================================================
# 9) マスター構築（VOWタイトル / CHAR一覧 / QUOTES）
# ============================================================
VOW_KEYS = [f"VOW_{i:02d}" for i in range(1, 13)]

def get_vow_titles_from_pack(pack: Optional[Dict]) -> Dict[str, str]:
    titles = dict(DEFAULT_VOW_TITLES)
    if not pack or not PANDAS_AVAILABLE:
        return titles
    df = pack.get("vow_master")
    if df is None or df.empty:
        return titles

    cols_norm = {_norm_colname(c): c for c in df.columns}
    id_col = cols_norm.get("vow_id") or cols_norm.get("vowid") or df.columns[0]
    title_col = cols_norm.get("title") or cols_norm.get("name") or cols_norm.get("vow_title") or None

    if title_col is None:
        return titles

    for _, r in df.iterrows():
        vid = str(r.get(id_col, "")).strip()
        if not vid:
            continue
        # VOW_1 なども標準化
        m = re.search(r"([1-9]|1[0-2])", vid.translate(str.maketrans("０１２３４５６７８９","0123456789")))
        if not m:
            continue
        idx = int(m.group(1))
        key = f"VOW_{idx:02d}"
        t = str(r.get(title_col, "")).strip()
        if t and t.lower() not in ("nan", "none"):
            titles[key] = t
    return titles

def get_char_list_from_pack(pack: Optional[Dict]) -> List[str]:
    # CHAR_TO_VOW の CHAR_ID__STD を使う
    if pack and PANDAS_AVAILABLE:
        df = pack.get("char_to_vow")
        if df is not None and "CHAR_ID__STD" in df.columns and not df.empty:
            return list(df["CHAR_ID__STD"].astype(str).unique())
    # fallback：12キャラ想定
    return [f"CHAR_{i:02d}" for i in range(1, 13)]

def get_char_to_vow_matrix(pack: Optional[Dict], chars: List[str]) -> np.ndarray:
    # shape: (n_char, 12)
    n = len(chars)
    M = np.zeros((n, 12), dtype=float)

    if pack and PANDAS_AVAILABLE:
        df = pack.get("char_to_vow")
        if df is not None and not df.empty and "CHAR_ID__STD" in df.columns:
            idx = {c: i for i, c in enumerate(chars)}
            for _, r in df.iterrows():
                cid = str(r.get("CHAR_ID__STD", "")).strip()
                if cid not in idx:
                    continue
                i = idx[cid]
                for j in range(12):
                    key = f"VOW_{j+1:02d}"
                    try:
                        M[i, j] = float(r.get(key, 0.0))
                    except Exception:
                        M[i, j] = 0.0

            # 正規化（重みのレンジが崩れても使えるように）
            mx = np.max(np.abs(M))
            if mx > 0:
                M = M / mx
            return M

    # fallback：ゆるい相関（適当でも動作）
    rng = np.random.default_rng(123)
    M = rng.normal(0, 1, size=(n, 12))
    M = M / max(1e-9, np.max(np.abs(M)))
    return M

def get_quotes_list(pack: Optional[Dict]) -> List[Dict]:
    quotes = list(DEFAULT_QUOTES)
    if not pack or not PANDAS_AVAILABLE:
        return quotes
    df = pack.get("quotes")
    if df is None or df.empty:
        return quotes

    cols_norm = {_norm_colname(c): c for c in df.columns}
    quote_col = cols_norm.get("quote") or cols_norm.get("格言") or cols_norm.get("テキスト") or None
    src_col = cols_norm.get("source") or cols_norm.get("出典") or cols_norm.get("作者") or None

    if quote_col is None:
        return quotes

    for _, r in df.iterrows():
        qt = str(r.get(quote_col, "")).strip()
        if not qt or qt.lower() in ("nan", "none"):
            continue
        src = str(r.get(src_col, "—")).strip() if src_col else "—"
        quotes.append({"quote": qt, "source": src})
    return quotes

VOW_TITLES = get_vow_titles_from_pack(pack)
CHARS = get_char_list_from_pack(pack)
CHAR_TO_VOW = get_char_to_vow_matrix(pack, CHARS)
QUOTES = get_quotes_list(pack)


# ============================================================
# 10) Step1：誓願入力（スライダー）＋テキスト
# ============================================================
left, right = st.columns([2.15, 1.0], gap="large")

with left:
    st.markdown("## Step 1：誓願入力（スライダー）＋テキスト（自動ベクトル化）")

    user_text = st.text_area(
        "あなたの状況を一文で（例：疲れていて決断ができない／新しい挑戦が怖い など）",
        value="疲れている。しばらく休みたい。",
        height=90,
        key="user_text",
    )

    st.caption("スライダー入力はTITLEを常時表示し、テキストからの自動推定と mix します。")

    manual_v = np.zeros(12, dtype=float)
    for i, k in enumerate(VOW_KEYS):
        label = f"{k}｜{VOW_TITLES.get(k, k)}"
        manual_v[i] = st.slider(label, 0.0, 5.0, 0.0 if i >= 8 else 2.0, 0.5, key=f"manual_{k}")

with right:
    st.markdown("## Step 3：結果（観測された神＋理由＋QUOTES神託）")
    st.caption("右側に「観測結果」「寄与した誓願」「格言（雰囲気）」を表示します。")


# ============================================================
# 11) テキスト→VOW（簡易ベクトル化）
#   ※本番はExcelにキーワード辞書があるならここへ統合可能
# ============================================================
# ざっくり：キーワードからVOWへ寄せるための辞書（必要なら増やす）
VOW_HINTS = {
    "VOW_01": ["迷い","決断","選択","断つ","優先","整理"],
    "VOW_02": ["静","待つ","焦り","時間","休む","落ち着く"],
    "VOW_03": ["内面","洞察","観察","深める","省察","考える"],
    "VOW_04": ["行動","踏み出す","やる","一歩","開始","検証"],
    "VOW_05": ["つながり","絆","家族","仲間","信頼","支え"],
    "VOW_06": ["挑戦","未知","学ぶ","獲得","選ぶ","成長"],
    "VOW_07": ["焦り","迷い","手放す","解放","軽く","離す"],
    "VOW_08": ["小さく","動く","少し","試す","小歩","改善"],
    "VOW_09": ["つながり","挑戦","協力","共創","場","連携"],
    "VOW_10": ["判断","機","待つ","急が","慎重","タイミング"],
    "VOW_11": ["静けさ","つながり","整える","調和","支え"],
    "VOW_12": ["加速","行動","挑戦","進む","勢い","推進"],
}

def text_to_vow_vector(text: str) -> np.ndarray:
    kws = extract_keywords_simple(text, top_n=8)
    v = np.zeros(12, dtype=float)
    for i, k in enumerate(VOW_KEYS):
        hints = VOW_HINTS.get(k, [])
        score = 0.0
        for kw in kws:
            for h in hints:
                if h in kw or kw in h:
                    score += 1.0
        v[i] = score
    # 0埋め回避
    if np.all(v == 0):
        v[1] = 1.0  # 静けさ
        v[0] = 0.6  # 迷い
    # 0..5 にスケール
    v = v / max(1e-9, np.max(v)) * 5.0
    return v

auto_v = text_to_vow_vector(user_text)
mix_v = alpha * manual_v + (1.0 - alpha) * auto_v


# ============================================================
# 12) Step2：グラフ（背景黒）＋ 誓願ベクトル表（ヘッダー黒）
# ============================================================
st.markdown("---")
st.markdown("## 📈 誓願ベクトル（manual / auto / mix）")

df_v = None
if PANDAS_AVAILABLE:
    df_v = pd.DataFrame({
        "VOW_ID": VOW_KEYS,
        "TITLE": [VOW_TITLES[k] for k in VOW_KEYS],
        "manual(0-5)": np.round(manual_v, 3),
        "auto(0-5)": np.round(auto_v, 3),
        "mix(0-5)": np.round(mix_v, 3),
    })

c1, c2 = st.columns([1.2, 1.0], gap="large")
with c1:
    # 折れ線：manual/auto/mix
    x = VOW_KEYS
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=x, y=auto_v, mode="lines+markers", name="auto"))
    fig_line.add_trace(go.Scatter(x=x, y=manual_v, mode="lines+markers", name="manual"))
    fig_line.add_trace(go.Scatter(x=x, y=mix_v, mode="lines+markers", name="mix"))

    fig_line.update_layout(
        height=340,
        paper_bgcolor="rgba(6,8,18,1)",
        plot_bgcolor="rgba(6,8,18,1)",
        font=dict(color="rgba(245,245,255,0.92)"),
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig_line.update_xaxes(showgrid=False)
    fig_line.update_yaxes(gridcolor="rgba(255,255,255,0.12)", zerolinecolor="rgba(255,255,255,0.12)")
    st.plotly_chart(fig_line, use_container_width=True, config={"displaylogo": False})

with c2:
    if df_v is not None:
        st.dataframe(style_df_dark(df_v), use_container_width=True, hide_index=True)
    else:
        st.caption("pandas未導入のため表のStyler表示ができません。")


# ============================================================
# 13) 観測（QUBO one-hot）
#   - energy を「mix_v とキャラ重み」の相性で作る（低いほど選ばれやすい）
# ============================================================
# 相性：- dot(mix_v, w_char) を energy にする（大きいほど低エネルギー）
# CHAR_TO_VOW は [-1..1] 程度に正規化されている想定（fallbackも正規化済）
compat = np.dot(CHAR_TO_VOW, (mix_v / 5.0))  # shape: (n_char,)
energy = -compat  # 低いほど良い

# 温度揺らぎ（観測分布用）
rng = np.random.default_rng(_adler_seed(user_text + f"|{alpha}|{P}|{sa_sweeps}|{sa_temp}"))
energies_samples = []
for _ in range(n_samples):
    noise = rng.normal(0, sa_temp * 0.02)
    e = energy + noise
    energies_samples.append(e)
energies_samples = np.array(energies_samples)  # (n_samples, n_char)

# 代表energy（平均）
energy_mean = np.mean(energies_samples, axis=0)

# one-hot QUBO（対角のみでOK：12～程度なら全探索）
x_best, e_best = one_hot_qubo(energy_mean, P=P)
best_idx = int(np.argmax(x_best))
best_char = CHARS[best_idx]
best_energy = float(energy_mean[best_idx])

# softmax（確率っぽく）
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / max(1e-12, np.sum(ez))

prob = softmax(-energy_mean)  # energy低→prob高
topk = min(3, len(CHARS))
top_idx = np.argsort(energy_mean)[:topk]


# ============================================================
# 14) Step3：結果表示（白ボックスを出さない）
# ============================================================
with right:
    # 結果テーブル
    if PANDAS_AVAILABLE:
        df_rank = pd.DataFrame({
            "順位": np.arange(1, topk+1),
            "CHAR_ID": [CHARS[i] for i in top_idx],
            "神": [CHARS[i] for i in top_idx],  # ここは本来「神名」列（Excelにあれば差し替え）
            "energy（低いほど選ばれやすい）": np.round([energy_mean[i] for i in top_idx], 6),
            "確率（softmax）": np.round([prob[i] for i in top_idx], 6),
        })
        st.dataframe(style_df_dark(df_rank), use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        <div class="card">
          <h3>✨ 今回“観測”された神：{best_char}</h3>
          <div class="smallnote">ここは “単発の観測（1回抽選）” です。下の観測分布は「同条件で複数回観測したらどう出るか」のサンプルです。</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # “雰囲気の良かった” 緑ボックス（継承）
    # 理由は「寄与したVOW上位」を文章化
    contrib = mix_v.copy()
    order = np.argsort(-contrib)[:4]
    reason_titles = [VOW_TITLES[VOW_KEYS[i]] for i in order]
    reason_vows = [VOW_KEYS[i] for i in order]
    reason_text = "・".join(reason_titles)

    st.markdown(
        f"""
        <div class="quote-green">
          いまの波は <b>{reason_text}</b> に寄っている。<br/>
          季節×時間（Stage）は流れを強める。<span class="smallnote">（解釈）</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 寄与した誓願（Top）
    if PANDAS_AVAILABLE:
        df_top = pd.DataFrame({
            "VOW": reason_vows,
            "TITLE": reason_titles,
            "mix(v)": np.round([mix_v[i] for i in order], 3),
            "寄与(w*v)": np.round([mix_v[i] * abs(CHAR_TO_VOW[best_idx, i]) for i in order], 3),
        })
        st.markdown("### 🧩 寄与した誓願（Top）")
        st.dataframe(style_df_dark(df_top), use_container_width=True, hide_index=True)

    # QUOTES神託（青ボックス継承）
    st.markdown("### 🗣️ QUOTES神託（温度付きで選択）")

    # 温度で「上位固定」⇔「ランダム」寄り
    # ここでは簡易に、候補をスコア付けして温度でサンプリング
    rng_q = np.random.default_rng(_adler_seed(user_text + f"|qt|{quote_temp}"))
    # スコア：テキストキーワードを含むほど高い（簡易）
    kws = extract_keywords_simple(user_text, top_n=6)
    scores = []
    for q in QUOTES:
        qt = q.get("quote", "")
        sc = 0.0
        for kw in kws:
            if kw in qt:
                sc += 2.0
        sc += 0.2
        scores.append(sc)
    scores = np.array(scores, dtype=float)

    # 温度：低いほど上位を選びやすく、高いほど均す
    logits = scores / max(1e-6, quote_temp)
    p = softmax(logits)
    pick = rng_q.choice(len(QUOTES), size=min(3, len(QUOTES)), replace=False, p=p)
    picked = [QUOTES[i] for i in pick]

    for idx, q in enumerate(picked, start=1):
        st.markdown(
            f"""
            <div class="quote-blue" style="margin-bottom:10px;">
              <b>神託{idx}</b><br/>
              「{q.get('quote','')}」<br/>
              <span class="smallnote">— {q.get('source','—')}</span>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 15) 2) エネルギー地形（全候補） + 3) 観測分布（サンプル）
# ============================================================
st.markdown("---")
st.markdown("## 2) エネルギー地形（全候補）")

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(x=CHARS, y=energy_mean, name="energy"))
fig_bar.update_layout(
    height=340,
    paper_bgcolor="rgba(6,8,18,1)",
    plot_bgcolor="rgba(6,8,18,1)",
    font=dict(color="rgba(245,245,255,0.92)"),
    margin=dict(l=10, r=10, t=20, b=10),
    showlegend=False,
)
fig_bar.update_xaxes(tickangle=90, showgrid=False)
fig_bar.update_yaxes(gridcolor="rgba(255,255,255,0.12)", zerolinecolor="rgba(255,255,255,0.12)")
st.plotly_chart(fig_bar, use_container_width=True, config={"displaylogo": False})

st.markdown("## 3) 観測分布（サンプル）")
# サンプルの最頻（出現回数）
# ここは簡易：各サンプルで最小energyのキャラを数える
winners = np.argmin(energies_samples, axis=1)
counts = np.bincount(winners, minlength=len(CHARS))

fig_hist = go.Figure()
fig_hist.add_trace(go.Bar(x=CHARS, y=counts, name="count"))
fig_hist.update_layout(
    height=340,
    paper_bgcolor="rgba(6,8,18,1)",
    plot_bgcolor="rgba(6,8,18,1)",
    font=dict(color="rgba(245,245,255,0.92)"),
    margin=dict(l=10, r=10, t=20, b=10),
    showlegend=False,
)
fig_hist.update_xaxes(tickangle=90, showgrid=False)
fig_hist.update_yaxes(gridcolor="rgba(255,255,255,0.12)", zerolinecolor="rgba(255,255,255,0.12)")
st.plotly_chart(fig_hist, use_container_width=True, config={"displaylogo": False})


# ============================================================
# 16) 4) テキストのキーワード抽出（UI） + 単語球体アート
# ============================================================
st.markdown("---")
st.markdown("## 4) テキストのキーワード抽出（簡易）")

kws = extract_keywords_simple(user_text, top_n=8)
st.caption("入力テキストが短い場合は、抽出が少なくなることがあります。")

cA, cB = st.columns([1.0, 1.2], gap="large")
with cA:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 抽出キーワード")
    st.write(" / ".join(kws))
    st.markdown("<div class='smallnote'>※ここを起点に “エネルギーが近い単語” を空間に配置して、縁（線）で結びます。</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with cB:
    st.markdown("### 🌐 単語の球体（誓願→キーワード→縁のネットワーク）")
    render_word_sphere(user_text, seed_key="sphere_v1", height=520)


# ============================================================
# 17) QUBO証拠（デバッグ表示）
# ============================================================
with st.expander("🧪 QUBO 証拠（デバッグ）", expanded=False):
    st.code(
        "E(x) = Σ_i (E_i * x_i) + P*(Σ_i x_i - 1)^2\n"
        f"P = {P:.2f}\n"
        f"x = {x_best.tolist()}\n"
        f"E(x) = {e_best:.6f}\n"
        f"best = {best_char} / energy_mean = {best_energy:.6f}\n",
        language="text"
    )
    st.caption("※ one-hot 制約（Σx=1）をペナルティ項で強制しています。")
