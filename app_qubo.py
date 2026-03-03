# -*- coding: utf-8 -*-
# ============================================================
# Q-Quest 量子神託（QUBO / STAGE×QUOTES） + 「縁の球体」(アート)
# - one-hot制約のQUBOで12神を観測
# - 誓願テキストからキーワード抽出し、単語の球体(アート)を表示（4枠）
# - UI: ダーク統一（file_uploader/入力/表/グラフ背景も黒）
# - Excel検出: VOW列検出を緩く（不足列は0補完、エラーで止めない）
# ============================================================

import io
import os
import re
import time
import zlib
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ------------------------------------------------------------
# 0) Streamlit config（必ず最初）
# ------------------------------------------------------------
st.set_page_config(
    page_title="Q-Quest 量子神託（QUBO / STAGE×QUOTES）",
    layout="wide",
)


# ------------------------------------------------------------
# 1) CSS（全体ダーク / 入力 / uploader / 表 / expander / plotly）
# ------------------------------------------------------------
DARK_CSS = """
<style>
:root{
  --bg0: rgba(6,8,18,1);
  --bg1: rgba(10,12,26,1);
  --panel: rgba(255,255,255,0.06);
  --panel2: rgba(255,255,255,0.08);
  --border: rgba(255,255,255,0.12);
  --text: rgba(245,245,255,0.94);
  --muted: rgba(235,235,255,0.72);
  --accent: rgba(255,80,80,0.95);
  --chip: rgba(255,255,255,0.10);
}

/* app background */
.stApp{
  background:
    radial-gradient(circle at 18% 24%, rgba(110,150,255,0.12), transparent 38%),
    radial-gradient(circle at 78% 68%, rgba(255,160,220,0.08), transparent 44%),
    radial-gradient(circle at 50% 50%, rgba(255,255,255,0.03), transparent 55%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
}

/* main padding */
.block-container{ padding-top: 1.2rem; }

/* base typography */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li{
  color: var(--text);
  font-family: "Hiragino Mincho ProN","Yu Mincho","Noto Serif JP",serif;
  letter-spacing: 0.02em;
}
h1,h2,h3,h4{
  color: rgba(250,250,255,0.96) !important;
  font-family: "Hiragino Mincho ProN","Yu Mincho","Noto Serif JP",serif !important;
  font-weight: 650 !important;
}

/* sidebar */
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.07);
  border-right: 1px solid var(--border);
  backdrop-filter: blur(10px);
}
section[data-testid="stSidebar"] *{
  color: var(--text);
}

/* --- inputs: text area / text input / select / number --- */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input{
  background: rgba(0,0,0,0.35) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
div[data-testid="stTextArea"] textarea::placeholder,
div[data-testid="stTextInput"] input::placeholder{
  color: rgba(235,235,255,0.55) !important;
}

/* selectbox */
div[data-testid="stSelectbox"] > div{
  background: rgba(0,0,0,0.35) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
div[data-testid="stSelectbox"] *{
  color: var(--text) !important;
}

/* file uploader (白地対策の本丸) */
div[data-testid="stFileUploader"]{
  background: rgba(0,0,0,0.30) !important;
  border: 1px dashed rgba(255,255,255,0.25) !important;
  border-radius: 14px !important;
  padding: 10px !important;
}
div[data-testid="stFileUploader"] *{
  color: var(--text) !important;
}
div[data-testid="stFileUploader"] button{
  background: rgba(255,255,255,0.10) !important;
  color: var(--text) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 12px !important;
}

/* sliders */
div[data-testid="stSlider"] *{ color: var(--text) !important; }

/* buttons */
.stButton>button{
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  background: linear-gradient(90deg, rgba(255,80,80,0.85), rgba(160,120,255,0.65)) !important;
  color: rgba(255,255,255,0.98) !important;
  padding: 0.55rem 0.9rem !important;
}
.stButton>button:hover{
  filter: brightness(1.07);
}

/* expander */
details{
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
  padding: 6px 10px !important;
}
details summary{ color: var(--text) !important; }

/* plotly container */
div[data-testid="stPlotlyChart"] > div{
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 18px 60px rgba(0,0,0,0.28);
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

/* dataframe/table (ヘッダ白問題の解決) */
div[data-testid="stDataFrame"]{
  background: rgba(0,0,0,0.25) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
  padding: 6px !important;
}
div[data-testid="stDataFrame"] *{
  color: rgba(245,245,255,0.92) !important;
}
div[data-testid="stDataFrame"] thead tr th{
  background: rgba(0,0,0,0.70) !important;
  color: rgba(255,255,255,0.95) !important;
  border-bottom: 1px solid rgba(255,255,255,0.12) !important;
}
div[data-testid="stDataFrame"] tbody tr td{
  background: rgba(0,0,0,0.35) !important;
  border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}

/* captions */
.smallnote{opacity:0.80; font-size:0.92rem;}
.card{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px;
  box-shadow: 0 18px 60px rgba(0,0,0,0.18);
}

/* quote boxes (雰囲気継承) */
.quote-green{
  background: rgba(28, 95, 55, 0.55);
  border: 1px solid rgba(70, 190, 120, 0.35);
  color: rgba(225,255,235,0.95);
  border-radius: 16px;
  padding: 14px 14px;
}
.quote-blue{
  background: rgba(20, 70, 140, 0.55);
  border: 1px solid rgba(120, 170, 255, 0.35);
  color: rgba(230,245,255,0.97);
  border-radius: 16px;
  padding: 14px 14px;
}
.quote-blue b{ color: rgba(210,235,255,1.0); }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------
# 2) 定数（VOWの表示名）
# ------------------------------------------------------------
VOW_TITLES = {
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
VOW_KEYS = list(VOW_TITLES.keys())


# ------------------------------------------------------------
# 3) Excel読み込み（ゆるい検出）
# ------------------------------------------------------------
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _norm_col(c: str) -> str:
    """列名をゆるく正規化（全角/半角/空白/大小など吸収）"""
    s = str(c).strip()
    s = s.replace("　", " ").strip()
    s = s.upper()
    s = re.sub(r"\s+", "", s)
    s = s.replace("-", "_")
    return s


def _guess_vow_cols(df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    """
    df内のVOW列をゆるく検出。
    戻り: (canonical->actual_col のマップ, 見つからなかったcanonical一覧)
    """
    actual_cols = list(df.columns)
    norm_map = {_norm_col(c): c for c in actual_cols}

    found: Dict[str, str] = {}
    missing: List[str] = []

    for canonical in VOW_KEYS:
        # canonical: VOW_01
        target_num = canonical.split("_")[1]  # "01"
        # いろんな表記ゆれを許容
        patterns = [
            f"VOW_{target_num}",
            f"VOW{target_num}",
            f"VOW_{int(target_num)}",     # VOW_1
            f"VOW{int(target_num)}",      # VOW1
        ]
        hit = None
        for p in patterns:
            if p in norm_map:
                hit = norm_map[p]
                break

        # さらにゆるい：正規表現で拾う
        if hit is None:
            # 例: "VOW01(重み)" / "vow_01_weight" / "VOW_01 " 等
            rgx = re.compile(rf"^VOW_?0?{int(target_num)}($|[^0-9].*)", re.I)
            for nc, orig in norm_map.items():
                if rgx.match(nc):
                    hit = orig
                    break

        if hit is None:
            missing.append(canonical)
        else:
            found[canonical] = hit

    return found, missing


def _find_best_sheet(excel: pd.ExcelFile) -> Tuple[str, pd.DataFrame]:
    """
    「CHARの定義 + VOW列」が入ってそうなシートを探して返す。
    優先:
      1) 列に CHAR_ID/神/NAME などがあり、VOWが複数ある
      2) なければ最初のシート
    """
    best_name = excel.sheet_names[0]
    best_score = -1
    best_df = excel.parse(best_name)

    for name in excel.sheet_names:
        try:
            df = excel.parse(name)
        except Exception:
            continue
        cols = [_norm_col(c) for c in df.columns]

        has_char = any(c in ("CHAR_ID", "CHARID", "ID") for c in cols) or any("神" in str(c) for c in df.columns)
        vow_found, _ = _guess_vow_cols(df)
        vow_count = len(vow_found)

        score = 0
        if has_char:
            score += 3
        score += vow_count

        if score > best_score:
            best_score = score
            best_name = name
            best_df = df

    return best_name, best_df


@dataclass
class ExcelPack:
    sheet_used: str
    df_char: pd.DataFrame
    df_quotes: Optional[pd.DataFrame]


@st.cache_data(show_spinner=False)
def load_excel_pack(excel_bytes: bytes, file_hash: str) -> ExcelPack:
    bio = io.BytesIO(excel_bytes)
    xls = pd.ExcelFile(bio, engine="openpyxl")

    sheet_used, df_char = _find_best_sheet(xls)

    # QUOTESシートはあれば読む（なければNone）
    df_quotes = None
    for s in xls.sheet_names:
        if _norm_col(s) in ("QUOTES", "QUOTE", "格言", "コトワザ"):
            try:
                df_quotes = xls.parse(s)
            except Exception:
                df_quotes = None
            break

    return ExcelPack(sheet_used=sheet_used, df_char=df_char, df_quotes=df_quotes)


# ------------------------------------------------------------
# 4) QUBO（one-hot）: E(x)= sum(E_i x_i) + P(sum x -1)^2
# ------------------------------------------------------------
def solve_onehot_qubo_sa(E: np.ndarray, P: float, sweeps: int, temp: float, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """
    12個くらいなら厳密解でも良いが、雰囲気としてSAで解く。
    """
    n = len(E)
    # 初期はランダムone-hot
    x = np.zeros(n, dtype=int)
    x[rng.integers(0, n)] = 1

    def energy(xv: np.ndarray) -> float:
        return float(np.dot(E, xv) + P * (np.sum(xv) - 1) ** 2)

    curE = energy(x)
    bestx = x.copy()
    bestE = curE

    # SA
    T0 = max(1e-6, float(temp))
    for t in range(int(sweeps)):
        T = T0 * (0.995 ** t)  # ほんのり冷却

        # one-hotのまま遷移（1を別位置へ移す）
        i1 = int(np.argmax(x))
        i2 = int(rng.integers(0, n))
        if i2 == i1:
            continue

        x2 = x.copy()
        x2[i1] = 0
        x2[i2] = 1

        e2 = energy(x2)
        dE = e2 - curE
        if dE <= 0 or rng.random() < np.exp(-dE / max(T, 1e-9)):
            x = x2
            curE = e2
            if curE < bestE:
                bestE = curE
                bestx = x.copy()

    return bestx, bestE


# ------------------------------------------------------------
# 5) QUOTES（雰囲気継承：緑/青）
# ------------------------------------------------------------
def pick_text(row: pd.Series, candidates: List[str]) -> str:
    for c in candidates:
        if c in row.index:
            v = str(row.get(c, "")).strip()
            if v and v.lower() not in ("nan", "none"):
                return v
    return ""


def build_quotes(df_quotes: Optional[pd.DataFrame]) -> List[Dict]:
    base = [
        {"quote": "未来は私たちの選択にかかっている。", "source": "カール・J・フォンペルティ"},
        {"quote": "途中であきらめちゃいけない。", "source": "ルイ・アームストロング"},
        {"quote": "人生の最良の瞬間は、今この瞬間だ。", "source": "ヒンディーのことわざ"},
        {"quote": "過去は過去であり、未来は未来である。今この瞬間こそがプレゼントだ。", "source": "エリオット・D・アディ"},
        {"quote": "成功は、自分の強みを活かすことから始まる。", "source": "ピーター・ドラッカー"},
    ]
    if df_quotes is None or df_quotes.empty:
        return base

    quotes = []
    for _, row in df_quotes.iterrows():
        qt = pick_text(row, ["格言", "QUOTE", "Quote", "quote", "言葉", "文"])
        if not qt:
            continue
        src = pick_text(row, ["出典", "SOURCE", "Source", "作者", "典拠"]) or "—"
        quotes.append({"quote": qt, "source": src})

    # baseを足して最低限確保
    if len(quotes) < 3:
        quotes = quotes + base
    return quotes


def choose_quotes_with_temperature(quotes: List[Dict], temperature: float, k: int, rng: np.random.Generator) -> List[Dict]:
    """
    温度が高いほどランダム、低いほど上位（ここでは擬似的に先頭寄り）。
    """
    if not quotes:
        return []

    temperature = float(np.clip(temperature, 0.2, 2.0))
    n = len(quotes)

    # 擬似スコア: 先頭ほど良い
    scores = np.linspace(1.0, 0.3, n)

    # 温度でsoftmax
    logits = scores / temperature
    ex = np.exp(logits - np.max(logits))
    p = ex / np.sum(ex)

    idx = rng.choice(np.arange(n), size=min(k, n), replace=False, p=p)
    return [quotes[i] for i in idx]


# ------------------------------------------------------------
# 6) キーワード抽出（簡易） + 球体アート（Plotly 3D）
# ------------------------------------------------------------
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
    "行動": ["努力","継続","忍耐","誠実","正直"],
    "哲学": ["調和","バランス","自然","美","道","真実","自由","正義"],
    "関係": ["絆","つながり","家族","友人","仲間","信頼","尊敬","協力"],
    "内的": ["静けさ","集中","覚悟","決意","勇気","強さ","柔軟性","寛容"],
    "時間": ["今","瞬間","過程","変化","進化","発展","循環","流れ"],
}
STOP_TOKENS = set([
    "した","たい","いる","こと","それ","これ","ため","よう","ので","から",
    "です","ます","ある","ない","そして","でも","しかし","また",
    "自分","私","あなた","もの","感じ","気持ち","今日",
    "に","を","が","は","と","も","で","へ","や","の"
])


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    text = (text or "").strip()
    if not text:
        return ["静けさ", "迷い"]

    found = [w for w in GLOBAL_WORDS_DATABASE if w in text]
    if found:
        return found[:top_n]

    text_clean = re.sub(r"[0-9０-９、。．,.!！?？\(\)\[\]{}「」『』\"'：:;／/\\\n\r\t]+", " ", text)
    tokens = [t.strip() for t in re.split(r"\s+", text_clean) if t.strip()]
    tokens = [t for t in tokens if (len(t) >= 2 and t not in STOP_TOKENS)]
    if not tokens:
        return ["静けさ", "迷い"]
    tokens = sorted(tokens, key=lambda s: (-len(s), s))
    return tokens[:top_n]


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


def solve_qubo_placement(Q: np.ndarray, words: List[str], center_indices: List[int], energies: Dict[str, float],
                         rng: np.random.Generator, n_iterations: int = 80) -> np.ndarray:
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

    for _ in range(int(n_iterations)):
        for i in range(n):
            if i in center_indices:
                continue
            force = np.zeros(3, dtype=float)

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

            for j in range(n):
                if i == j or j in center_indices:
                    continue
                eij = Q[i, j]
                if eij < -0.3:
                    vec = pos[j] - pos[i]
                    d = np.linalg.norm(vec)
                    if d > 0.01:
                        force += vec / d * (abs(eij) * 0.08)
                elif eij > 0.2:
                    vec = pos[i] - pos[j]
                    d = np.linalg.norm(vec)
                    if d > 0.01:
                        force += vec / d * (abs(eij) * 0.03)

            pos[i] += force * 0.15

    return pos


def make_seed(s: str) -> int:
    return int(zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF)


def build_sphere_figure(user_text: str, topn_kw: int, n_total: int, jitter: float, noise: float, iters: int) -> Tuple[go.Figure, List[str]]:
    seed = make_seed(f"{user_text}|{topn_kw}|{n_total}|{jitter}|{noise}|{iters}")
    rng = np.random.default_rng(seed)

    keywords = extract_keywords(user_text, top_n=topn_kw)
    center_set = set(keywords)

    network = build_word_network(keywords, GLOBAL_WORDS_DATABASE, n_total=n_total, rng=rng, jitter=jitter)
    pos = solve_qubo_placement(network["Q"], network["words"], network["center_indices"], network["energies"], rng=rng, n_iterations=iters)
    if noise > 0:
        pos = pos + rng.normal(0, noise, size=pos.shape)

    fig = go.Figure()

    # stars (固定)
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

    words = network["words"]
    energies = network["energies"]
    edges = network["edges"]
    center_indices = network["center_indices"]

    # edges
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

    # nodes
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
                colors.append("rgba(220,240,255,0.72)")
            else:
                colors.append("rgba(255,255,255,0.55)")
            labels.append(w)

    center_idx = [i for i, w in enumerate(labels) if w in center_set]
    other_idx = [i for i, w in enumerate(labels) if w not in center_set]

    if other_idx:
        oi = np.array(other_idx, dtype=int)
        fig.add_trace(go.Scatter3d(
            x=pos[oi, 0], y=pos[oi, 1], z=pos[oi, 2],
            mode="markers+text",
            text=[labels[i] for i in oi],
            textposition="top center",
            textfont=dict(size=18, color="rgba(255,255,255,1.0)"),
            marker=dict(size=[sizes[i] for i in oi], color=[colors[i] for i in oi]),
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
            textfont=dict(size=24, color="rgba(255,80,80,1.0)"),
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
    return fig, keywords


# ------------------------------------------------------------
# 7) UI：サイドバー（Excel / QUBO設定 / テキスト設定 / QUOTES温度）
# ------------------------------------------------------------
def init_state():
    defaults = {
        "uploaded_excel_hash": "",
        "P": 40.0,
        "samples": 300,
        "sweeps": 420,
        "temp": 1.20,

        "ngram": 3,
        "mix_alpha": 0.55,  # 1=スライダー寄り, 0=テキスト寄り

        "quote_temp": 1.20,
        "stage_id": "ST_01|春×朝",
        "auto_stage": True,

        "wish_text": "例：迷いを断ちたいが、今は焦らず機を待つべきか悩んでいる…",
        "vows": {k: 0.0 for k in VOW_KEYS},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


st.title("🔮 Q-Quest 量子神託（QUBO / STAGE×QUOTES）")
st.caption("one-hot制約のQUBOで12神を観測し、誓願（スライダー＋テキスト）と格言（緑/青）と可視化を表示します。")

with st.sidebar:
    st.markdown("## 📁 データ")
    uploaded = st.file_uploader("統合Excel（pack）", type=["xlsx"])

    st.markdown("---")
    st.markdown("## ⚙️ QUBO設定（one-hot）")
    st.slider("one-hot ペナルティ P", 1.0, 200.0, float(st.session_state["P"]), 1.0, key="P")
    st.slider("サンプル数（観測分布）", 50, 1200, int(st.session_state["samples"]), 10, key="samples")
    st.slider("SA sweeps（揺らぎ）", 50, 2000, int(st.session_state["sweeps"]), 10, key="sweeps")
    st.slider("SA温度（大→揺らぐ）", 0.20, 2.00, float(st.session_state["temp"]), 0.01, key="temp")

    st.markdown("---")
    st.markdown("## 🧠 テキスト＋誓願（自動ベクトル化）")
    st.selectbox("n-gram（簡易）", options=[2, 3, 4], index=[2,3,4].index(int(st.session_state["ngram"])), key="ngram")
    st.slider("mix比率 α（1=スライダー寄り / 0=テキスト寄り）", 0.0, 1.0, float(st.session_state["mix_alpha"]), 0.01, key="mix_alpha")

    st.markdown("---")
    st.markdown("## 🟦 QUOTES神託（温度付きで選択）")
    st.selectbox("LANG", options=["ja"], index=0)
    st.slider("格言温度（高→ランダム / 低→上位固定）", 0.20, 2.00, float(st.session_state["quote_temp"]), 0.01, key="quote_temp")


# ------------------------------------------------------------
# 8) Excelを読む（なければデモ）
# ------------------------------------------------------------
excel_pack: Optional[ExcelPack] = None
df_char: Optional[pd.DataFrame] = None
df_quotes: Optional[pd.DataFrame] = None
excel_status = ""

if uploaded is not None:
    b = uploaded.getvalue()
    h = _sha(b)
    excel_pack = load_excel_pack(b, file_hash=h)
    df_char = excel_pack.df_char
    df_quotes = excel_pack.df_quotes
    excel_status = f"Excel読込OK（sheet: {excel_pack.sheet_used}）"
else:
    excel_status = "Excel未指定（デモモード）"

st.info(excel_status)

# デモ用：12神だけ最低限作る（Excelが無い時）
if df_char is None:
    df_char = pd.DataFrame({
        "CHAR_ID": [f"CHAR_{i:02d}" for i in range(1, 13)],
        "神": ["秋葉三尺坊","真空管大将軍","LED弁財天","磁気記録黒龍","無線傍受観音","基板曼荼羅",
              "絶対零度明王","ジャンク再生童子","真空オーディオ如来","光速通信韋駄天","半導体文殊","絶対温度明王"],
    })
    # VOW列（適当に形だけ）
    for k in VOW_KEYS:
        df_char[k] = np.random.default_rng(0).uniform(0.0, 1.0, len(df_char))


# ここが「2)検出が厳しすぎる」の対策
# - VOW列をゆるく検出
# - 見つからないVOWは0で補完し、止めずに警告
vow_map, vow_missing = _guess_vow_cols(df_char)
for canonical in VOW_KEYS:
    if canonical not in vow_map:
        # 無ければ作る（0）
        df_char[canonical] = 0.0
    else:
        # 見つかった列名をcanonicalにコピー（元列名が変でも統一する）
        src = vow_map[canonical]
        if src != canonical:
            df_char[canonical] = df_char[src]

if vow_missing:
    st.warning(
        "Excel内で見つからないVOW列がありました（止めずに 0 で補完しています）: "
        + ", ".join(vow_missing)
        + "  ※列名例：VOW_01, VOW01, VOW_1 などは自動検出します。"
    )

# CHAR列もゆるく（無ければ作る）
if "CHAR_ID" not in df_char.columns:
    # それっぽい列を探す
    cand = None
    for c in df_char.columns:
        if _norm_col(c) in ("CHARID","ID"):
            cand = c
            break
    if cand is not None:
        df_char["CHAR_ID"] = df_char[cand].astype(str)
    else:
        df_char["CHAR_ID"] = [f"CHAR_{i:02d}" for i in range(1, len(df_char)+1)]

if "神" not in df_char.columns:
    # NAMEっぽい列
    cand = None
    for c in df_char.columns:
        if _norm_col(c) in ("NAME","GOD","神名"):
            cand = c
            break
    if cand is not None:
        df_char["神"] = df_char[cand].astype(str)
    else:
        df_char["神"] = [f"神_{i:02d}" for i in range(1, len(df_char)+1)]

df_char = df_char.copy()
df_char["CHAR_ID"] = df_char["CHAR_ID"].astype(str)
df_char["神"] = df_char["神"].astype(str)


# ------------------------------------------------------------
# 9) Step1：誓願入力（テキスト＋スライダー）
# ------------------------------------------------------------
st.markdown("## Step 1：誓願入力（スライダー）＋テキスト（自動ベクトル化）")

left1, right1 = st.columns([2.1, 1.1], gap="large")

with left1:
    wish_text = st.text_area(
        "あなたの状況を一文で（例：疲れていて決断ができない／新しい挑戦が怖い など）",
        value=st.session_state["wish_text"],
        height=90,
        key="wish_text",
    )
    st.caption("スライダー入力はTITLEを常時表示し、テキストからの自動推定と mix します。")

    # VOW sliders
    for k in VOW_KEYS:
        title = VOW_TITLES[k]
        st.slider(
            f"{k}｜{title}",
            min_value=0.0,
            max_value=5.0,
            value=float(st.session_state["vows"].get(k, 0.0)),
            step=0.5,
            key=f"slider_{k}",
        )
    # stateに反映（widget->session代入はOK：widget keyが別なので）
    st.session_state["vows"] = {k: float(st.session_state.get(f"slider_{k}", 0.0)) for k in VOW_KEYS}

with right1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Step 3：結果（観測された神＋理由＋QUOTES神託）")
    st.caption("右側に「観測結果」「寄与した誓願」「格言（緑/青）」を表示します。")
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# 10) 観測（QUBO one-hot）
# ------------------------------------------------------------
def vow_vector_from_text(text: str) -> np.ndarray:
    """
    簡易：テキストから抽出したキーワードがVOWタイトルに近いほど加点…などを入れても良いが
    ここでは“雰囲気用”に、キーワード数に応じて少しだけ分配する。
    """
    kws = extract_keywords(text, top_n=5)
    v = np.zeros(len(VOW_KEYS), dtype=float)
    if not kws:
        return v
    # なんとなく：文字列の長さで分散
    for i, k in enumerate(kws):
        idx = (len(k) + i) % len(VOW_KEYS)
        v[idx] += 1.0
    if np.max(v) > 0:
        v = v / np.max(v) * 2.0  # 0〜2程度
    return v


def compute_mix_vow(slider_v: Dict[str, float], text: str, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    manual = np.array([slider_v[k] for k in VOW_KEYS], dtype=float)
    auto = vow_vector_from_text(text)
    mix = alpha * manual + (1.0 - alpha) * auto
    return manual, auto, mix


def compute_char_energy(df: pd.DataFrame, vow_mix: np.ndarray) -> np.ndarray:
    """
    各神のエネルギー（小さいほど選ばれやすい）を作る。
    ここはプロジェクト固有なので、基本は「重み×vow」の線形でOK。
    """
    W = df[VOW_KEYS].to_numpy(dtype=float)  # (n_char, 12)
    # “寄与が大きいほどエネルギーが小さくなる”ようにマイナス
    score = W @ vow_mix  # 大きいほど良い
    E = -score
    return E.astype(float)


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


# 実行ボタン（左下の雰囲気に寄せる）
st.markdown("---")
run_col1, run_col2 = st.columns([1.0, 3.0])
with run_col1:
    run = st.button("🧪 観測する（QUBOから抽出）", use_container_width=True)
with run_col2:
    st.caption("※ Excel列名ゆれは許容。見つからないVOW列は0補完して動作します。")

# 初回も自動で1回まわす（runを押さなくても表示）
if "last_result" not in st.session_state:
    st.session_state["last_result"] = {}

if run or not st.session_state["last_result"]:
    rng = np.random.default_rng(make_seed(st.session_state["wish_text"] + str(time.time())[:6]))

    manual_v, auto_v, mix_v = compute_mix_vow(st.session_state["vows"], st.session_state["wish_text"], float(st.session_state["mix_alpha"]))
    E = compute_char_energy(df_char, mix_v)

    # QUBO one-hot
    x_best, Ebest = solve_onehot_qubo_sa(
        E=E,
        P=float(st.session_state["P"]),
        sweeps=int(st.session_state["sweeps"]),
        temp=float(st.session_state["temp"]),
        rng=rng
    )
    chosen_idx = int(np.argmax(x_best))

    # 観測分布（サンプル）
    counts = np.zeros(len(df_char), dtype=int)
    for _ in range(int(st.session_state["samples"])):
        xb, _ = solve_onehot_qubo_sa(E=E, P=float(st.session_state["P"]), sweeps=int(st.session_state["sweeps"]), temp=float(st.session_state["temp"]), rng=rng)
        counts[int(np.argmax(xb))] += 1

    # 上位3も出す（エネルギーの低い順）
    order = np.argsort(E)[:3]

    st.session_state["last_result"] = {
        "manual": manual_v,
        "auto": auto_v,
        "mix": mix_v,
        "E": E,
        "x": x_best,
        "Ebest": Ebest,
        "chosen_idx": chosen_idx,
        "counts": counts,
        "top3": order,
    }

res = st.session_state["last_result"]
manual_v = res["manual"]
auto_v = res["auto"]
mix_v = res["mix"]
E = res["E"]
x_best = res["x"]
chosen_idx = res["chosen_idx"]
counts = res["counts"]
top3 = res["top3"]


# ------------------------------------------------------------
# 11) Step3：結果表示（表・キャラ・寄与・QUOTES）
# ------------------------------------------------------------
st.markdown("## Step 3：結果（観測された神＋理由＋QUOTES神託）")

left2, right2 = st.columns([2.1, 1.1], gap="large")

# 右上の結果表
with right2:
    # 結果表（上位3）
    top_df = pd.DataFrame({
        "順位": [1, 2, 3],
        "CHAR_ID": df_char.loc[top3, "CHAR_ID"].values,
        "神": df_char.loc[top3, "神"].values,
        "energy（低いほど選ばれやすい）": E[top3],
        "確率（softmax）": softmax(-E)[top3],  # -Eがスコア
    })
    st.dataframe(top_df, use_container_width=True, hide_index=True)

    chosen = df_char.iloc[chosen_idx]
    st.markdown(f"### 🌟 今回“観測”された神：{chosen['神']}（{chosen['CHAR_ID']}）")

    # キャラクター画像（assets/images/characters に置いてある前提）
    # 例: CHAR_01.png / CHAR_01.webp / CHAR_01.jpg / CHAR_01_p1.png など複数を許容
    img_dir = Path("assets/images/characters")
    img = None
    if img_dir.exists():
        candidates = [
            img_dir / f"{chosen['CHAR_ID']}.png",
            img_dir / f"{chosen['CHAR_ID']}.webp",
            img_dir / f"{chosen['CHAR_ID']}.jpg",
            img_dir / f"{chosen['CHAR_ID']}_p1.png",
            img_dir / f"{chosen['CHAR_ID']}_p1.webp",
        ]
        for cp in candidates:
            if cp.exists():
                img = cp
                break
    if img is not None:
        st.image(str(img), use_container_width=True)
        st.caption(f"{chosen['神']}（{img.name}）")
    else:
        st.caption("※キャラクター画像が見つかりません（assets/images/characters 配下を確認）")

    # 緑の格言（雰囲気継承）
    st.markdown(
        "<div class='quote-green'>"
        "いまの波は <b>迷いを断つ</b>・<b>行動と挑戦を加速</b>・<b>判断を急がず、機を待つ</b> に寄っている。"
        "季節や時間（Stage）は流れを強める。"
        "</div>",
        unsafe_allow_html=True
    )

    # 寄与した誓願（Top）
    contrib = pd.DataFrame({
        "VOW": VOW_KEYS,
        "TITLE": [VOW_TITLES[k] for k in VOW_KEYS],
        "mix(v)": mix_v,
        "W(char,v)": df_char.loc[chosen_idx, VOW_KEYS].to_numpy(dtype=float),
    })
    contrib["寄与(w*v)"] = contrib["mix(v)"] * contrib["W(char,v)"]
    contrib = contrib.sort_values("寄与(w*v)", ascending=False).head(6)

    st.markdown("### 🧩 寄与した誓願（Top）")
    st.dataframe(contrib, use_container_width=True, hide_index=True)

    # QUOTES（青）—温度付きで3つ
    quotes_list = build_quotes(df_quotes)
    rngq = np.random.default_rng(make_seed(st.session_state["wish_text"] + "|quotes"))
    picked = choose_quotes_with_temperature(quotes_list, temperature=float(st.session_state["quote_temp"]), k=3, rng=rngq)

    st.markdown("### 🟦 QUOTES神託（温度付きで選択）")
    for i, q in enumerate(picked, start=1):
        st.markdown(
            "<div class='quote-blue'>"
            f"<b>神託{i}</b><br>"
            f"「{q['quote']}」<br>"
            f"— {q.get('source','—')}"
            "</div>",
            unsafe_allow_html=True
        )


# ------------------------------------------------------------
# 12) 可視化：1) auto/manual/mix 2) energy 3) 観測分布 4) 球体アート
# ------------------------------------------------------------
with left2:
    st.markdown("## 📊 可視化：テキストの影響・観測分布・エネルギー地形")

    # 1) テキスト＋誓願 自動推定の影響
    df_v = pd.DataFrame({
        "VOW": VOW_KEYS,
        "manual": manual_v,
        "auto": auto_v,
        "mix": mix_v
    })
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_v["VOW"], y=df_v["auto"], mode="lines+markers", name="auto"))
    fig1.add_trace(go.Scatter(x=df_v["VOW"], y=df_v["manual"], mode="lines+markers", name="manual"))
    fig1.add_trace(go.Scatter(x=df_v["VOW"], y=df_v["mix"], mode="lines+markers", name="mix"))
    fig1.update_layout(
        title="1) テキスト＋誓願 自動推定の影響（auto vs manual vs mix）",
        paper_bgcolor="rgba(6,8,18,1)",
        plot_bgcolor="rgba(6,8,18,1)",
        font=dict(color="rgba(245,245,255,0.92)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.10)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.10)"),
        legend=dict(bgcolor="rgba(0,0,0,0.20)")
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 2) エネルギー地形（全候補）
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df_char["神"], y=E))
    fig2.update_layout(
        title="2) エネルギー地形（全候補）",
        paper_bgcolor="rgba(6,8,18,1)",
        plot_bgcolor="rgba(6,8,18,1)",
        font=dict(color="rgba(245,245,255,0.92)"),
        xaxis=dict(tickangle=90, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.10)"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3) 観測分布（サンプル）
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=df_char["神"], y=counts))
    fig3.update_layout(
        title="3) 観測分布（サンプル）",
        paper_bgcolor="rgba(6,8,18,1)",
        plot_bgcolor="rgba(6,8,18,1)",
        font=dict(color="rgba(245,245,255,0.92)"),
        xaxis=dict(tickangle=90, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.10)"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # 4) テキストのキーワード抽出 → 球体アート（追加機能）
    st.markdown("### 4) テキストのキーワード抽出（簡易）＋単語の球体（アート）")

    if (st.session_state["wish_text"] or "").strip():
        fig_sphere, kws = build_sphere_figure(
            user_text=st.session_state["wish_text"],
            topn_kw=5,
            n_total=30,
            jitter=0.10,
            noise=0.06,
            iters=80
        )
        st.caption("抽出キーワード：" + " / ".join(kws))
        st.plotly_chart(
            fig_sphere,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "scrollZoom": True,
                "displaylogo": False,
                "doubleClick": "reset",
            }
        )
    else:
        st.caption("（入力テキストが空のため、球体を生成できません）")

    # QUBO証拠（デバッグ）
    with st.expander("🧠 QUBO 証拠（デバッグ）", expanded=False):
        st.write(f"P = {float(st.session_state['P']):.2f}")
        st.write("x =", x_best.tolist())
        st.write("Ebest =", float(res["Ebest"]))
        st.latex(r"E(\mathbf{x})=\sum_i E_i x_i + P\left(\sum_i x_i-1\right)^2")


# ------------------------------------------------------------
# 13) Excelエラー対策の説明（UI内に短く表示）
# ------------------------------------------------------------
with st.expander("🔧 Excel検出が不安なとき（列名のコツ）", expanded=False):
    st.markdown(
        "- VOW列は `VOW_01` 〜 `VOW_12` が理想ですが、`VOW01` / `VOW1` / `vow_01_weight` のような表記ゆれも自動検出します。\n"
        "- 見つからないVOW列は **0で補完**して動くので、まずは動作優先でOKです。\n"
        "- `CHAR_ID` と `神` 列があると安定します（無い場合も推定して作ります）。"
    )
