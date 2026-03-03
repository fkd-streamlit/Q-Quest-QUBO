# -*- coding: utf-8 -*-
"""
Q-Quest 量子神託（QUBO / STAGE×QUOTES）
- Excel統合読込（柔軟なシート名/列名検出）
- VOWスライダー + テキスト（簡易キーワード抽出/自動推定） → mix
- CHAR_TO_VOW から QUBOを構築し、one-hot制約で1キャラ観測（SA）
- Step3で「候補表」「観測キャラ」「理由」「キャラ画像」「格言」表示
- UI全面ダーク化（入力欄/サイドバー/表/ヘッダ）
"""

import os
import re
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# =========================
# 0) Streamlit config (must be first)
# =========================
st.set_page_config(
    page_title="Q-Quest 量子神託（QUBO / STAGE×QUOTES）",
    page_icon="🔮",
    layout="wide",
)


# =========================
# 1) THEME / CSS (force dark inputs & tables)
# =========================
SPACE_CSS = """
<style>
/* ---- base ---- */
html, body, [class*="css"]  { font-family: "Noto Sans JP", system-ui, -apple-system, "Segoe UI", sans-serif; }
.stApp {
  background:
    radial-gradient(1200px 600px at 20% 10%, rgba(130, 100, 255, 0.18), transparent 60%),
    radial-gradient(1200px 600px at 80% 30%, rgba( 80, 220, 255, 0.10), transparent 55%),
    linear-gradient(180deg, #070814 0%, #070812 60%, #060712 100%);
  color: rgba(245,245,255,0.92);
}

/* ---- header (top white bar mitigation) ---- */
header[data-testid="stHeader"]{
  background: rgba(6, 8, 18, 0.60) !important;
  backdrop-filter: blur(8px) !important;
  border-bottom: 1px solid rgba(255,255,255,0.08) !important;
}

/* ---- sidebar ---- */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(10,12,28,0.92), rgba(10,12,28,0.78)) !important;
  border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: rgba(245,245,255,0.90) !important; }

/* ---- cards ---- */
.card{
  background: rgba(14,16,34,0.62);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.card h3, .card h2, .card h1 { margin: 0 0 8px 0; }

/* ---- input visibility fix (white-on-white) ---- */
div[data-baseweb="input"] input{
  background: rgba(20,22,40,0.92) !important;
  color: rgba(245,245,255,0.95) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}
div[data-baseweb="textarea"] textarea{
  background: rgba(20,22,40,0.92) !important;
  color: rgba(245,245,255,0.95) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}
div[data-baseweb="input"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder{
  color: rgba(245,245,255,0.55) !important;
}
div[data-baseweb="select"] > div{
  background: rgba(20,22,40,0.92) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}
div[data-baseweb="select"] span{
  color: rgba(245,245,255,0.95) !important;
}

/* ---- slider label ---- */
div[data-testid="stSlider"] label, div[data-testid="stSlider"] span {
  color: rgba(245,245,255,0.90) !important;
}

/* ---- dataframe dark ---- */
div[data-testid="stDataFrame"]{
  background: rgba(12,14,28,0.65) !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}
div[data-testid="stDataFrame"] *{
  color: rgba(245,245,255,0.92) !important;
}

/* ---- table (st.table) ---- */
table{
  color: rgba(245,245,255,0.92) !important;
}
thead tr th{
  background: rgba(18,20,42,0.85) !important;
  color: rgba(245,245,255,0.95) !important;
}
tbody tr td{
  background: rgba(10,12,26,0.55) !important;
}

/* ---- small note ---- */
.smallnote{ color: rgba(245,245,255,0.60); font-size: 12px; }
.badge-ok{
  display:inline-block; padding:4px 10px; border-radius:999px;
  background: rgba(40,190,120,0.18); border:1px solid rgba(40,190,120,0.35);
  color: rgba(190,255,220,0.95);
}
.badge-warn{
  display:inline-block; padding:4px 10px; border-radius:999px;
  background: rgba(220,160,40,0.16); border:1px solid rgba(220,160,40,0.30);
  color: rgba(255,235,190,0.95);
}
</style>
"""
st.markdown(SPACE_CSS, unsafe_allow_html=True)


# =========================
# 2) Utilities (normalize / detect)
# =========================
def norm_col(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("　", " ").lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def safe_int_from_any(s: str) -> Optional[int]:
    if s is None:
        return None
    m = re.search(r"(\d{1,3})", str(s))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def vow_id_from_col(colname: str) -> str:
    """VOW列名から VOW_01.. を安全に作る（None.groupで落ちない）"""
    s = norm_col(colname)
    m = re.search(r"(\d{1,2})", s)
    if m:
        return f"VOW_{int(m.group(1)):02d}"
    # 数字拾えない列はそのまま（落とさない）
    return s


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 1e-9)
    z = (x - np.max(x)) / t
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


# =========================
# 3) Keyword extraction (simple, no external deps)
# =========================
JA_STOP = set([
    "する","いる","ある","なる","もの","こと","これ","それ","ため","よう","さん","てる","です","ます",
    "そして","しかし","なので","から","まで","また","とか","など","でも","ので","なら","もし",
    "私","あなた","自分","今回","状況","感じ","思う","考える","出来る","できる","できない",
])

def simple_keywords(text: str, top_k: int = 8, ngram_max: int = 3) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # ざっくり分割（日本語でも動く程度）
    raw = re.split(r"[ \n\r\t,、。．・/|:;!?！？（）()「」『』【】\[\]<>]+", text)
    toks = []
    for w in raw:
        w = w.strip()
        if not w:
            continue
        if len(w) <= 1:
            continue
        if w in JA_STOP:
            continue
        toks.append(w)

    if not toks:
        # 漢字/英数の連続を拾う救済
        raw2 = re.findall(r"[一-龥]{2,}|[a-zA-Z]{2,}|\d{2,}", text)
        toks = [w for w in raw2 if w not in JA_STOP]

    # n-gram（簡易）
    counts: Dict[str, float] = {}
    for n in range(1, ngram_max + 1):
        for i in range(0, len(toks) - n + 1):
            g = "".join(toks[i:i+n]) if n > 1 else toks[i]
            if len(g) <= 1:
                continue
            # 長いもの少し優遇
            counts[g] = counts.get(g, 0.0) + 1.0 + 0.15 * (len(g) - 2)

    # 同じ語の包含（長い語を優先）
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    picked: List[str] = []
    for w, _ in items:
        if any((w in p) or (p in w) for p in picked):
            # すでに似たのがある場合はスキップ（重複回避）
            continue
        picked.append(w)
        if len(picked) >= top_k:
            break
    return picked


# =========================
# 4) Load Excel (flexible sheet detection)
# =========================
@dataclass
class PackData:
    sheets: List[str]
    vow_master: pd.DataFrame               # columns: VOW_ID, TITLE
    char_master: pd.DataFrame              # columns: CHAR_ID, NAME (optional)
    char_to_vow: pd.DataFrame              # columns: CHAR_ID, VOW_01..VOW_12 (float)
    stage_master: pd.DataFrame             # columns: STAGE_ID, STAGE_NAME (optional)
    stage_to_vow: Optional[pd.DataFrame]   # columns: STAGE_ID, VOW_01..VOW_12 (float) or None
    quotes: pd.DataFrame                   # columns: LANG, TEXT, AUTHOR (optional), TEMP (optional)


def find_sheet(xls: pd.ExcelFile, keywords: List[str]) -> Optional[str]:
    """sheet名が違っても拾えるように、キーワード一致で探す"""
    candidates = []
    for sh in xls.sheet_names:
        n = norm_col(sh)
        score = 0
        for k in keywords:
            if k in n:
                score += 1
        if score > 0:
            candidates.append((score, sh))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def read_sheet(xls: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(xls, sheet_name=sheet)
    # 空列/空行の軽い掃除
    df = df.dropna(how="all")
    df.columns = [str(c) for c in df.columns]
    return df


def ensure_vow_cols(df: pd.DataFrame, min_vow: int = 6) -> List[str]:
    """VOW列候補を柔軟に拾う"""
    cols = list(df.columns)
    normed = [(c, norm_col(c)) for c in cols]

    vow_cols = []
    for c, n in normed:
        # VOW_01 / vow01 / vow-1 / vow 1 / 01 など
        if ("vow" in n) and re.search(r"\d{1,2}", n):
            vow_cols.append(c)
        elif re.fullmatch(r"vow_\d{1,2}", n):
            vow_cols.append(c)

    # それでも足りない場合：VOWの見出しがないが 1..12 が並ぶケース救済
    if len(vow_cols) < min_vow:
        for c, n in normed:
            if re.fullmatch(r"\d{1,2}", n):
                vow_cols.append(c)

    # 重複排除（元の順番維持）
    seen = set()
    vow_cols2 = []
    for c in vow_cols:
        if c in seen:
            continue
        seen.add(c)
        vow_cols2.append(c)
    return vow_cols2


def to_vow_matrix(df: pd.DataFrame, id_col_candidates: List[str]) -> Tuple[pd.DataFrame, str]:
    """CHAR_TO_VOW / STAGE_TO_VOW のような表を (ID + VOW_01..12) に正規化"""
    cols_norm = {norm_col(c): c for c in df.columns}

    id_col = None
    for cand in id_col_candidates:
        if cand in cols_norm:
            id_col = cols_norm[cand]
            break

    if id_col is None:
        # 先頭列をIDとして扱う救済
        id_col = df.columns[0]

    vow_cols = ensure_vow_cols(df, min_vow=1)
    if not vow_cols:
        raise ValueError("VOW列が見つかりません（VOW_01.. のような列をExcelに用意してください）")

    out = pd.DataFrame()
    out["ID"] = df[id_col].astype(str)

    # VOW列を VOW_01.. にリネームしつつ数値化
    vow_map = {}
    for c in vow_cols:
        vow_id = vow_id_from_col(c)   # ← ここが安全（AttributeError回避）
        vow_map[c] = vow_id

    mat = df[vow_cols].copy()
    mat = mat.rename(columns=vow_map)

    # 数値化
    for c in mat.columns:
        mat[c] = pd.to_numeric(mat[c], errors="coerce").fillna(0.0)

    # 01..12を揃える（不足は0）
    for i in range(1, 13):
        k = f"VOW_{i:02d}"
        if k not in mat.columns:
            mat[k] = 0.0

    mat = mat[[f"VOW_{i:02d}" for i in range(1, 13)]]
    out = pd.concat([out, mat], axis=1)
    return out, id_col


def load_pack(uploaded_file) -> PackData:
    xls = pd.ExcelFile(uploaded_file)
    sheets = xls.sheet_names

    # シート探索（名前が違っても拾える）
    sh_vow_master = find_sheet(xls, ["vow_master", "vow", "vows", "vowlist"])
    sh_char_master = find_sheet(xls, ["char_master", "character", "char", "神", "キャラ"])
    sh_char_to_vow = find_sheet(xls, ["char_to_vow", "char2vow", "char_vow", "char_to", "char2", "対応"])
    sh_stage_master = find_sheet(xls, ["stage_master", "stage", "季節", "時間"])
    sh_stage_to_vow = find_sheet(xls, ["stage_to_vow", "stage2vow", "stage_vow", "stage_to", "季節_to", "season"])
    sh_quotes = find_sheet(xls, ["quotes", "quote", "格言", "名言"])

    # 読込（無いものは最小で作る）
    # VOW_MASTER
    if sh_vow_master:
        df_vow = read_sheet(xls, sh_vow_master)
    else:
        # 最低限の雛形
        df_vow = pd.DataFrame({
            "VOW_ID": [f"VOW_{i:02d}" for i in range(1, 13)],
            "TITLE": [f"誓願{i:02d}" for i in range(1, 13)]
        })

    # 列名整備
    cn = {norm_col(c): c for c in df_vow.columns}
    vow_id_col = cn.get("vow_id") or cn.get("vow") or df_vow.columns[0]
    title_col = cn.get("title") or cn.get("name") or (df_vow.columns[1] if len(df_vow.columns) > 1 else df_vow.columns[0])

    vow_master = pd.DataFrame({
        "VOW_ID": df_vow[vow_id_col].astype(str).apply(lambda x: x.strip().upper().replace("-", "_")),
        "TITLE": df_vow[title_col].astype(str).fillna("")
    })
    # VOW_01..12 を埋める救済
    def normalize_vow_id(x: str) -> str:
        x = str(x).strip().upper()
        if x.startswith("VOW") and re.search(r"\d{1,2}", x):
            n = safe_int_from_any(x)
            if n is not None:
                return f"VOW_{n:02d}"
        # 01 だけの可能性
        n = safe_int_from_any(x)
        if n is not None and 1 <= n <= 99:
            return f"VOW_{n:02d}"
        return x

    vow_master["VOW_ID"] = vow_master["VOW_ID"].apply(normalize_vow_id)

    # CHAR_MASTER
    if sh_char_master:
        df_char = read_sheet(xls, sh_char_master)
        cn = {norm_col(c): c for c in df_char.columns}
        char_id_col = cn.get("char_id") or cn.get("char") or cn.get("id") or df_char.columns[0]
        name_col = cn.get("name") or cn.get("title") or cn.get("神") or (df_char.columns[1] if len(df_char.columns) > 1 else df_char.columns[0])
        char_master = pd.DataFrame({
            "CHAR_ID": df_char[char_id_col].astype(str),
            "NAME": df_char[name_col].astype(str)
        })
    else:
        char_master = pd.DataFrame({
            "CHAR_ID": [f"CHAR_{i:02d}" for i in range(1, 13)],
            "NAME": [f"CHAR_{i:02d}" for i in range(1, 13)]
        })

    # CHAR_TO_VOW
    if sh_char_to_vow:
        df_c2v_raw = read_sheet(xls, sh_char_to_vow)
        c2v, _idcol = to_vow_matrix(df_c2v_raw, ["char_id", "char", "id"])
        c2v = c2v.rename(columns={"ID": "CHAR_ID"})
    else:
        # 雛形（ランダム）
        rng = np.random.default_rng(7)
        c2v = pd.DataFrame({"CHAR_ID": [f"CHAR_{i:02d}" for i in range(1, 13)]})
        for i in range(1, 13):
            c2v[f"VOW_{i:02d}"] = rng.uniform(0.0, 1.0, size=len(c2v))

    # STAGE_MASTER
    if sh_stage_master:
        df_stage = read_sheet(xls, sh_stage_master)
        cn = {norm_col(c): c for c in df_stage.columns}
        stage_id_col = cn.get("stage_id") or cn.get("stage") or cn.get("id") or df_stage.columns[0]
        name_col = cn.get("name") or cn.get("title") or cn.get("stage_name") or (df_stage.columns[1] if len(df_stage.columns) > 1 else df_stage.columns[0])
        stage_master = pd.DataFrame({
            "STAGE_ID": df_stage[stage_id_col].astype(str),
            "STAGE_NAME": df_stage[name_col].astype(str)
        })
    else:
        stage_master = pd.DataFrame({"STAGE_ID": ["ST_01"], "STAGE_NAME": ["春×朝"]})

    # STAGE_TO_VOW（あれば季節バイアス）
    stage_to_vow = None
    if sh_stage_to_vow:
        df_s2v_raw = read_sheet(xls, sh_stage_to_vow)
        s2v, _ = to_vow_matrix(df_s2v_raw, ["stage_id", "stage", "id"])
        stage_to_vow = s2v.rename(columns={"ID": "STAGE_ID"})

    # QUOTES
    if sh_quotes:
        df_q = read_sheet(xls, sh_quotes)
        cn = {norm_col(c): c for c in df_q.columns}
        lang_col = cn.get("lang") or cn.get("language") or cn.get("jp") or df_q.columns[0]
        text_col = cn.get("text") or cn.get("quote") or cn.get("格言") or (df_q.columns[1] if len(df_q.columns) > 1 else df_q.columns[0])
        author_col = cn.get("author") or cn.get("by") or cn.get("出典")
        temp_col = cn.get("temp") or cn.get("temperature") or cn.get("t")

        quotes = pd.DataFrame({
            "LANG": df_q[lang_col].astype(str).fillna("ja"),
            "TEXT": df_q[text_col].astype(str).fillna(""),
        })
        if author_col and author_col in df_q.columns:
            quotes["AUTHOR"] = df_q[author_col].astype(str).fillna("")
        else:
            quotes["AUTHOR"] = ""
        if temp_col and temp_col in df_q.columns:
            quotes["TEMP"] = pd.to_numeric(df_q[temp_col], errors="coerce").fillna(1.0)
        else:
            quotes["TEMP"] = 1.0
    else:
        quotes = pd.DataFrame({
            "LANG": ["ja", "ja", "ja"],
            "TEXT": [
                "困難な瞬間にこそ、真の性格が現れる。",
                "幸福とは、自分自身を探す旅の中で見つけるものだ。",
                "努力は才能を超える。"
            ],
            "AUTHOR": ["モンテパート", "リリアン", ""],
            "TEMP": [1.0, 1.2, 0.9]
        })

    return PackData(
        sheets=sheets,
        vow_master=vow_master,
        char_master=char_master,
        char_to_vow=c2v,
        stage_master=stage_master,
        stage_to_vow=stage_to_vow,
        quotes=quotes,
    )


# =========================
# 5) QUBO + Simulated Annealing (one-hot)
# =========================
def build_energy_vector(
    pack: PackData,
    vow_vec: np.ndarray,         # (12,)
    stage_id: Optional[str] = None,
    beta_stage: float = 0.35,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    各キャラの「選ばれやすさ」をエネルギーとして作る（低いほど選ばれやすい）
    energy_i = - dot(CHAR_TO_VOW[i], vow_vec)  + stage_bias(optional)
    """
    c2v = pack.char_to_vow.copy()
    c2v["CHAR_ID"] = c2v["CHAR_ID"].astype(str)

    # CHAR_IDを master 名称に結合（表示用）
    cm = pack.char_master.copy()
    cm["CHAR_ID"] = cm["CHAR_ID"].astype(str)
    show = c2v.merge(cm, on="CHAR_ID", how="left")
    show["NAME"] = show["NAME"].fillna(show["CHAR_ID"])

    M = show[[f"VOW_{i:02d}" for i in range(1, 13)]].to_numpy(dtype=float)
    score = M @ vow_vec.reshape(-1, 1)  # (n,1)
    score = score.reshape(-1)

    energy = -score

    # stageバイアス（あれば）
    if pack.stage_to_vow is not None and stage_id:
        s2v = pack.stage_to_vow.copy()
        s2v["STAGE_ID"] = s2v["STAGE_ID"].astype(str)
        row = s2v[s2v["STAGE_ID"] == str(stage_id)]
        if len(row) > 0:
            b = row[[f"VOW_{i:02d}" for i in range(1, 13)]].to_numpy(dtype=float).reshape(-1)
            # stageの方向に近いほど少し下げる（選ばれやすく）
            stage_score = M @ b.reshape(-1, 1)
            stage_score = stage_score.reshape(-1)
            energy = energy - beta_stage * stage_score

    char_ids = show["CHAR_ID"].tolist()
    names = show["NAME"].tolist()
    return energy, char_ids, names


def sa_one_hot_sample(
    energy: np.ndarray,          # (n,)
    penalty_p: float = 40.0,
    sweeps: int = 420,
    temp: float = 1.2,
    noise: float = 0.08,
    rng: Optional[random.Random] = None,
) -> int:
    """
    one-hot QUBO を「擬似的」にSAでサンプルする
    - 状態xは one-hot を基本に、遷移で別indexへ移動
    - エネルギー: E(i) = energy[i] + small_noise
      ※one-hot制約は遷移をone-hotに限定するため、実質満たされる（PはUI説明用に保持）
    """
    if rng is None:
        rng = random.Random(0)
    n = len(energy)
    # 初期
    cur = rng.randrange(n)
    cur_e = float(energy[cur]) + rng.uniform(-noise, noise)

    # 温度スケジュール
    T0 = max(temp, 1e-6)
    for t in range(1, sweeps + 1):
        # 温度を徐々に下げる（指数）
        T = T0 * (0.995 ** t)
        nxt = rng.randrange(n)
        if nxt == cur:
            continue
        nxt_e = float(energy[nxt]) + rng.uniform(-noise, noise)
        dE = nxt_e - cur_e

        if dE <= 0:
            cur, cur_e = nxt, nxt_e
        else:
            p = math.exp(-dE / max(T, 1e-9))
            if rng.random() < p:
                cur, cur_e = nxt, nxt_e
    return cur


def observe_distribution(
    energy: np.ndarray,
    samples: int = 300,
    penalty_p: float = 40.0,
    sweeps: int = 420,
    temp: float = 1.2,
    noise: float = 0.08,
    seed: int = 7,
) -> np.ndarray:
    rng = random.Random(seed)
    n = len(energy)
    counts = np.zeros(n, dtype=int)
    for _ in range(samples):
        idx = sa_one_hot_sample(
            energy=energy,
            penalty_p=penalty_p,
            sweeps=sweeps,
            temp=temp,
            noise=noise,
            rng=rng
        )
        counts[idx] += 1
    return counts


# =========================
# 6) Character image resolver (match your GitHub files)
# =========================
def find_character_image(char_id: str, base_dir: str) -> Optional[str]:
    """
    あなたのrepo: assets/images/characters/CHAR_p1.png ... CHAR_p12.png
    なので:
      CHAR_01 -> CHAR_p1.png
      CHAR_10 -> CHAR_p10.png
    も探す。さらに CHAR_01.png 等も一応探す。
    """
    if not char_id:
        return None
    cid = str(char_id).strip()

    # CHAR_01 -> 1
    n = safe_int_from_any(cid)
    patterns = []
    if n is not None:
        patterns += [
            f"CHAR_p{n}.png",
            f"CHAR_p{n:02d}.png",
            f"CHAR_{n:02d}.png",
            f"CHAR_{n}.png",
            f"char_p{n}.png",
        ]
    patterns += [
        f"{cid}.png",
        f"{cid.lower()}.png",
        f"{cid.upper()}.png",
    ]

    # base_dir が相対/絶対どちらでもOK
    for fn in patterns:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            return p
    return None


# =========================
# 7) Sidebar controls
# =========================
st.sidebar.markdown("## 📁 データ")

uploaded = st.sidebar.file_uploader("統合Excel（pack）", type=["xlsx"])

img_dir = st.sidebar.text_input(
    "画像フォルダ（相対/絶対）",
    value="./assets/images/characters",
    help="GitHubの構成が assets/images/characters の場合はこのままでOKです。"
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🧠 QUBO設定（one-hot）")
penalty_p = st.sidebar.slider("one-hot ペナルティ P", 1.0, 200.0, 40.0, 1.0)
samples_n = st.sidebar.slider("サンプル数（観測分布）", 50, 1000, 300, 10)
sa_sweeps = st.sidebar.slider("SA sweeps（揺らぎ）", 50, 2000, 420, 10)
sa_temp = st.sidebar.slider("SA温度（大揺らぎ）", 0.1, 5.0, 1.2, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🌀 揺らぎ（観測のブレ）")
beta_stage = st.sidebar.slider("β（Stage寄与）", 0.0, 2.5, 0.35, 0.05)
noise = st.sidebar.slider("微小ノイズ（エネルギーに加える）", 0.0, 0.5, 0.08, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("## ✍️ テキスト＋誓願（自動ベクトル化）")
ngram_max = st.sidebar.slider("n-gram（簡易）", 1, 5, 3, 1)
mix_alpha = st.sidebar.slider("mix比率 α（1=スライダー寄り / 0=テキスト寄り）", 0.0, 1.0, 0.55, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🧾 QUOTES神託（温度付きで選択）")
lang = st.sidebar.selectbox("LANG", ["ja", "en"], index=0)
quote_temp = st.sidebar.slider("格言温度（高=ランダム / 低=上位固定）", 0.2, 3.0, 1.2, 0.05)


# =========================
# 8) Load pack / fallback
# =========================
if uploaded is None:
    st.markdown(
        "<div class='card'><h2>🔮 Q-Quest 量子神託（QUBO / STAGE×QUOTES）</h2>"
        "<div class='smallnote'>左のサイドバーから統合Excel（pack）をアップロードしてください。</div></div>",
        unsafe_allow_html=True
    )
    st.stop()

try:
    pack = load_pack(uploaded)
    st.markdown(f"<span class='badge-ok'>Excel読込OK（sheets: {len(pack.sheets)}）</span>", unsafe_allow_html=True)
except Exception as e:
    st.error(f"Excel読み込みでエラー: {e}")
    st.stop()


# =========================
# 9) Prepare masters
# =========================
# VOW順を 01..12 に揃え、タイトル辞書
vow_title = {row["VOW_ID"]: str(row["TITLE"]) for _, row in pack.vow_master.iterrows()}
vow_ids = [f"VOW_{i:02d}" for i in range(1, 13)]
for vid in vow_ids:
    if vid not in vow_title:
        vow_title[vid] = vid

# Stage selector
stages = pack.stage_master.copy()
stages["STAGE_ID"] = stages["STAGE_ID"].astype(str)
stages["STAGE_NAME"] = stages.get("STAGE_NAME", stages["STAGE_ID"]).astype(str)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🍁 季節×時間（Stage）")
auto_stage = st.sidebar.toggle("現在時刻から自動推定（簡易）", value=False)

def guess_stage_id() -> str:
    # “簡易”：最初の行にする（本格推定はExcel側でstage_to_vowを使う想定）
    return str(stages["STAGE_ID"].iloc[0])

if auto_stage:
    stage_id = guess_stage_id()
else:
    stage_options = [f"{r.STAGE_ID}｜{r.STAGE_NAME}" for r in stages.itertuples()]
    pick = st.sidebar.selectbox("STAGE_ID（手動上書き可）", stage_options, index=0)
    stage_id = pick.split("｜", 1)[0].strip()

st.sidebar.markdown(
    "<div class='smallnote'>STAGE_IDは「季節×時間の状態」を表すキーです。Excelに STAGE_TO_VOW があれば、"
    "そのStageが“どの誓願を強めるか”のバイアスとして反映されます。</div>",
    unsafe_allow_html=True
)


# =========================
# 10) Main layout: Step1 / Step3
# =========================
st.markdown(
    "<div class='card'>"
    "<h1>🔮 Q-Quest 量子神託（QUBO / STAGE×QUOTES）</h1>"
    "</div>",
    unsafe_allow_html=True
)

colL, colR = st.columns([1.7, 1.0], gap="large")

with colL:
    st.markdown("<div class='card'><h2>Step 1：誓願入力（スライダー）＋テキスト（自動ベクトル化）</h2>"
                "<div class='smallnote'>スライダーはTITLEを参考表示。テキストからキーワード抽出→簡易推定し、mixします。</div></div>",
                unsafe_allow_html=True)

    text = st.text_area(
        "あなたの状況を一文で（例：疲れていて決断ができない／新しい挑戦が怖い など）",
        value="",
        height=90,
        placeholder="例：迷いを断ちたいが、今は待つべきか？"
    )

    # スライダー（0..5）
    manual = np.zeros(12, dtype=float)
    st.markdown("<div class='card'><h3>誓願スライダー（manual）</h3></div>", unsafe_allow_html=True)

    for i in range(1, 13):
        vid = f"VOW_{i:02d}"
        title = vow_title.get(vid, vid)
        manual[i-1] = st.slider(f"{vid}｜{title}", 0.0, 5.0, 0.0, 0.5)

    # 4) キーワード抽出
    kws = simple_keywords(text, top_k=10, ngram_max=ngram_max)
    with st.expander("4) テキストのキーワード抽出（簡易）", expanded=True):
        if kws:
            st.markdown("**抽出キーワード**")
            st.write(" / ".join(kws))
        else:
            st.markdown("<div class='smallnote'>（テキストが短い/記号のみ等のため、抽出語がありません）</div>", unsafe_allow_html=True)

    # 自動推定（簡易）：キーワード数に応じて少しだけ押し上げる（※本格化はExcel側で辞書を持たせる）
    auto = np.zeros(12, dtype=float)
    if kws:
        # “迷い/焦り/挑戦/静けさ/内省/行動/つながり/小さく”等の雑マップ（無ければ均等）
        hint_map = {
            "迷い": 1, "断つ": 1, "決断": 1,
            "静けさ": 2, "待つ": 2, "焦り": 2,
            "内面": 3, "内省": 3, "深め": 3,
            "行動": 4, "踏み出": 4,
            "つながり": 5, "支え": 5,
            "挑戦": 6, "進む": 6,
            "手放": 7, "焦": 7,
            "小さく": 8, "一歩": 8,
        }
        bump = 1.6 / max(len(kws), 3)
        for w in kws:
            for k, idx in hint_map.items():
                if k in w:
                    auto[idx-1] += bump
        # 上限
        auto = np.clip(auto, 0.0, 5.0)

    # mix
    mix = mix_alpha * manual + (1.0 - mix_alpha) * auto
    mix = np.clip(mix, 0.0, 5.0)

    st.markdown("<div class='card'><h3>誓願ベクトル（manual / auto / mix）</h3></div>", unsafe_allow_html=True)
    df_v = pd.DataFrame({
        "VOW_ID": [f"VOW_{i:02d}" for i in range(1, 13)],
        "TITLE": [vow_title.get(f"VOW_{i:02d}", f"VOW_{i:02d}") for i in range(1, 13)],
        "manual(0-5)": manual,
        "auto(0-5)": auto,
        "mix(0-5)": mix
    })
    st.dataframe(df_v, use_container_width=True, hide_index=True)

    # 簡易チャート（見た目の確認）
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_v["VOW_ID"], y=df_v["manual(0-5)"], mode="lines+markers", name="manual"))
    fig.add_trace(go.Scatter(x=df_v["VOW_ID"], y=df_v["auto(0-5)"], mode="lines+markers", name="auto"))
    fig.add_trace(go.Scatter(x=df_v["VOW_ID"], y=df_v["mix(0-5)"], mode="lines+markers", name="mix"))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(245,245,255,0.92)"),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", range=[0, 5])
    )
    st.plotly_chart(fig, use_container_width=True)


with colR:
    st.markdown("<div class='card'><h2>Step 3：結果（観測された神＋理由＋QUOTES神託）</h2>"
                "<div class='smallnote'>右側に「候補表」「観測結果」「理由（寄与誓願）」「キャラ画像」「格言」を表示します。</div></div>",
                unsafe_allow_html=True)

    # energy作成
    energy, char_ids, char_names = build_energy_vector(
        pack=pack,
        vow_vec=mix.astype(float),
        stage_id=stage_id,
        beta_stage=beta_stage
    )

    # 観測分布
    counts = observe_distribution(
        energy=energy,
        samples=samples_n,
        penalty_p=penalty_p,
        sweeps=sa_sweeps,
        temp=sa_temp,
        noise=noise,
        seed=11
    )
    prob = counts / max(counts.sum(), 1)

    # 候補表（energy低い順）
    order = np.argsort(energy)
    topk = min(10, len(order))
    rows = []
    for rank, idx in enumerate(order[:topk], start=1):
        rows.append({
            "順位": rank,
            "CHAR_ID": char_ids[idx],
            "神": char_names[idx],
            "energy（低いほど選ばれやすい）": float(energy[idx]),
            "確率（samples比）": float(prob[idx]),
        })
    df_top = pd.DataFrame(rows)
    st.dataframe(df_top, use_container_width=True, hide_index=True)

    # 1回観測（単発）
    pick_idx = sa_one_hot_sample(
        energy=energy,
        penalty_p=penalty_p,
        sweeps=sa_sweeps,
        temp=sa_temp,
        noise=noise,
        rng=random.Random(5)
    )
    picked_id = char_ids[pick_idx]
    picked_name = char_names[pick_idx]

    # 寄与誓願（Top）
    # CHAR_TO_VOWの該当行×mix を寄与とする
    c2v = pack.char_to_vow.copy()
    c2v["CHAR_ID"] = c2v["CHAR_ID"].astype(str)
    row = c2v[c2v["CHAR_ID"] == str(picked_id)]
    contrib = np.zeros(12, dtype=float)
    if len(row) > 0:
        v = row[[f"VOW_{i:02d}" for i in range(1, 13)]].to_numpy(dtype=float).reshape(-1)
        contrib = v * mix

    contrib_rows = []
    for i in range(1, 13):
        vid = f"VOW_{i:02d}"
        contrib_rows.append({
            "VOW": vid,
            "TITLE": vow_title.get(vid, vid),
            "mix(v)": float(mix[i-1]),
            "W(char,v)": float(row[f"VOW_{i:02d}"].iloc[0]) if len(row) > 0 else 0.0,
            "寄与(v*w)": float(contrib[i-1]),
        })
    df_contrib = pd.DataFrame(contrib_rows).sort_values("寄与(v*w)", ascending=False).head(6)

    st.markdown(
        f"<div class='card'><h3>🌟 今回“観測”された神：{picked_name}（{picked_id}）</h3>"
        f"<div class='smallnote'>ここは単発の観測（1回抽選）です。上の分布（samples）は“何回も観測したらどうなるか”の目安です。</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # キャラクター画像
    img_path = find_character_image(picked_id, img_dir)
    if img_path and os.path.exists(img_path):
        st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
    else:
        st.markdown(
            f"<div class='card'><span class='badge-warn'>※キャラクター画像が見つかりません</span>"
            f"<div class='smallnote'>探索フォルダ：{img_dir}<br>"
            f"探し方：CHAR_01→CHAR_p1.png / CHAR_p01.png / CHAR_01.png など</div></div>",
            unsafe_allow_html=True
        )

    # 理由（寄与誓願）
    st.markdown("<div class='card'><h3>🧩 寄与した誓願（Top）</h3></div>", unsafe_allow_html=True)
    st.dataframe(df_contrib, use_container_width=True, hide_index=True)

    # “いまの波”文（要約）
    wave = "・".join(df_contrib["TITLE"].astype(str).tolist())
    st.markdown(
        f"<div class='card'>"
        f"<h3>🟩 いまの波（要約）</h3>"
        f"<div>いまの波は <b>{wave}</b> に寄っている。 Stage（{stage_id}）は流れを強める。</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # QUOTES選択
    q = pack.quotes.copy()
    q["LANG"] = q["LANG"].astype(str).str.lower()
    q_pick = q[q["LANG"] == lang].copy()
    if len(q_pick) == 0:
        q_pick = q.copy()

    # スコア：寄与上位の“熱”でランダム寄りに（温度）
    # ここは簡易：テキスト長×温度でsoftmax
    texts = q_pick["TEXT"].astype(str).fillna("").tolist()
    base = np.array([max(len(t), 1) for t in texts], dtype=float)
    p = softmax(base, temperature=quote_temp)
    idx_q = int(np.random.default_rng(3).choice(len(q_pick), p=p))
    qt = q_pick.iloc[idx_q]

    quote_text = str(qt.get("TEXT", "")).strip()
    quote_author = str(qt.get("AUTHOR", "")).strip()

    st.markdown("<div class='card'><h3>📘 QUOTES神託（温度付きで選択）</h3></div>", unsafe_allow_html=True)
    if quote_text:
        if quote_author:
            st.markdown(f"**「{quote_text}」**  \n— {quote_author}")
        else:
            st.markdown(f"**「{quote_text}」**")
    else:
        st.markdown("<div class='smallnote'>（格言データが空です）</div>", unsafe_allow_html=True)


# =========================
# 11) Footer debug (optional)
# =========================
with st.expander("🔧 デバッグ（読込シート名・検出状況）", expanded=False):
    st.write("Sheets:", pack.sheets)
    st.write("VOW_MASTER head:", pack.vow_master.head())
    st.write("CHAR_MASTER head:", pack.char_master.head())
    st.write("CHAR_TO_VOW head:", pack.char_to_vow.head())
    st.write("STAGE_MASTER head:", pack.stage_master.head())
    if pack.stage_to_vow is not None:
        st.write("STAGE_TO_VOW head:", pack.stage_to_vow.head())
    st.write("QUOTES head:", pack.quotes.head())
