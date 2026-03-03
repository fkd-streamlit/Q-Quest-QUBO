# -*- coding: utf-8 -*-
"""
Q-Quest-QUBO｜量子神託（黒背景UI / 誓願テキスト / QUOTES神託 / QUBO one-hot）
- 統合Excel(pack) をアップロードして動作
- 12神 one-hot QUBO + SAサンプリング
- 誓願入力：スライダー + テキスト自動ベクトル化（mix）
- 結果：ランキング表 / 観測神 / 寄与Top / QUOTES神託 / 可視化 / QUBO証拠
"""

import os
import re
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# ============================================================
# Streamlit config（必ず最初）
# ============================================================
st.set_page_config(page_title="Q-Quest-QUBO｜量子神託", layout="wide")


# ============================================================
# Dark UI CSS（app09風）
# ============================================================
def inject_dark_css():
    st.markdown(
        """
<style>
/* 全体背景 */
.stApp {
  background: radial-gradient(1200px 700px at 20% 10%, rgba(40,20,80,0.35), rgba(0,0,0,0.0)),
              radial-gradient(900px 500px at 90% 20%, rgba(0,120,255,0.18), rgba(0,0,0,0.0)),
              linear-gradient(180deg, #060816 0%, #070b18 35%, #050612 100%);
  color: #EAEAF2;
}

/* サイドバー */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(18,20,44,0.98) 0%, rgba(10,12,30,0.98) 100%);
  border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * {
  color: #EAEAF2;
}

/* 見出し */
h1, h2, h3, h4 { color: #F4F4FF; }
small, .stCaption { color: rgba(234,234,242,0.75) !important; }

/* カード */
.q-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

/* 強調ボックス */
.q-hero {
  background: linear-gradient(135deg, rgba(0,180,255,0.18), rgba(155,90,255,0.14));
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 14px 16px;
}

/* ボタン */
.stButton > button {
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.18);
  background: linear-gradient(90deg, rgba(95,80,255,0.95), rgba(160,90,255,0.95));
  color: white;
  font-weight: 700;
}
.stButton > button:hover {
  filter: brightness(1.05);
}

/* テキスト入力 */
textarea, input {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  color: #F4F4FF !important;
  border-radius: 12px !important;
}

/* データフレーム（軽く暗く） */
[data-testid="stDataFrame"] {
  background: rgba(255,255,255,0.03);
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.08);
}

/* 区切り線 */
hr { border-color: rgba(255,255,255,0.10); }
</style>
        """,
        unsafe_allow_html=True,
    )

inject_dark_css()


# ============================================================
# Utilities
# ============================================================
def _safe_str(x) -> str:
    return "" if pd.isna(x) else str(x)

def _safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def normalize01(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    mn, mx = float(np.min(v)), float(np.max(v))
    if mx - mn < 1e-12:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn)

def softmax(x: np.ndarray, t: float = 1.0) -> np.ndarray:
    t = max(1e-9, float(t))
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    z = np.exp(x / t)
    s = np.sum(z)
    return z / s if s > 0 else np.ones_like(z) / len(z)

def vow_key_to_num(v: str) -> int:
    m = re.search(r"VOW_(\d+)", str(v))
    return int(m.group(1)) if m else 10**9

def get_vow_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if re.match(r"VOW_\d+", str(c))]
    cols.sort(key=vow_key_to_num)
    return cols

def ensure_cols(df: pd.DataFrame, required: List[str], sheet: str):
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{sheet} の列が不足: {miss}\n検出列={df.columns.tolist()}")

@st.cache_data(show_spinner=False)
def load_image(path: str) -> Optional[Image.Image]:
    try:
        if not path or not os.path.exists(path):
            return None
        return Image.open(path)
    except Exception:
        return None


# ============================================================
# Text -> vow（軽量 char n-gram）
# ============================================================
from collections import Counter

def char_ngrams(text: str, n=3) -> Counter:
    s = re.sub(r"\s+", "", _safe_str(text))
    if len(s) < n:
        return Counter()
    return Counter(s[i:i+n] for i in range(len(s) - n + 1))

def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0) for k in keys], dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))

def build_vow_texts(vow_dict: pd.DataFrame, vow_ids: List[str]) -> List[str]:
    cols = ["LABEL", "TITLE", "SUBTITLE", "DESCRIPTION_LONG", "UI_HINT"]
    texts = []
    for vid in vow_ids:
        row = vow_dict[vow_dict["VOW_ID"].astype(str) == str(vid)]
        if row.empty:
            texts.append(str(vid))
            continue
        r = row.iloc[0]
        parts = []
        for c in cols:
            if c in vow_dict.columns:
                v = _safe_str(r.get(c, ""))
                if v:
                    parts.append(v)
        texts.append(" ".join(parts) if parts else str(vid))
    return texts

def text_to_vow_vector(user_text: str, vow_texts: List[str], n=3) -> np.ndarray:
    cu = char_ngrams(user_text, n=n)
    v = np.zeros(len(vow_texts), dtype=float)
    for i, t in enumerate(vow_texts):
        cr = char_ngrams(t, n=n)
        v[i] = cosine_counter(cu, cr)
    return normalize01(v)

def extract_keywords_simple(text: str, topk: int = 10) -> List[str]:
    s = re.sub(r"\s+", "", _safe_str(text))
    if not s:
        return []
    grams = []
    for n in [2, 3]:
        if len(s) >= n:
            grams += [s[i:i+n] for i in range(len(s)-n+1)]
    grams = [g for g in grams if re.search(r"[ぁ-んァ-ン一-龥]", g)]
    if not grams:
        return []
    cnt = Counter(grams)
    return [w for w, _ in cnt.most_common(topk)]


# ============================================================
# QUBO core（one-hot）
# ============================================================
def build_qubo_onehot(linear_E: np.ndarray, P: float) -> np.ndarray:
    n = len(linear_E)
    Q = np.zeros((n, n), dtype=float)
    for i in range(n):
        Q[i, i] += float(linear_E[i] - P)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += float(2.0 * P)
    return Q

def energy_qubo(Q: np.ndarray, x: np.ndarray) -> float:
    x = x.astype(float)
    return float(x @ Q @ x)

def onehot_index(x: np.ndarray) -> Optional[int]:
    ones = np.where(x == 1)[0]
    return int(ones[0]) if len(ones) == 1 else None

def sa_sample_qubo(Q: np.ndarray, num_reads=200, sweeps=400, t0=5.0, t1=0.2, seed=0):
    rng = random.Random(seed)
    n = Q.shape[0]
    samples, energies = [], []
    for _ in range(int(num_reads)):
        x = np.array([rng.randint(0, 1) for _ in range(n)], dtype=int)
        E = energy_qubo(Q, x)
        for s in range(int(sweeps)):
            t = t0 + (t1 - t0) * (s / max(1, sweeps - 1))
            i = rng.randrange(n)
            xn = x.copy()
            xn[i] ^= 1
            En = energy_qubo(Q, xn)
            dE = En - E
            if dE <= 0 or rng.random() < math.exp(-dE / max(t, 1e-9)):
                x, E = xn, En
        samples.append(x)
        energies.append(E)
    return np.array(samples, dtype=int), np.array(energies, dtype=float)


# ============================================================
# Pack loader
# ============================================================
@dataclass
class Pack:
    vow_dict: pd.DataFrame
    char_to_vow: pd.DataFrame
    char_master: Optional[pd.DataFrame]
    stage_dict: Optional[pd.DataFrame]
    stage_to_axis: Optional[pd.DataFrame]
    quotes: Optional[pd.DataFrame]

@st.cache_data(show_spinner=False)
def load_pack_excel_bytes(xlsx_bytes: bytes) -> Pack:
    xls = pd.ExcelFile(xlsx_bytes)

    if "VOW_DICT" not in xls.sheet_names or "CHAR_TO_VOW" not in xls.sheet_names:
        raise ValueError(f"必須シート不足。検出={xls.sheet_names}")

    vow_dict = pd.read_excel(xls, "VOW_DICT")
    char_to_vow = pd.read_excel(xls, "CHAR_TO_VOW")
    ensure_cols(vow_dict, ["VOW_ID", "TITLE"], "VOW_DICT")
    ensure_cols(char_to_vow, ["CHAR_ID", "公式キャラ名"], "CHAR_TO_VOW")

    if "IMAGE_FILE" not in char_to_vow.columns:
        char_to_vow["IMAGE_FILE"] = ""

    char_master = pd.read_excel(xls, "CHAR_MASTER") if "CHAR_MASTER" in xls.sheet_names else None
    stage_dict = pd.read_excel(xls, "STAGE_DICT") if "STAGE_DICT" in xls.sheet_names else None
    stage_to_axis = pd.read_excel(xls, "STAGE_TO_AXIS") if "STAGE_TO_AXIS" in xls.sheet_names else None
    quotes = pd.read_excel(xls, "QUOTES") if "QUOTES" in xls.sheet_names else None

    # CHAR_MASTERの軸列が無い場合は使わない
    if char_master is not None:
        need = ["CHAR_ID", "AXIS_SEI", "AXIS_RYU", "AXIS_MA", "AXIS_MAKOTO"]
        if any(c not in char_master.columns for c in need):
            char_master = None

    return Pack(vow_dict, char_to_vow, char_master, stage_dict, stage_to_axis, quotes)


# ============================================================
# QUOTES selection
# ============================================================
def pick_quotes(
    df_quotes: pd.DataFrame,
    observed_char_id: str,
    top_vow_ids: List[str],
    v_mix_map: Dict[str, float],
    keywords: List[str],
    lang: str,
    temperature: float,
    k: int = 3,
    topn: int = 60,
) -> List[Dict]:
    if df_quotes is None or df_quotes.empty:
        return []

    if "QUOTE" not in df_quotes.columns:
        return []

    cand = df_quotes.copy()

    # LANGフィルタ
    if lang and "LANG" in cand.columns:
        sub = cand[cand["LANG"].fillna("").astype(str).str.lower() == lang.lower()].copy()
        if not sub.empty:
            cand = sub

    def score_row(r: pd.Series) -> float:
        s = 0.0
        q = _safe_str(r.get("QUOTE", ""))
        cid = _safe_str(r.get("CHAR_ID", ""))
        vid = _safe_str(r.get("VOW_ID", ""))

        if cid and cid == str(observed_char_id):
            s += 2.5

        if vid:
            s += 1.2 * float(v_mix_map.get(vid, 0.0))
            if vid in top_vow_ids:
                s += 0.6

        for kw in keywords:
            if kw and kw in q:
                s += 0.25

        return float(s)

    cand["SCORE"] = [score_row(r) for _, r in cand.iterrows()]
    cand = cand.sort_values("SCORE", ascending=False).head(int(topn)).copy()
    if cand.empty:
        return []

    probs = softmax(cand["SCORE"].to_numpy(float), t=max(1e-6, float(temperature)))
    rng = np.random.default_rng()
    n_pick = min(int(k), len(cand))
    idxs = rng.choice(len(cand), size=n_pick, replace=False, p=probs)

    out = []
    for r in cand.iloc[idxs].to_dict("records"):
        out.append({
            "QUOTE": _safe_str(r.get("QUOTE", "")).strip(),
            "SOURCE": _safe_str(r.get("SOURCE", "")).strip(),
            "CHAR_ID": _safe_str(r.get("CHAR_ID", "")),
            "VOW_ID": _safe_str(r.get("VOW_ID", "")),
            "SCORE": float(r.get("SCORE", 0.0)),
        })
    return out


# ============================================================
# UI header
# ============================================================
st.markdown(
    """
<div class="q-card">
  <h1 style="margin:0; font-size: 34px;">🔮 Q-Quest 量子神託 <span style="opacity:0.75; font-size:20px;">(QUBO one-hot / STAGE×QUOTES)</span></h1>
  <div style="opacity:0.8; margin-top:6px;">Step1: 誓願入力（スライダー）＋テキスト（自動ベクトル化） / Step3: 結果（観測＋理由＋神託）</div>
</div>
""",
    unsafe_allow_html=True
)
st.write("")


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("📁 データ")
    pack_file = st.file_uploader("統合Excel（pack）", type=["xlsx"])

    st.header("🖼️ 画像フォルダ")
    img_dir = st.text_input("画像フォルダ（相対/絶対）", value="./assets/images/characters")

    st.divider()
    st.header("🕰️ 季節×時間（Stage）")
    auto_now = st.checkbox("現在時刻から自動推定", value=True)
    if auto_now:
        from datetime import datetime
        now = datetime.now()
        month = now.month
        hour = now.hour
    else:
        month = st.slider("月", 1, 12, 2)
        hour = st.slider("時刻（0-23）", 0, 23, 21)

    st.divider()
    st.header("🎲 揺らぎ（観測のブレ）")
    # βは「最小エネルギー寄り」っぽい演出用（ランキング表示のsoftmax温度に使う）
    beta = st.slider("β（大→最小エネルギー寄り / 小→多様）", 0.5, 6.0, 2.2, 0.1)
    noise_sigma = st.slider("微小ノイズε（エネルギーに加える）", 0.0, 0.2, 0.08, 0.01)

    st.divider()
    st.header("⚙️ QUBO設定（one-hot）")
    penalty = st.slider("one-hot ペナルティ P", 1.0, 200.0, 40.0, 1.0)
    num_reads = st.slider("サンプル数（観測分布）", 50, 800, 300, 10)
    sweeps = st.slider("SA sweeps", 50, 1200, 420, 10)
    temperature = st.slider("SA温度（大→揺らぐ）", 0.1, 5.0, 1.2, 0.1)

    st.divider()
    st.header("🧠 テキスト→誓願（自動ベクトル化）")
    ngram_n = st.selectbox("n-gram", [2, 3], index=1)
    alpha_text = st.slider("mix比率 α（1=テキスト寄り / 0=スライダーのみ）", 0.0, 1.0, 0.55, 0.05)

    st.divider()
    st.header("🗣️ QUOTES神託（温度付き選択）")
    quote_lang = st.selectbox("LANG", ["ja", "en", ""], index=0, help="空=全言語")
    quote_temp = st.slider("格言温度（高→ランダム / 低→上位固定）", 0.2, 3.0, 1.2, 0.1)

    run_btn = st.button("🧪 観測する（QUBOから抽出）", use_container_width=True)

if pack_file is None:
    st.info("左サイドバーから **統合Excel（pack）** をアップロードしてください。")
    st.stop()

# ============================================================
# Load pack
# ============================================================
try:
    pack = load_pack_excel_bytes(pack_file.getvalue())
except Exception as e:
    st.error(f"統合Excelの解析に失敗: {e}")
    st.stop()

df_vow = pack.vow_dict.copy()
df_ctv = pack.char_to_vow.copy()

vow_cols = get_vow_cols(df_ctv)
ensure_cols(df_ctv, ["CHAR_ID", "公式キャラ名", "IMAGE_FILE"], "CHAR_TO_VOW")

char_ids = df_ctv["CHAR_ID"].astype(str).tolist()
char_names = df_ctv["公式キャラ名"].astype(str).tolist()
img_files = df_ctv["IMAGE_FILE"].astype(str).tolist()

W_char_vow = df_ctv[vow_cols].fillna(0).astype(float).to_numpy()

# VOWの表示ラベル（TITLE + LRがあれば付与）
vow_title = {}
vow_left = {}
vow_right = {}
for _, r in df_vow.iterrows():
    vid = _safe_str(r.get("VOW_ID", ""))
    vow_title[vid] = _safe_str(r.get("TITLE", vid))
    for cand in ["LEFT_LABEL", "LEFT", "LEFT_TEXT"]:
        if cand in df_vow.columns:
            vow_left[vid] = _safe_str(r.get(cand, ""))
            break
    for cand in ["RIGHT_LABEL", "RIGHT", "RIGHT_TEXT"]:
        if cand in df_vow.columns:
            vow_right[vid] = _safe_str(r.get(cand, ""))
            break


# ============================================================
# Layout (Step1 / Step3)
# ============================================================
colL, colR = st.columns([1.15, 1.0], gap="large")

with colL:
    st.markdown('<div class="q-card"><h3 style="margin:0;">Step 1：誓願入力（スライダー）＋テキスト（自動ベクトル化）</h3></div>', unsafe_allow_html=True)
    st.write("")

    # ✅ ユーザ誓願入力窓（復活）
    user_text = st.text_area(
        "あなたの状況を一文で（例：疲れていて決断ができない / 新しい挑戦が怖い など）",
        height=90,
        placeholder="例：迷いを断ち切って、新しい一歩を踏み出したい。焦らず待つ勇気もほしい。",
    )
    st.caption("スライダー入力はTITLEを常時表示し、テキストからの自動推定と mix します。")

    # sliders
    v_user = np.zeros(len(vow_cols), dtype=float)
    for i, vid in enumerate(vow_cols):
        title = vow_title.get(vid, vid)
        l = vow_left.get(vid, "")
        r = vow_right.get(vid, "")
        if l or r:
            label = f"{vid}｜{title}  ←{l}｜{r}→"
        else:
            label = f"{vid}｜{title}"
        v_user[i] = st.slider(label, 0.0, 5.0, 0.0, 0.5, key=f"sl_{vid}")

    v_user01 = v_user / 5.0

    # auto from text
    vow_texts = build_vow_texts(df_vow, vow_cols)
    v_text01 = text_to_vow_vector(user_text, vow_texts, n=int(ngram_n))

    # mix
    v_mix01 = (1.0 - alpha_text) * v_user01 + alpha_text * v_text01
    v_mix01 = np.clip(v_mix01, 0.0, 1.0)

    with st.expander("🔎 誓願ベクトル（manual / auto / mix）"):
        st.dataframe(
            pd.DataFrame({
                "VOW": vow_cols,
                "TITLE": [vow_title.get(v, v) for v in vow_cols],
                "manual": np.round(v_user01, 3),
                "auto": np.round(v_text01, 3),
                "mix": np.round(v_mix01, 3),
            }),
            use_container_width=True,
            hide_index=True
        )

    st.write("")
    st.markdown('<div class="q-card"><h3 style="margin:0;">12神キャラクター（ギャラリー）</h3><div style="opacity:0.8;">※画像が repo にある場合、全員表示します</div></div>', unsafe_allow_html=True)
    st.write("")

    gcols = st.columns(4)
    for i, name in enumerate(char_names):
        with gcols[i % 4]:
            p = os.path.join(img_dir, img_files[i]) if img_files[i] else ""
            im = load_image(p)
            if im is not None:
                st.image(im, use_container_width=True)
            st.caption(name)

with colR:
    st.markdown('<div class="q-card"><h3 style="margin:0;">Step 3：結果（観測された神＋理由＋QUOTES神託）</h3></div>', unsafe_allow_html=True)
    st.write("")

# ============================================================
# Build energies + QUBO (always computed)
# ============================================================
score_vow = W_char_vow @ v_mix01

# 演出用ノイズ（エネルギー側に加える）
rng_np = np.random.default_rng(0)
score = score_vow + rng_np.normal(0, float(noise_sigma) * 0.02, size=score_vow.shape)

linear_E = -score  # minimize

# penalty safe
minP = float(np.max(np.abs(linear_E)) * 3.0 + 1.0)
P = max(float(penalty), minP)

Q = build_qubo_onehot(linear_E, P)

# ranking (softmax with beta)
# beta大 → 低エネルギーが選ばれやすい（温度=1/beta的）
prob = softmax(-linear_E * float(beta), t=1.0)
rank_idx = np.argsort(linear_E)[:12]

rank_df = pd.DataFrame({
    "順位": np.arange(1, len(rank_idx) + 1),
    "CHAR_ID": [char_ids[i] for i in rank_idx],
    "神": [char_names[i] for i in rank_idx],
    "energy（低いほど選ばれやすい）": np.round(linear_E[rank_idx], 4),
    "確率（softmax）": np.round(prob[rank_idx], 4),
})

# ============================================================
# Observe (QUBO sampling) when button pressed
# ============================================================
if run_btn:
    samples, Es = sa_sample_qubo(
        Q, num_reads=int(num_reads), sweeps=int(sweeps),
        t0=float(5.0 * temperature), t1=float(0.2 * temperature),
        seed=random.randint(0, 10**9)
    )

    idxs = [onehot_index(x) for x in samples]
    ok = [k for k in idxs if k is not None]
    counts = np.zeros(len(char_names), dtype=int)
    for k in ok:
        counts[k] += 1

    violation = 1.0 - (len(ok) / max(1, len(samples)))

    best_k = None
    best_E = None
    for x, e in zip(samples, Es):
        k = onehot_index(x)
        if k is None:
            continue
        if best_E is None or e < best_E:
            best_E = float(e)
            best_k = int(k)

    if best_k is None:
        best_k = int(np.argmin(linear_E))

    st.session_state["counts"] = counts
    st.session_state["best_k"] = best_k
    st.session_state["violation"] = float(violation)

# ============================================================
# Render result panel
# ============================================================
with colR:
    st.dataframe(rank_df.head(3), use_container_width=True, hide_index=True)

    if "best_k" not in st.session_state:
        st.info("左の入力を行い、サイドバーの **「観測する（QUBOから抽出）」** を押してください。")
    else:
        best_k = int(st.session_state["best_k"])
        counts = st.session_state["counts"]
        violation = float(st.session_state["violation"])

        st.markdown(
            f"""
<div class="q-hero">
  <h2 style="margin:0;">🌟 今回 “観測” された神：{char_names[best_k]} <span style="opacity:0.85; font-size:18px;">({char_ids[best_k]})</span></h2>
  <div style="opacity:0.8; margin-top:6px;">ここは “単発の観測（1回抽選）” です。下の観測分布（サンプル）は「同条件で何回も観測したらどう出るか」のヒストグラムです。</div>
</div>
""",
            unsafe_allow_html=True
        )
        st.write("")

        p = os.path.join(img_dir, img_files[best_k]) if img_files[best_k] else ""
        im = load_image(p)
        if im is not None:
            st.image(im, use_container_width=True)
        else:
            st.warning(f"画像が見つかりません: {p}")

        # 寄与Top
        char_w = W_char_vow[best_k, :]
        contrib = char_w * v_mix01
        top_idx = np.argsort(contrib)[::-1][:7]
        top_vow_ids = [vow_cols[i] for i in top_idx]
        v_mix_map = {vow_cols[i]: float(v_mix01[i]) for i in range(len(vow_cols))}

        st.markdown('<div class="q-card"><h3 style="margin:0;">🧩 寄与した誓願（Top）</h3></div>', unsafe_allow_html=True)
        st.write("")
        st.dataframe(
            pd.DataFrame({
                "VOW": top_vow_ids,
                "TITLE": [vow_title.get(v, v) for v in top_vow_ids],
                "mix(v)": [round(v_mix_map.get(v, 0.0) * 5.0, 3) for v in top_vow_ids],  # 0..5の見た目
                "W(char,v)": [round(float(char_w[vow_cols.index(v)]), 3) for v in top_vow_ids],
                "寄与(v*w)": [round(float(contrib[vow_cols.index(v)]), 3) for v in top_vow_ids],
            }),
            use_container_width=True, hide_index=True
        )

        # QUOTES神託（復活）
        st.write("")
        st.markdown('<div class="q-card"><h3 style="margin:0;">🗣️ QUOTES神託（温度付きで選択）</h3></div>', unsafe_allow_html=True)
        st.write("")

        if pack.quotes is None:
            st.info("Excelに **QUOTES** シートがありません。追加すると神託が出ます。")
        else:
            keywords = extract_keywords_simple(user_text, topk=10)
            quotes = pick_quotes(
                df_quotes=pack.quotes,
                observed_char_id=char_ids[best_k],
                top_vow_ids=top_vow_ids,
                v_mix_map=v_mix_map,
                keywords=keywords,
                lang=quote_lang,
                temperature=quote_temp,
                k=3,
                topn=60
            )
            if not quotes:
                st.warning("QUOTESから神託が選べませんでした（列不足 / LANGフィルタ / データ）")
                if "LANG" in pack.quotes.columns:
                    st.caption("ヒント：LANGが一致しない場合、サイドバーで LANG を空（全言語）にしてください。")
            else:
                for i, q in enumerate(quotes, start=1):
                    qt = q.get("QUOTE", "")
                    src = q.get("SOURCE", "")
                    st.markdown(f'<div class="q-card"><b>神託 {i}</b><br><div style="font-size:18px; margin-top:6px;">「{qt}」</div><div style="opacity:0.7; margin-top:8px;">— {src}</div></div>', unsafe_allow_html=True)
                    st.write("")

# ============================================================
# Visualization section（app09風）
# ============================================================
st.write("")
st.markdown('<div class="q-card"><h2 style="margin:0;">📊 可視化：テキストの影響・観測分布・エネルギー地形</h2></div>', unsafe_allow_html=True)
st.write("")

v_df = pd.DataFrame({
    "VOW": vow_cols,
    "auto": v_text01 * 5.0,
    "manual": v_user01 * 5.0,
    "mix": v_mix01 * 5.0
}).set_index("VOW")

c1, c2 = st.columns([1.2, 1.0], gap="large")
with c1:
    st.markdown("### 1) テキスト→誓願 自動推定の影響（auto vs manual vs mix）")
    st.line_chart(v_df)
with c2:
    st.markdown("### 2) エネルギー地形（全候補）")
    e_df = pd.DataFrame({"神": char_names, "energy": linear_E}).set_index("神")
    st.bar_chart(e_df)

st.write("")
st.markdown("### 3) 観測分布（サンプル）")
if "counts" in st.session_state:
    hist = pd.DataFrame({"神": char_names, "count": st.session_state["counts"]}).set_index("神")
    st.bar_chart(hist)
else:
    st.caption("まだ観測していません。サイドバーの **観測** を押すと分布が出ます。")

st.write("")
st.markdown("### 4) テキストのキーワード抽出（簡易）")
kws = extract_keywords_simple(user_text, topk=12)
st.write(" / ".join(kws) if kws else "（入力テキストが短い/空のため、抽出できません）")

st.write("")
st.markdown('<div class="q-card"><h3 style="margin:0;">🧠 QUBO 証拠</h3></div>', unsafe_allow_html=True)
st.write("")
if "best_k" in st.session_state:
    best_k = int(st.session_state["best_k"])
    x_best = [1 if i == best_k else 0 for i in range(len(char_names))]
    violation = float(st.session_state.get("violation", 0.0))
    st.code(
        "E(x) = Σ_i (E_i * x_i) + P(Σ_i x_i − 1)^2\n"
        f"P = {P:.2f}\n"
        f"x = {x_best}\n"
        f"違反率 = {violation*100:.1f}%\n"
        f"E(x) = {energy_qubo(Q, np.array(x_best, dtype=int)):.6f}",
        language="text"
    )
else:
    st.caption("観測後にQUBO証拠（x / P / 違反率）が表示されます。")
