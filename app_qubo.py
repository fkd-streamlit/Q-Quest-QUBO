# -*- coding: utf-8 -*-
"""
Q-Quest 量子神託（QUBO one-hot 実装 + app09系UI）
- 統合Excel(pack)で完結（VOW/CHAR/STAGE/QUOTES）
- 神(12)を one-hot QUBO で選択
- E(x) = Σ E_i x_i + P(Σx-1)^2
- Simulated Annealing でサンプリング
- 12神ギャラリー表示 + 観測神 + 観測分布 + QUOTES神託
"""

import os
import re
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ============================================================
# Streamlit config（必ず最初）
# ============================================================
st.set_page_config(page_title="Q-Quest-QUBO｜量子神託（one-hot QUBO）", layout="wide")

APP_TITLE = "🔮 Q-Quest 量子神託（QUBO one-hot 実装 / app09 UI寄せ）"

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

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(1e-9, float(temperature))
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    z = np.exp(x / t)
    s = np.sum(z)
    return z / s if s > 0 else np.ones_like(z) / len(z)

def vow_key_to_num(v: str) -> int:
    m = re.search(r"VOW_(\d+)", str(v))
    return int(m.group(1)) if m else -1

def get_vow_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if re.match(r"VOW_\d+", str(c))]
    cols.sort(key=lambda x: vow_key_to_num(x))
    return cols

def ensure_cols(df: pd.DataFrame, required: List[str], sheet_name: str):
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{sheet_name} の列が不足: {miss}\n検出列={df.columns.tolist()}")

@st.cache_data(show_spinner=False)
def load_image(path: str) -> Optional[Image.Image]:
    try:
        if not path or not os.path.exists(path):
            return None
        return Image.open(path)
    except Exception:
        return None

# ============================================================
# Stage helpers
# ============================================================
def season_from_month(month: int) -> str:
    if month in [3, 4, 5]:
        return "SPRING"
    if month in [6, 7, 8]:
        return "SUMMER"
    if month in [9, 10, 11]:
        return "AUTUMN"
    return "WINTER"

def time_slot_from_hour(hour: int) -> str:
    if 5 <= hour <= 10:
        return "MORNING"
    if 11 <= hour <= 16:
        return "DAY"
    if 17 <= hour <= 20:
        return "EVENING"
    return "NIGHT"

def build_stage_id(season: str, time_slot: str) -> str:
    return f"{season}_{time_slot}"

def get_stage_axis_weights(stage_to_axis: pd.DataFrame, stage_id: str) -> np.ndarray:
    row = stage_to_axis[stage_to_axis["STAGE_ID"].astype(str) == str(stage_id)]
    if row.empty:
        return np.zeros(4, dtype=float)
    r = row.iloc[0]
    return np.array([
        _safe_float(r.get("AXIS_SEI", 0.0)),
        _safe_float(r.get("AXIS_RYU", 0.0)),
        _safe_float(r.get("AXIS_MA", 0.0)),
        _safe_float(r.get("AXIS_MAKOTO", 0.0)),
    ], dtype=float)

# ============================================================
# Text -> vow (軽量な char n-gram コサイン)
# ============================================================
def _char_ngrams(text: str, n=3):
    s = _safe_str(text)
    s = re.sub(r"\s+", "", s)
    if len(s) < n:
        return {}
    from collections import Counter
    return Counter(s[i:i+n] for i in range(len(s) - n + 1))

def _cosine_from_counters(a, b) -> float:
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

def build_vow_texts(vow_dict: pd.DataFrame, vow_ids: List[str]) -> List[str]:
    # ある列だけ連結して参照文書にする
    cols = ["LABEL", "TITLE", "SUBTITLE", "DESCRIPTION_LONG", "UI_HINT", "TRAIT_FROM_FILE"]
    texts = []
    for vid in vow_ids:
        row = vow_dict[vow_dict["VOW_ID"].astype(str) == str(vid)]
        if row.empty:
            texts.append(str(vid))
            continue
        r = row.iloc[0]
        parts = []
        for c in cols:
            if c in vow_dict.columns and pd.notna(r.get(c)):
                parts.append(str(r.get(c)))
        texts.append(" ".join(parts) if parts else str(vid))
    return texts

def text_to_vow_vector(user_text: str, vow_texts: List[str], n=3) -> np.ndarray:
    c_user = _char_ngrams(user_text, n=n)
    v = np.zeros(len(vow_texts), dtype=float)
    for i, t in enumerate(vow_texts):
        c_ref = _char_ngrams(t, n=n)
        v[i] = _cosine_from_counters(c_user, c_ref)
    v = normalize01(v)  # 0..1
    return v

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
    from collections import Counter
    cnt = Counter(grams)
    return [w for w, _ in cnt.most_common(topk)]

# ============================================================
# QUBO core（one-hot）
# ============================================================
def build_qubo_onehot(linear_E: np.ndarray, penalty_P: float) -> np.ndarray:
    """
    QUBO:
      E(x) = Σ_i linear_E[i]*x_i + P*(Σ x_i - 1)^2
    (定数項は無視)

    (Σx-1)^2 = -Σx_i + 2Σ_{i<j} x_i x_j + 1
    -> Q_ii += (linear_E[i] - P)
       Q_ij += (2P) for i<j
    """
    n = len(linear_E)
    Q = np.zeros((n, n), dtype=float)
    for i in range(n):
        Q[i, i] += float(linear_E[i] - penalty_P)
    for i in range(n):
        for j in range(i+1, n):
            Q[i, j] += float(2.0 * penalty_P)
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
    samples = []
    energies = []

    for _ in range(int(num_reads)):
        x = np.array([rng.randint(0, 1) for _ in range(n)], dtype=int)
        E = energy_qubo(Q, x)

        for s in range(int(sweeps)):
            t = t0 + (t1 - t0) * (s / max(1, sweeps - 1))
            i = rng.randrange(n)
            x_new = x.copy()
            x_new[i] ^= 1
            E_new = energy_qubo(Q, x_new)
            dE = E_new - E
            if dE <= 0 or rng.random() < math.exp(-dE / max(t, 1e-9)):
                x, E = x_new, E_new

        samples.append(x)
        energies.append(E)

    return np.array(samples, dtype=int), np.array(energies, dtype=float)

# ============================================================
# Pack model / Loader
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

    # 必須
    for s in ["VOW_DICT", "CHAR_TO_VOW"]:
        if s not in xls.sheet_names:
            raise ValueError(f"統合Excelに必要なシート '{s}' がありません。検出={xls.sheet_names}")

    vow_dict = pd.read_excel(xls, "VOW_DICT")
    char_to_vow = pd.read_excel(xls, "CHAR_TO_VOW")

    ensure_cols(vow_dict, ["VOW_ID", "TITLE"], "VOW_DICT")
    ensure_cols(char_to_vow, ["CHAR_ID", "公式キャラ名"], "CHAR_TO_VOW")
    if "IMAGE_FILE" not in char_to_vow.columns:
        # 画像なしでも動かす
        char_to_vow["IMAGE_FILE"] = ""

    # 任意
    char_master = pd.read_excel(xls, "CHAR_MASTER") if "CHAR_MASTER" in xls.sheet_names else None
    stage_dict = pd.read_excel(xls, "STAGE_DICT") if "STAGE_DICT" in xls.sheet_names else None
    stage_to_axis = pd.read_excel(xls, "STAGE_TO_AXIS") if "STAGE_TO_AXIS" in xls.sheet_names else None
    quotes = pd.read_excel(xls, "QUOTES") if "QUOTES" in xls.sheet_names else None

    # optional validate
    if char_master is not None:
        for c in ["CHAR_ID", "AXIS_SEI", "AXIS_RYU", "AXIS_MA", "AXIS_MAKOTO"]:
            if c not in char_master.columns:
                # 軸が無いならstage補正は使わない
                char_master = None
                break

    return Pack(
        vow_dict=vow_dict,
        char_to_vow=char_to_vow,
        char_master=char_master,
        stage_dict=stage_dict,
        stage_to_axis=stage_to_axis,
        quotes=quotes,
    )

# ============================================================
# QUOTES selection
# ============================================================
def score_quote_row(
    r: pd.Series,
    observed_char_id: str,
    top_vow_ids: List[str],
    v_mix_map: Dict[str, float],
    keywords: List[str],
    stage_axis_label: str,
) -> float:
    s = 0.0

    q_char = _safe_str(r.get("CHAR_ID", ""))
    q_vow = _safe_str(r.get("VOW_ID", ""))
    q_text = _safe_str(r.get("QUOTE", ""))
    q_sense = _safe_str(r.get("SENSE_TAG", ""))
    q_axis = _safe_str(r.get("AXIS_TAG", ""))

    # 強い一致
    if q_char and q_char == str(observed_char_id):
        s += 2.5

    # vow寄与（mixが高いほど加点）
    if q_vow:
        s += 1.2 * float(v_mix_map.get(q_vow, 0.0))
        if q_vow in top_vow_ids:
            s += 0.6

    # キーワード一致（軽め）
    for kw in keywords:
        if kw and (kw in q_text or kw in q_sense):
            s += 0.25

    # stage軸タグ一致（軽め）
    if stage_axis_label and q_axis and stage_axis_label in q_axis:
        s += 0.5

    return float(s)

def pick_quotes(
    quotes_df: pd.DataFrame,
    lang: str,
    observed_char_id: str,
    top_vow_ids: List[str],
    v_mix_map: Dict[str, float],
    keywords: List[str],
    stage_axis_label: str,
    temperature: float,
    k: int = 3,
    topn: int = 50,
    rng: Optional[np.random.Generator] = None,
):
    if rng is None:
        rng = np.random.default_rng()

    df = quotes_df.copy()
    if "QUOTE" not in df.columns:
        return []

    # lang filter（列があれば）
    if lang and "LANG" in df.columns:
        cand = df[df["LANG"].fillna("").astype(str).str.lower() == lang.lower()].copy()
        if cand.empty:
            cand = df.copy()
    else:
        cand = df.copy()

    # スコア計算
    scores = []
    for _, r in cand.iterrows():
        scores.append(score_quote_row(r, observed_char_id, top_vow_ids, v_mix_map, keywords, stage_axis_label))
    cand["SCORE"] = scores

    cand = cand.sort_values("SCORE", ascending=False).head(int(topn)).copy()
    if cand.empty:
        return []

    p = softmax(cand["SCORE"].to_numpy(dtype=float), temperature=max(1e-6, float(temperature)))

    picks = []
    n_pick = min(int(k), len(cand))
    # 重複なし抽選（上位偏重）
    idxs = rng.choice(len(cand), size=n_pick, replace=False, p=p)
    rows = cand.iloc[idxs].to_dict("records")
    for r in rows:
        picks.append({
            "QUOTE": _safe_str(r.get("QUOTE", "")).strip(),
            "SOURCE": _safe_str(r.get("SOURCE", "")).strip(),
            "SCORE": float(r.get("SCORE", 0.0)),
            "CHAR_ID": _safe_str(r.get("CHAR_ID", "")),
            "VOW_ID": _safe_str(r.get("VOW_ID", "")),
        })
    return picks

# ============================================================
# UI
# ============================================================
st.title(APP_TITLE)

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

    season = season_from_month(month)
    time_slot = time_slot_from_hour(hour)
    stage_id_guess = build_stage_id(season, time_slot)
    st.caption(f"推定STAGE_ID: {stage_id_guess}")

    st.divider()
    st.header("⚙️ QUBO設定（one-hot）")
    penalty = st.slider("one-hot ペナルティ P", 1.0, 200.0, 40.0, 1.0)
    num_reads = st.slider("サンプル数（観測回数）", 50, 800, 240, 10)
    sweeps = st.slider("SA sweeps（探索深さ）", 50, 1200, 420, 10)
    temperature = st.slider("温度（大→揺らぐ）", 0.1, 5.0, 1.2, 0.1)

    st.divider()
    st.header("🧠 テキスト影響")
    ngram_n = st.selectbox("n-gram", [2, 3], index=1)
    alpha_text = st.slider("テキスト影響 α（0=スライダーのみ / 1=テキスト優勢）", 0.0, 1.0, 0.45, 0.05)

    st.divider()
    st.header("🗣️ 神託（QUOTES）")
    quote_lang = st.selectbox("LANG", ["ja", "en", ""], index=0, help="空=全言語")
    quote_temp = st.slider("格言温度（高→ランダム / 低→上位固定）", 0.2, 3.0, 1.2, 0.1)

    run_btn = st.button("🧪 QUBOで観測する", use_container_width=True)

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

# matrices
c2v = pack.char_to_vow.copy()
vow_cols = get_vow_cols(c2v)
if len(vow_cols) != 12:
    st.warning(f"VOW列が12本でないようです（検出={len(vow_cols)}本）: {vow_cols}")

ensure_cols(c2v, ["CHAR_ID", "公式キャラ名", "IMAGE_FILE"], "CHAR_TO_VOW")
W_char_vow = c2v[vow_cols].fillna(0).astype(float).to_numpy()  # (n_char, n_vow)

char_ids = c2v["CHAR_ID"].astype(str).tolist()
char_names = c2v["公式キャラ名"].astype(str).tolist()
img_files = c2v["IMAGE_FILE"].astype(str).tolist()

# VOW title map（必ず意味が見えるUI用）
vow_title_map = {}
vow_left_map = {}
vow_right_map = {}
for _, r in pack.vow_dict.iterrows():
    vid = _safe_str(r.get("VOW_ID", ""))
    vow_title_map[vid] = _safe_str(r.get("TITLE", vid))
    # LRラベルが存在すれば利用（with_lr系Excel対応）
    for cand in ["LEFT_LABEL", "LEFT", "LEFT_TEXT"]:
        if cand in pack.vow_dict.columns:
            vow_left_map[vid] = _safe_str(r.get(cand, ""))
            break
    for cand in ["RIGHT_LABEL", "RIGHT", "RIGHT_TEXT"]:
        if cand in pack.vow_dict.columns:
            vow_right_map[vid] = _safe_str(r.get(cand, ""))
            break

# Stage axis
stage_axis = np.zeros(4, dtype=float)
stage_axis_label = ""
stage_gain = 0.25  # 影響は控えめ（必要ならUI化）
if pack.stage_to_axis is not None:
    # STAGE_DICT があれば stage_id を候補から選ばせる
    stage_id = stage_id_guess
    if pack.stage_dict is not None and "STAGE_ID" in pack.stage_dict.columns:
        stage_ids = pack.stage_dict["STAGE_ID"].astype(str).tolist()
        stage_label_map = {}
        if "LABEL" in pack.stage_dict.columns:
            stage_label_map = dict(zip(pack.stage_dict["STAGE_ID"].astype(str), pack.stage_dict["LABEL"].astype(str)))
        if stage_ids:
            # 推定が無ければ先頭
            default_idx = stage_ids.index(stage_id_guess) if stage_id_guess in stage_ids else 0
            stage_id = st.sidebar.selectbox(
                "STAGE_ID（手動上書き可）",
                options=stage_ids,
                index=default_idx,
                format_func=lambda x: f"{x} | {stage_label_map.get(x, '')}",
            )
    stage_axis = get_stage_axis_weights(pack.stage_to_axis, stage_id)
    axis_labels = ["静", "流", "間", "誠"]
    if np.any(np.abs(stage_axis) > 1e-12):
        stage_axis_label = axis_labels[int(np.argmax(np.abs(stage_axis)))]

# Char axis (optional)
A_char_axis = None
if pack.char_master is not None:
    cm = pack.char_master.copy()
    cm_map = cm.set_index("CHAR_ID")[["AXIS_SEI", "AXIS_RYU", "AXIS_MA", "AXIS_MAKOTO"]].to_dict(orient="index")
    axis_mat = []
    for cid in char_ids:
        r = cm_map.get(cid, {})
        axis_mat.append([_safe_float(r.get("AXIS_SEI", 0)), _safe_float(r.get("AXIS_RYU", 0)), _safe_float(r.get("AXIS_MA", 0)), _safe_float(r.get("AXIS_MAKOTO", 0))])
    A_char_axis = np.array(axis_mat, dtype=float)  # (n_char, 4)

# ============================================================
# Main layout
# ============================================================
left, right = st.columns([1.05, 1.0], gap="large")

with left:
    st.subheader("✅ Step1：誓願入力（スライダー）＋ テキスト（自動）")

    user_text = st.text_area(
        "あなたの誓願（文章）",
        height=100,
        placeholder="例：迷いを断ち切って、新しい一歩を踏み出したい…"
    )

    # スライダー（意味が見える表示）
    v_user = np.zeros(len(vow_cols), dtype=float)

    cols = st.columns(2)
    for i, vid in enumerate(vow_cols):
        title = vow_title_map.get(vid, vid)
        l = vow_left_map.get(vid, "")
        r = vow_right_map.get(vid, "")
        if l or r:
            label = f"{vid}｜{title}  ←{l}｜{r}→"
        else:
            label = f"{vid}｜{title}"

        with cols[i % 2]:
            v_user[i] = st.slider(label, 0.0, 5.0, 0.0, 0.5, key=f"vow_{vid}")

    v_user01 = v_user / 5.0  # 0..1

    # テキスト→誓願（0..1）
    vow_texts = build_vow_texts(pack.vow_dict, vow_cols)
    v_text01 = text_to_vow_vector(user_text, vow_texts, n=int(ngram_n))

    # 混合（0..1）
    v_mix01 = (1.0 - alpha_text) * v_user01 + alpha_text * v_text01
    v_mix01 = np.clip(v_mix01, 0.0, 1.0)

    # 表示
    df_show = pd.DataFrame({
        "VOW": vow_cols,
        "TITLE": [vow_title_map.get(v, v) for v in vow_cols],
        "slider(v_user01)": np.round(v_user01, 3),
        "text(v_text01)": np.round(v_text01, 3),
        "mix(v_mix01)": np.round(v_mix01, 3),
    })
    with st.expander("🔎 誓願ベクトル（確認）"):
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.subheader("✅ Step2：12神（キャラクター）一覧")
    st.caption("※画像が repo にある場合、全員表示します（表示できない場合はファイル名/パス確認）。")

    gallery_cols = st.columns(4)
    for i in range(len(char_names)):
        with gallery_cols[i % 4]:
            p = os.path.join(img_dir, img_files[i]) if img_files[i] else ""
            img = load_image(p)
            if img is not None:
                st.image(img, use_container_width=True)
            st.caption(f"{char_names[i]}")

    st.subheader("✅ Step3：QUBO(one-hot)を組む")
    # スコア（大きいほど良い）→ エネルギーは -score
    score_vow = W_char_vow @ v_mix01  # (n_char,)
    score = score_vow.copy()

    if A_char_axis is not None and np.linalg.norm(stage_axis) > 1e-12:
        score_axis = A_char_axis @ stage_axis
        score_axis = normalize01(score_axis) * stage_gain
        score = score + score_axis

    # 微小ノイズ（完全同点回避）
    rng_np = np.random.default_rng(0)
    score = score + rng_np.normal(0, 0.002, size=score.shape)

    linear_E = -score  # 最小化

    # Pを最低限確保（one-hotを効かせる）
    minP = float(np.max(np.abs(linear_E)) * 3.0 + 1.0)
    P = max(float(penalty), minP)

    Q = build_qubo_onehot(linear_E, P)

    st.markdown(
        f"""
**QUBO（本実装）**
- 変数：`x_i ∈ {{0,1}}`（12神）
- 目的：`E(x) = Σ_i (E_i x_i) + P(Σ_i x_i − 1)^2`
- `P = {P:.2f}`（one-hot制約を強制）
"""
    )

    with st.expander("🔎 QUBO行列 Q（上三角）を表示（第三者説明用）"):
        st.dataframe(pd.DataFrame(np.round(Q, 3)), use_container_width=True)

# ============================================================
# QUBO sampling & results
# ============================================================
def run_qubo_once():
    samples, Es = sa_sample_qubo(
        Q,
        num_reads=int(num_reads),
        sweeps=int(sweeps),
        t0=float(5.0 * temperature),
        t1=float(0.2 * temperature),
        seed=random.randint(0, 10**9),
    )

    idxs = []
    for x in samples:
        k = onehot_index(x)
        if k is not None:
            idxs.append(k)

    violation = 1.0 - (len(idxs) / max(1, len(samples)))

    counts = np.zeros(len(char_names), dtype=int)
    for k in idxs:
        counts[k] += 1

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

    return counts, best_k, violation

if run_btn:
    counts, best_k, violation = run_qubo_once()
    st.session_state["counts"] = counts
    st.session_state["best_k"] = best_k
    st.session_state["violation"] = float(violation)
    st.session_state["P"] = float(P)

# ============================================================
# Right panel
# ============================================================
with right:
    st.subheader("✅ Step4：結果（観測された神 / 観測分布 / 神託）")

    if "best_k" not in st.session_state:
        st.info("左サイドバーの **「🧪 QUBOで観測する」** を押すと結果が出ます。")
    else:
        best_k = int(st.session_state["best_k"])
        counts = st.session_state["counts"]
        violation = float(st.session_state["violation"])

        st.markdown(f"### 🌟 観測された神（QUBO解）\n**{char_names[best_k]}**（CHAR_ID={char_ids[best_k]}）")

        # Image big
        p = os.path.join(img_dir, img_files[best_k]) if img_files[best_k] else ""
        img = load_image(p)
        if img is not None:
            st.image(img, caption=img_files[best_k], use_container_width=True)
        else:
            st.warning(f"画像が見つかりません: {p}")

        st.markdown("### 📊 観測分布（同一サンプル集合・one-hot成立のみ）")
        df_hist = pd.DataFrame({"神": char_names, "count": counts}).sort_values("count", ascending=False)
        st.bar_chart(df_hist.set_index("神")["count"])

        st.markdown("### 🧾 one-hot制約の状態")
        st.write(f"制約違反率: **{violation*100:.1f}%**（低いほど良い）")
        if violation > 0.15:
            st.info("違反率が高い場合：Pを上げる / sweepsを増やす / 温度を下げる と安定します。")

        # Contributing vows
        char_w = W_char_vow[best_k, :]
        contrib = char_w * v_mix01
        top_idx = np.argsort(contrib)[::-1][:6]
        top_vow_ids = [vow_cols[i] for i in top_idx]
        v_mix_map = {vow_cols[i]: float(v_mix01[i]) for i in range(len(vow_cols))}

        contrib_df = pd.DataFrame({
            "VOW": top_vow_ids,
            "TITLE": [vow_title_map.get(v, v) for v in top_vow_ids],
            "mix": [float(np.round(v_mix_map.get(v, 0.0), 3)) for v in top_vow_ids],
            "W(char,v)": [float(np.round(char_w[vow_cols.index(v)], 3)) for v in top_vow_ids],
            "寄与(v*w)": [float(np.round(contrib[vow_cols.index(v)], 3)) for v in top_vow_ids],
        })
        st.markdown("### 🧩 寄与した誓願（Top）")
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

        # QUOTES oracle
        st.markdown("### 🗣️ 神託（QUOTES）")
        if pack.quotes is None:
            st.info("Excelに QUOTES シートがありません（あれば神託を表示できます）。")
        else:
            keywords = extract_keywords_simple(user_text, topk=10)
            quotes = pick_quotes(
                quotes_df=pack.quotes,
                lang=quote_lang,
                observed_char_id=char_ids[best_k],
                top_vow_ids=top_vow_ids,
                v_mix_map=v_mix_map,
                keywords=keywords,
                stage_axis_label=stage_axis_label,
                temperature=quote_temp,
                k=3,
                topn=60,
                rng=np.random.default_rng(),
            )

            if not quotes:
                st.warning("QUOTESから神託が選べませんでした（列不足/内容/フィルタを確認）。")
                if "LANG" in pack.quotes.columns:
                    st.caption("ヒント：LANGが ja/en 以外なら、サイドバーで空（全言語）にしてください。")
            else:
                # 神託本文（簡易ストーリー）
                top_titles = [vow_title_map.get(v, v) for v in top_vow_ids[:3]]
                header = f"いまの波は **{'・'.join(top_titles)}** に寄っている。"
                if stage_axis_label:
                    header += f"（Stageは **{stage_axis_label}** を強める）"
                st.success(header)

                for i, q in enumerate(quotes, start=1):
                    qt = q.get("QUOTE", "")
                    src = q.get("SOURCE", "")
                    st.markdown(f"**神託 {i}**")
                    st.write(f"「{qt}」")
                    if src:
                        st.caption(f"出典: {src}")

        # QUBO evidence
        st.markdown("### 🧠 QUBO 証拠（第三者説明用）")
        x_best = [1 if i == best_k else 0 for i in range(len(char_names))]
        st.code(
            "E(x) = Σ_i (E_i * x_i) + P(Σ_i x_i − 1)^2\n"
            f"P = {P:.2f}\n"
            f"x = {x_best}\n"
            f"E(x) = {energy_qubo(Q, np.array(x_best, dtype=int)):.6f}",
            language="text"
        )

st.caption("© Q-Quest / Quantum Shintaku prototype (QUBO one-hot)")
