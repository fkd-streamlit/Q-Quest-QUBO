# -*- coding: utf-8 -*-
"""
Q-Quest-QUBO : Quantum Shintaku (QUBO one-hot version)
-----------------------------------------------------
- 神(12)を one-hot QUBO で選択
- E(x) = Σ E_i x_i + P(Σx - 1)^2
- Simulated Annealing によるサンプリング
"""

import math
import random
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ============================================================
# Streamlit config（必ず最初）
# ============================================================
st.set_page_config(
    page_title="Q-Quest-QUBO｜量子神託（one-hot QUBO）",
    layout="wide"
)

# ============================================================
# Utility
# ============================================================
def normalize01(v):
    mn, mx = np.min(v), np.max(v)
    if mx - mn < 1e-9:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn)

def char_ngrams(text, n=3):
    s = re.sub(r"\s+", "", str(text))
    return Counter(s[i:i+n] for i in range(len(s)-n+1)) if len(s) >= n else Counter()

def cosine_counter(a: Counter, b: Counter):
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0) for k in keys], float)
    vb = np.array([b.get(k, 0) for k in keys], float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))

# ============================================================
# QUBO core
# ============================================================
def build_qubo_onehot(linear_E, P):
    """
    E(x)=ΣE_i x_i + P(Σx-1)^2
    """
    n = len(linear_E)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] += linear_E[i] - P
        for j in range(i+1, n):
            Q[i, j] += 2 * P
    return Q

def energy(Q, x):
    return float(x @ Q @ x)

def onehot_index(x):
    idx = np.where(x == 1)[0]
    return int(idx[0]) if len(idx) == 1 else None

def sa_sample(Q, num_reads=200, sweeps=400, t0=5.0, t1=0.2):
    rng = random.Random()
    n = Q.shape[0]
    samples, energies = [], []

    for _ in range(num_reads):
        x = np.array([rng.randint(0, 1) for _ in range(n)], int)
        E = energy(Q, x)

        for s in range(sweeps):
            t = t0 + (t1 - t0) * (s / max(1, sweeps-1))
            i = rng.randrange(n)
            xn = x.copy()
            xn[i] ^= 1
            En = energy(Q, xn)
            if En <= E or rng.random() < math.exp(-(En-E)/max(t,1e-9)):
                x, E = xn, En

        samples.append(x)
        energies.append(E)

    return np.array(samples), np.array(energies)

# ============================================================
# UI
# ============================================================
st.title("🔮 Q-Quest-QUBO｜量子神託（one-hot QUBO 実装）")

with st.sidebar:
    st.header("📁 データ")
    excel_file = st.file_uploader(
        "統合Excel（pack）",
        type=["xlsx"]
    )

    st.header("⚙ QUBO設定")
    P_user = st.slider("one-hot ペナルティ P", 1.0, 200.0, 40.0, 1.0)
    num_reads = st.slider("サンプル数", 50, 600, 240, 10)
    sweeps = st.slider("SA sweeps", 50, 1000, 420, 10)

    run = st.button("🧪 QUBOで観測")

if excel_file is None:
    st.info("左サイドバーから **統合Excel** をアップロードしてください。")
    st.stop()

# ============================================================
# Excel load
# ============================================================
xl = pd.ExcelFile(excel_file)
required = ["VOW_DICT", "CHAR_TO_VOW"]
for s in required:
    if s not in xl.sheet_names:
        st.error(f"必須シート不足: {s}")
        st.stop()

df_vow = xl.parse("VOW_DICT")
df_char = xl.parse("CHAR_TO_VOW")

vow_cols = [c for c in df_char.columns if c.startswith("VOW_")]
char_names = df_char["公式キャラ名"].astype(str).tolist()
img_files = df_char["IMAGE_FILE"].astype(str).tolist()

# ============================================================
# Step1 誓願入力
# ============================================================
st.subheader("① 誓願入力（スライダー）")
v_user = np.zeros(len(vow_cols))
for i, c in enumerate(vow_cols):
    title = df_vow[df_vow["VOW_ID"] == c]["TITLE"].values
    label = title[0] if len(title) else c
    v_user[i] = st.slider(label, 0.0, 5.0, 0.0, 0.5)

v_user = v_user / 5.0

# ============================================================
# Step2 線形エネルギー
# ============================================================
W = df_char[vow_cols].fillna(0).to_numpy(float)
score = W @ v_user
linear_E = -score

P_min = np.max(np.abs(linear_E)) * 3 + 1
P = max(P_user, P_min)

Q = build_qubo_onehot(linear_E, P)

# ============================================================
# Step3 QUBO sampling
# ============================================================
if run:
    samples, Es = sa_sample(Q, num_reads, sweeps)

    idxs = [onehot_index(x) for x in samples if onehot_index(x) is not None]
    counts = np.zeros(len(char_names), int)
    for k in idxs:
        counts[k] += 1

    best_k = min(
        [(energy(Q, x), onehot_index(x)) for x in samples if onehot_index(x) is not None],
        key=lambda t: t[0]
    )[1]

    st.session_state["counts"] = counts
    st.session_state["best_k"] = best_k
    st.session_state["P"] = P
    st.session_state["Q"] = Q

# ============================================================
# Result
# ============================================================
if "best_k" in st.session_state:
    k = st.session_state["best_k"]
    st.subheader("🌟 観測された神（QUBO解）")
    st.write(f"**{char_names[k]}**")

    img_path = Path("assets/images/characters") / img_files[k]
    if img_path.exists():
        st.image(Image.open(img_path), use_container_width=True)

    st.subheader("📊 観測分布")
    hist = pd.DataFrame({
        "神": char_names,
        "count": st.session_state["counts"]
    }).sort_values("count", ascending=False)
    st.bar_chart(hist.set_index("神"))

    st.subheader("🧠 QUBO 証拠")
    x = [1 if i == k else 0 for i in range(len(char_names))]
    st.code(
        f"E(x)=ΣE_i x_i + P(Σx-1)^2\n"
        f"P={st.session_state['P']:.2f}\n"
        f"x={x}",
        language="text"
    )