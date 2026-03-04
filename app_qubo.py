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
  background: rgba(6,8,18,0.95) !important;
  border-right: 1px solid rgba(255,255,255,0.10);
  backdrop-filter: blur(10px);
}
section[data-testid="stSidebar"] *{
  color: rgba(245,245,255,0.92) !important;
}

/* --- Sidebar content boxes --- */
section[data-testid="stSidebar"] > div {
  background: rgba(6,8,18,0.95) !important;
}
section[data-testid="stSidebar"] [data-baseweb="base-input"],
section[data-testid="stSidebar"] [data-baseweb="input"] {
  background: rgba(10,12,26,0.95) !important;
  border: 1px solid rgba(255,255,255,0.20) !important;
}

/* --- inputs: make text visible --- */
textarea, input[type="text"], input[type="number"] {
  color: rgba(245,245,255,0.95) !important;
  background: rgba(10,12,26,0.95) !important;
  border: 1px solid rgba(255,255,255,0.20) !important;
}
textarea::placeholder, input::placeholder {
  color: rgba(245,245,255,0.55) !important;
}

/* --- Streamlit specific input components --- */
div[data-baseweb="base-input"],
div[data-baseweb="input"],
div[data-baseweb="textarea"] {
  background: rgba(10,12,26,0.95) !important;
}
div[data-baseweb="base-input"] input,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
  color: rgba(245,245,255,0.95) !important;
  background: rgba(10,12,26,0.95) !important;
  border: 1px solid rgba(255,255,255,0.20) !important;
}

/* --- Text area specific --- */
div[data-testid="stTextArea"] textarea {
  color: rgba(245,245,255,0.95) !important;
  background: rgba(10,12,26,0.95) !important;
  border: 1px solid rgba(255,255,255,0.20) !important;
}

/* --- Text input specific --- */
div[data-testid="stTextInput"] input {
  color: rgba(245,245,255,0.95) !important;
  background: rgba(10,12,26,0.95) !important;
  border: 1px solid rgba(255,255,255,0.20) !important;
}

/* --- Sidebar selectbox, slider, toggle --- */
section[data-testid="stSidebar"] [data-baseweb="select"],
section[data-testid="stSidebar"] [data-baseweb="slider"],
section[data-testid="stSidebar"] [data-baseweb="checkbox"] {
  background: rgba(10,12,26,0.95) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: rgba(10,12,26,0.95) !important;
  color: rgba(245,245,255,0.95) !important;
  border: 1px solid rgba(255,255,255,0.20) !important;
}

/* --- Slider track and thumb --- */
section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
  background: rgba(100,150,255,0.8) !important;
}
section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"]:hover {
  background: rgba(120,170,255,1) !important;
}

/* --- Toggle/Checkbox --- */
section[data-testid="stSidebar"] [data-baseweb="checkbox"] label {
  color: rgba(245,245,255,0.95) !important;
}

/* --- file uploader (black panel fix) --- */
div[data-testid="stFileUploader"],
div[data-testid="stFileUploader"] > div,
div[data-testid="stFileUploader"] > div > div {
  background: rgba(10,12,26,0.95) !important;
  border: 1px solid rgba(255,255,255,0.20) !important;
  border-radius: 14px !important;
  padding: 10px !important;
}
/* すべてのテキストを白に */
div[data-testid="stFileUploader"],
div[data-testid="stFileUploader"] *,
div[data-testid="stFileUploader"] * * {
  color: rgba(245,245,255,0.95) !important;
}
/* ドラッグ&ドロップエリア */
div[data-testid="stFileUploader"] [data-baseweb="file-uploader"],
div[data-testid="stFileUploader"] [data-baseweb="file-uploader"] * {
  background: rgba(10,12,26,0.95) !important;
  color: rgba(245,245,255,0.95) !important;
  border-color: rgba(255,255,255,0.20) !important;
}
/* ファイルアップローダーの内部テキスト */
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] div,
div[data-testid="stFileUploader"] label {
  color: rgba(245,245,255,0.95) !important;
  background: rgba(10,12,26,0.95) !important;
}
/* ボタン */
div[data-testid="stFileUploader"] button,
div[data-testid="stFileUploader"] [role="button"] {
  background: rgba(20,30,50,0.8) !important;
  color: rgba(245,245,255,0.95) !important;
  border: 1px solid rgba(255,255,255,0.20) !important;
}
div[data-testid="stFileUploader"] button:hover,
div[data-testid="stFileUploader"] [role="button"]:hover {
  background: rgba(30,40,60,0.9) !important;
}
/* アップロード済みファイル名 */
div[data-testid="stFileUploader"] [data-baseweb="file-uploader"] [data-baseweb="file-name"],
div[data-testid="stFileUploader"] [data-baseweb="file-uploader"] [data-baseweb="file-size"] {
  color: rgba(245,245,255,0.95) !important;
}
/* 白い背景を黒に上書き */
div[data-testid="stFileUploader"] [style*="background-color: rgb(255"],
div[data-testid="stFileUploader"] [style*="background-color:rgb(255"],
div[data-testid="stFileUploader"] [style*="background: rgb(255"],
div[data-testid="stFileUploader"] [style*="background:rgb(255"],
div[data-testid="stFileUploader"] [style*="#ffffff"],
div[data-testid="stFileUploader"] [style*="#FFFFFF"] {
  background-color: rgba(10,12,26,0.95) !important;
  background: rgba(10,12,26,0.95) !important;
  color: rgba(245,245,255,0.95) !important;
}
/* 薄い灰色の文字を白に */
div[data-testid="stFileUploader"] [style*="color: rgb(128"],
div[data-testid="stFileUploader"] [style*="color:rgb(128"],
div[data-testid="stFileUploader"] [style*="color: rgb(200"],
div[data-testid="stFileUploader"] [style*="color:rgb(200"],
div[data-testid="stFileUploader"] [style*="color: rgb(150"],
div[data-testid="stFileUploader"] [style*="color:rgb(150"] {
  color: rgba(245,245,255,0.95) !important;
}

/* --- Expander (折りたたみ可能セクション) - より強力に --- */
div[data-testid="stExpander"],
div[data-testid="stExpander"] > div,
div[data-testid="stExpander"] > div > div,
div[data-testid="stExpander"] > div > div > div {
  background: rgba(10,12,26,0.95) !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  border-radius: 8px !important;
  color: rgba(245,245,255,0.95) !important;
}
/* Expanderのタイトル（ヘッダー） */
div[data-testid="stExpander"] [data-baseweb="accordion"],
div[data-testid="stExpander"] [data-baseweb="accordion"] * {
  background: rgba(10,12,26,0.95) !important;
  color: rgba(245,245,255,0.95) !important;
}
div[data-testid="stExpander"] [data-baseweb="accordion"] button,
div[data-testid="stExpander"] [data-baseweb="accordion"] [role="button"],
div[data-testid="stExpander"] summary {
  background: rgba(10,12,26,0.95) !important;
  color: rgba(245,245,255,0.95) !important;
  border: none !important;
}
div[data-testid="stExpander"] [data-baseweb="accordion"] button *,
div[data-testid="stExpander"] [data-baseweb="accordion"] [role="button"] *,
div[data-testid="stExpander"] summary *,
div[data-testid="stExpander"] [data-baseweb="accordion"] button * * {
  color: rgba(245,245,255,0.95) !important;
}
/* Expanderのコンテンツ */
div[data-testid="stExpander"] [data-baseweb="accordion-panel"],
div[data-testid="stExpander"] [data-baseweb="accordion-panel"] *,
div[data-testid="stExpander"] [data-baseweb="accordion-panel"] * * {
  background: rgba(10,12,26,0.95) !important;
  color: rgba(245,245,255,0.95) !important;
}
/* Expander内のすべてのテキスト（最優先） */
div[data-testid="stExpander"],
div[data-testid="stExpander"] *,
div[data-testid="stExpander"] * *,
div[data-testid="stExpander"] * * *,
div[data-testid="stExpander"] * * * * {
  color: rgba(245,245,255,0.95) !important;
}
/* Expander内のmarkdownテキスト */
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] span,
div[data-testid="stExpander"] div,
div[data-testid="stExpander"] strong,
div[data-testid="stExpander"] b,
div[data-testid="stExpander"] h1,
div[data-testid="stExpander"] h2,
div[data-testid="stExpander"] h3,
div[data-testid="stExpander"] h4,
div[data-testid="stExpander"] h5,
div[data-testid="stExpander"] h6 {
  color: rgba(245,245,255,0.95) !important;
}
/* Expander内のDataFrameも白文字に */
div[data-testid="stExpander"] div[data-testid="stDataFrame"],
div[data-testid="stExpander"] div[data-testid="stDataFrame"] *,
div[data-testid="stExpander"] div[data-testid="stDataFrame"] * * {
  color: rgba(245,245,255,0.95) !important;
}
/* Expander内のst.markdownコンテンツ */
div[data-testid="stExpander"] [data-testid="stMarkdownContainer"],
div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] *,
div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] * * {
  color: rgba(245,245,255,0.95) !important;
}
/* 白い背景を黒に上書き */
div[data-testid="stExpander"] [style*="background-color: rgb(255"],
div[data-testid="stExpander"] [style*="background-color:rgb(255"],
div[data-testid="stExpander"] [style*="background: rgb(255"],
div[data-testid="stExpander"] [style*="background:rgb(255"],
div[data-testid="stExpander"] [style*="#ffffff"],
div[data-testid="stExpander"] [style*="#FFFFFF"] {
  background-color: rgba(10,12,26,0.95) !important;
  background: rgba(10,12,26,0.95) !important;
  color: rgba(245,245,255,0.95) !important;
}
/* 黒い文字を白に上書き（すべてのパターン） */
div[data-testid="stExpander"] [style*="color"],
div[data-testid="stExpander"] [style*="color"] * {
  color: rgba(245,245,255,0.95) !important;
}
/* 特定の色パターンも上書き */
div[data-testid="stExpander"] [style*="color: rgb(0"],
div[data-testid="stExpander"] [style*="color:rgb(0"],
div[data-testid="stExpander"] [style*="color: rgb(38"],
div[data-testid="stExpander"] [style*="color:rgb(38"],
div[data-testid="stExpander"] [style*="color: rgb(49"],
div[data-testid="stExpander"] [style*="color:rgb(49"],
div[data-testid="stExpander"] [style*="color: rgb(19"],
div[data-testid="stExpander"] [style*="color:rgb(19"] {
  color: rgba(245,245,255,0.95) !important;
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
div[data-testid="stDataFrame"],
div[data-testid="stDataFrame"] > div,
div[data-testid="stDataFrame"] > div > div,
div[data-testid="stDataFrame"] > div > div > div {
  border-radius: 16px !important;
  overflow: hidden !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  background: rgba(10,12,26,0.95) !important;
}
/* Force all text to white - comprehensive override (最優先) */
div[data-testid="stDataFrame"],
div[data-testid="stDataFrame"] *,
div[data-testid="stDataFrame"] * *,
div[data-testid="stDataFrame"] * * *,
div[data-testid="stDataFrame"] * * * * {
  color: rgba(245,245,255,0.95) !important;
}
/* さらに強力: すべてのテキストノードを強制 */
div[data-testid="stDataFrame"] {
  color: rgba(245,245,255,0.95) !important;
}
div[data-testid="stDataFrame"] * {
  color: rgba(245,245,255,0.95) !important;
}
/* BaseWeb Table (Streamlit DataFrame internal) - より強力に --- */
div[data-testid="stDataFrame"] [data-baseweb="table"],
div[data-testid="stDataFrame"] [data-baseweb="table"] *,
div[data-testid="stDataFrame"] [data-baseweb="table"] * *,
div[data-testid="stDataFrame"] [data-baseweb="table"] * * *,
div[data-testid="stDataFrame"] [data-baseweb="table"] * * * * {
  color: rgba(245,245,255,0.95) !important;
  background: rgba(10,12,26,0.95) !important;
}
/* BaseWebのすべての要素 */
div[data-testid="stDataFrame"] [data-baseweb] {
  color: rgba(245,245,255,0.95) !important;
}
div[data-testid="stDataFrame"] [data-baseweb] * {
  color: rgba(245,245,255,0.95) !important;
}
div[data-testid="stDataFrame"] [data-baseweb="table"] [data-baseweb="table-head"],
div[data-testid="stDataFrame"] [data-baseweb="table"] [data-baseweb="table-body"],
div[data-testid="stDataFrame"] [data-baseweb="table"] [data-baseweb="table-head"] *,
div[data-testid="stDataFrame"] [data-baseweb="table"] [data-baseweb="table-body"] * {
  color: rgba(245,245,255,0.95) !important;
  background: rgba(10,12,26,0.95) !important;
}
div[data-testid="stDataFrame"] [role="grid"],
div[data-testid="stDataFrame"] [role="row"],
div[data-testid="stDataFrame"] [role="rowgroup"],
div[data-testid="stDataFrame"] [role="gridcell"],
div[data-testid="stDataFrame"] [role="columnheader"]{
  color: rgba(245,245,255,0.95) !important;
  background: rgba(10,12,26,0.95) !important;
}
div[data-testid="stDataFrame"] [role="gridcell"] *,
div[data-testid="stDataFrame"] [role="gridcell"] * *,
div[data-testid="stDataFrame"] [role="columnheader"] *,
div[data-testid="stDataFrame"] [role="columnheader"] * * {
  color: rgba(245,245,255,0.95) !important;
}
div[data-testid="stDataFrame"] [role="columnheader"],
div[data-testid="stDataFrame"] [role="columnheader"] *,
div[data-testid="stDataFrame"] [role="columnheader"] * * {
  background: rgba(15,18,35,0.98) !important;
  color: rgba(245,245,255,1) !important;
  border-bottom: 1px solid rgba(255,255,255,0.15) !important;
  font-weight: 600 !important;
}
div[data-testid="stDataFrame"] [role="gridcell"],
div[data-testid="stDataFrame"] [role="gridcell"] *,
div[data-testid="stDataFrame"] [role="gridcell"] * * {
  background: rgba(10,12,26,0.95) !important;
  color: rgba(245,245,255,0.95) !important;
  border-bottom: 1px solid rgba(255,255,255,0.08) !important;
}
div[data-testid="stDataFrame"] [data-testid="stTable"] {
  background: rgba(10,12,26,0.95) !important;
}

/* --- Streamlit DataFrame wrapper --- */
div[data-testid="stDataFrame"] > div[style*="background"] {
  background: rgba(10,12,26,0.95) !important;
}

/* --- st.table fallback --- */
table,
table * {
  background: rgba(10,12,26,0.95) !important;
  color: rgba(245,245,255,0.95) !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
}
thead tr th,
thead tr th * {
  background: rgba(15,18,35,0.98) !important;
  color: rgba(245,245,255,1) !important;
  border-bottom: 1px solid rgba(255,255,255,0.15) !important;
  font-weight: 600 !important;
}
tbody tr td,
tbody tr td * {
  background: rgba(10,12,26,0.95) !important;
  color: rgba(245,245,255,0.95) !important;
  border-bottom: 1px solid rgba(255,255,255,0.08) !important;
}
tbody tr:hover td {
  background: rgba(15,18,35,0.8) !important;
}

/* --- Force all table elements to dark --- */
[data-testid="stDataFrame"] [style*="background-color"],
[data-testid="stDataFrame"] [style*="background"],
[data-testid="stDataFrame"] div[style],
[data-testid="stDataFrame"] span[style] {
  background-color: rgba(10,12,26,0.95) !important;
  background: rgba(10,12,26,0.95) !important;
}

/* --- Override any white backgrounds in tables --- */
div[data-testid="stDataFrame"] [style*="rgb(255, 255, 255)"],
div[data-testid="stDataFrame"] [style*="rgb(255,255,255)"],
div[data-testid="stDataFrame"] [style*="#ffffff"],
div[data-testid="stDataFrame"] [style*="#FFFFFF"] {
  background-color: rgba(10,12,26,0.95) !important;
  background: rgba(10,12,26,0.95) !important;
  color: rgba(245,245,255,0.95) !important;
}

/* --- Force text color in all table cells --- */
div[data-testid="stDataFrame"] p,
div[data-testid="stDataFrame"] span,
div[data-testid="stDataFrame"] div,
div[data-testid="stDataFrame"] td,
div[data-testid="stDataFrame"] th,
div[data-testid="stDataFrame"] [role="gridcell"] *,
div[data-testid="stDataFrame"] [role="columnheader"] * {
  color: rgba(245,245,255,0.95) !important;
}

/* --- Force white text in all table elements (override any black text) --- */
div[data-testid="stDataFrame"] [style*="color: rgb(0"],
div[data-testid="stDataFrame"] [style*="color:rgb(0"],
div[data-testid="stDataFrame"] [style*="color:#000"],
div[data-testid="stDataFrame"] [style*="color:#000000"],
div[data-testid="stDataFrame"] [style*="color: rgb(38"],
div[data-testid="stDataFrame"] [style*="color:rgb(38"],
div[data-testid="stDataFrame"] [style*="color: rgb(49"],
div[data-testid="stDataFrame"] [style*="color:rgb(49"] {
  color: rgba(245,245,255,0.95) !important;
}

/* --- Force all text in tables to be white (comprehensive) --- */
div[data-testid="stDataFrame"] *,
div[data-testid="stDataFrame"] * *,
div[data-testid="stDataFrame"] * * * {
  color: rgba(245,245,255,0.95) !important;
}
div[data-testid="stDataFrame"] [class*="text"],
div[data-testid="stDataFrame"] [class*="Text"],
div[data-testid="stDataFrame"] [class*="data"],
div[data-testid="stDataFrame"] [class*="Data"] {
  color: rgba(245,245,255,0.95) !important;
}

/* --- Override Streamlit's default table text colors --- */
div[data-testid="stDataFrame"] [data-testid="stTable"] *,
div[data-testid="stDataFrame"] [data-testid="stTable"] * *,
div[data-testid="stDataFrame"] [data-testid="stTable"] [class*="data"] *,
div[data-testid="stDataFrame"] [class*="data"] *,
div[data-testid="stDataFrame"] [class*="Data"] * {
  color: rgba(245,245,255,0.95) !important;
}

/* --- Force white text in all DataFrame wrapper elements --- */
[data-testid="stDataFrame"] [class*="stDataFrame"] *,
[data-testid="stDataFrame"] [class*="dataframe"] *,
[data-testid="stDataFrame"] [class*="DataFrame"] * {
  color: rgba(245,245,255,0.95) !important;
}

/* --- Override any inline styles that set text color to black/dark --- */
div[data-testid="stDataFrame"] [style],
div[data-testid="stDataFrame"] [style] *,
div[data-testid="stDataFrame"] [style] * * {
  color: rgba(245,245,255,0.95) !important;
}
/* インラインスタイルの色指定を無視して強制上書き */
div[data-testid="stDataFrame"] [style*="color"],
div[data-testid="stDataFrame"] [style*="color"] * {
  color: rgba(245,245,255,0.95) !important;
}
/* すべてのテキストコンテンツを強制的に白に */
div[data-testid="stDataFrame"] ::before,
div[data-testid="stDataFrame"] ::after {
  color: rgba(245,245,255,0.95) !important;
}

/* --- さらに強力なセレクタ: すべてのテキストノードを強制的に白に --- */
div[data-testid="stDataFrame"] * {
  color: rgba(245,245,255,0.95) !important;
}
/* すべての要素タイプに適用 */
div[data-testid="stDataFrame"] p,
div[data-testid="stDataFrame"] span,
div[data-testid="stDataFrame"] div,
div[data-testid="stDataFrame"] td,
div[data-testid="stDataFrame"] th,
div[data-testid="stDataFrame"] tr,
div[data-testid="stDataFrame"] table,
div[data-testid="stDataFrame"] thead,
div[data-testid="stDataFrame"] tbody,
div[data-testid="stDataFrame"] tfoot {
  color: rgba(245,245,255,0.95) !important;
}

/* --- BaseWebの特定のクラス名にも対応 --- */
div[data-testid="stDataFrame"] [class*="BaseTable"],
div[data-testid="stDataFrame"] [class*="base-table"],
div[data-testid="stDataFrame"] [class*="Table"],
div[data-testid="stDataFrame"] [class*="table"] {
  color: rgba(245,245,255,0.95) !important;
}
div[data-testid="stDataFrame"] [class*="BaseTable"] *,
div[data-testid="stDataFrame"] [class*="base-table"] *,
div[data-testid="stDataFrame"] [class*="Table"] *,
div[data-testid="stDataFrame"] [class*="table"] * {
  color: rgba(245,245,255,0.95) !important;
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

def render_dataframe_as_html_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    """
    DataFrameをHTMLテーブルとしてレンダリング（白文字・黒背景で確実に表示）
    """
    import html as html_escape
    
    if df is None or len(df) == 0:
        return "<p style='color:rgba(245,245,255,0.95);'>データがありません</p>"
    
    df_display = df.head(max_rows) if max_rows else df
    
    html = """
    <div style='background:rgba(10,12,26,0.95); border:1px solid rgba(255,255,255,0.15); border-radius:16px; overflow:auto; margin:8px 0;'>
    <table style='width:100%; border-collapse:collapse; background:rgba(10,12,26,0.95); color:rgba(245,245,255,0.95);'>
    <thead>
    <tr style='background:rgba(15,18,35,0.98);'>
    """
    
    # ヘッダー
    for col in df_display.columns:
        col_escaped = html_escape.escape(str(col))
        html += f"<th style='padding:12px 16px; border-bottom:1px solid rgba(255,255,255,0.15); color:rgba(245,245,255,1); font-weight:600; text-align:left;'>{col_escaped}</th>"
    
    html += """
    </tr>
    </thead>
    <tbody>
    """
    
    # データ行
    for idx, row in df_display.iterrows():
        html += "<tr style='border-bottom:1px solid rgba(255,255,255,0.08);'>"
        for col in df_display.columns:
            if pd.notna(row[col]):
                val = row[col]
                # 数値の場合は適切にフォーマット
                if isinstance(val, (int, float)):
                    if isinstance(val, float):
                        val_str = f"{val:.3f}" if abs(val) < 1000 else f"{val:.2f}"
                    else:
                        val_str = str(val)
                else:
                    val_str = str(val)
            else:
                val_str = ""
            
            val_escaped = html_escape.escape(val_str)
            html += f"<td style='padding:12px 16px; color:rgba(245,245,255,0.95); background:rgba(10,12,26,0.95);'>{val_escaped}</td>"
        html += "</tr>"
    
    html += """
    </tbody>
    </table>
    </div>
    """
    
    return html

def build_oracle_from_char_master_and_vows(
    char_master: Optional[pd.DataFrame],
    char_id: str,
    god_name: str,
    df_top_vows: pd.DataFrame,
    stage_id: str,
) -> str:
    """
    CHAR_MASTER の「役割」系の文言 + 寄与Topの誓願タイトルを使って、短い神託文を生成する。
    """
    role_text = ""
    if char_master is not None and len(char_master) > 0:
        cols = {norm_col(c): c for c in char_master.columns}
        cid_col = cols.get("CHAR_ID") or cols.get("CHAR") or cols.get("ID")
        if cid_col:
            cm = char_master.copy()
            cm[cid_col] = cm[cid_col].astype(str).str.strip()
            row = cm[cm[cid_col] == str(char_id)].head(1)
            if len(row) > 0:
                r0 = row.iloc[0]
                # 役割らしき列を探索（sense_to_vow_initial_filled_from_user.xlsx対応）
                # 「役割」と「役割補足説明」を優先的に使用
                role_col = cols.get(norm_col("役割"))
                role_supplement_col = cols.get(norm_col("役割補足説明"))
                
                if role_col and pd.notna(r0.get(role_col, None)):
                    role_main = str(r0.get(role_col, "")).strip()
                    if len(role_main) >= 3:
                        role_text = role_main
                        # 「役割補足説明」もあれば結合
                        if role_supplement_col and pd.notna(r0.get(role_supplement_col, None)):
                            role_supplement = str(r0.get(role_supplement_col, "")).strip()
                            if len(role_supplement) >= 3:
                                role_text = f"{role_main}\n{role_supplement}"
                elif role_supplement_col and pd.notna(r0.get(role_supplement_col, None)):
                    # 「役割」がなくても「役割補足説明」があれば使用
                    role_supplement = str(r0.get(role_supplement_col, "")).strip()
                    if len(role_supplement) >= 3:
                        role_text = role_supplement
                else:
                    # フォールバック: その他の列を探索
                    role_candidates = [
                        "ROLE", "CHAR_ROLE", "MISSION", "ミッション",
                        "CONCEPT", "コンセプト", "ARCHETYPE", "アーキタイプ",
                        "DESCRIPTION_LONG", "DESCRIPTION", "説明", "説明_長文",
                        "TRAIT", "TRAITS", "性格", "特徴", "キャラクター説明",
                        "CHAR_DESCRIPTION", "CHAR_CONCEPT", "CHAR_MISSION",
                        "公式キャラ名", "キャラクター名", "CHAR_NAME",
                    ]
                    for key in role_candidates:
                        col = cols.get(norm_col(key))
                        if col and pd.notna(r0.get(col, None)):
                            txt = str(r0.get(col, "")).strip()
                            if len(txt) >= 3:
                                role_text = txt
                                break

    # 寄与Topの誓願タイトル（最大3）
    top_titles = []
    if df_top_vows is not None and len(df_top_vows) > 0 and "TITLE" in df_top_vows.columns:
        for t in df_top_vows["TITLE"].astype(str).tolist():
            t = (t or "").strip()
            if t and t not in top_titles:
                top_titles.append(t)
            if len(top_titles) >= 3:
                break

    # 文章生成（短く・UI向け）
    if not role_text:
        role_text = f"{god_name}は「整える／導く」役目を持つ。"

    if top_titles:
        vows_part = "・".join(top_titles[:2])
        vow_line = f"いま寄与している誓願：{vows_part}"
    else:
        vow_line = "いま寄与している誓願：—"

    stage_line = f"季節×時間（Stage）：{stage_id}"
    return f"{role_text}\n\n{vow_line}\n{stage_line}"

def pick_quotes_by_char_master(char_master: Optional[pd.DataFrame], char_id: str, dfq: pd.DataFrame, lang: str, seed: int) -> Optional[pd.DataFrame]:
    """
    CHAR_MASTERシートから選ばれた神に関連する格言を選ぶ
    char_master: CHAR_MASTERシートのDataFrame
    char_id: 選ばれた神のCHAR_ID
    dfq: QUOTESシートのDataFrame（フォールバック用）
    """
    if char_master is None or len(char_master) == 0:
        return None
    
    # CHAR_MASTERから該当する行を取得
    cols = {norm_col(c): c for c in char_master.columns}
    cid_col = cols.get("CHAR_ID") or cols.get("CHAR") or cols.get("ID")
    
    if not cid_col:
        return None
    
    char_master = char_master.copy()
    char_master[cid_col] = char_master[cid_col].astype(str).str.strip()
    char_row = char_master[char_master[cid_col] == char_id]
    
    if len(char_row) == 0:
        return None
    
    char_row = char_row.iloc[0]
    
    # CHAR_MASTERから格言関連の列を探す
    quote_cols = []
    for col_name in ["格言", "QUOTE", "QUOTES", "神託", "ORACLE", "MESSAGE", "DESCRIPTION", "DESCRIPTION_LONG", "説明"]:
        norm_col_name = norm_col(col_name)
        for c in char_master.columns:
            if norm_col(c) == norm_col_name:
                quote_cols.append(c)
                break
    
    # 格言が見つかった場合
    if quote_cols:
        quote_text = str(char_row.get(quote_cols[0], "")).strip()
        if quote_text and len(quote_text) > 5:  # 有効な格言がある場合
            source_cols = []
            for col_name in ["出典", "SOURCE", "AUTHOR", "著者", "作者"]:
                norm_col_name = norm_col(col_name)
                for c in char_master.columns:
                    if norm_col(c) == norm_col_name:
                        source_cols.append(c)
                        break
            
            source = str(char_row.get(source_cols[0], "")) if source_cols else "—"
            
            return pd.DataFrame([{
                "QUOTE_ID": f"CHAR_{char_id}_QUOTE",
                "QUOTE": quote_text,
                "SOURCE": source,
                "LANG": lang or "ja"
            }])
    
    # CHAR_MASTERに格言がない場合、QUOTESシートから選ばれた神の傾向に合うものを選ぶ
    return None

def pick_quotes_by_character_tendency(dfq: pd.DataFrame, char_vow_weights: np.ndarray, vow_titles: List[str], lang: str, k: int, seed: int) -> pd.DataFrame:
    """
    選ばれた神の誓願への重みから、その神の傾向に合う格言を選ぶ
    char_vow_weights: 選ばれた神の各誓願への重み（W[obs_idx]）
    vow_titles: 誓願のタイトルリスト
    """
    d = dfq.copy()
    d["LANG"] = d["LANG"].astype(str).str.strip().str.lower()
    lang = (lang or "ja").strip().lower()
    pool = d[d["LANG"].str.contains(lang, na=False)]
    if len(pool) < k:
        pool = d  # fallback
    
    if len(pool) == 0:
        return pd.DataFrame(columns=["QUOTE_ID", "QUOTE", "SOURCE", "LANG"])
    
    rng = np.random.default_rng(seed)
    
    # 各格言を誓願のタイトルと照合してスコアを計算
    scores = np.zeros(len(pool), dtype=float)
    for i, (_, row) in enumerate(pool.iterrows()):
        quote_text = str(row.get("QUOTE", "")).lower()
        score = 0.0
        # 各誓願のタイトルが格言に含まれているか、重みを掛けてスコア化
        for j, (vow_title, weight) in enumerate(zip(vow_titles, char_vow_weights)):
            if weight > 0.1:  # 重みが大きい誓願のみ考慮
                title_lower = str(vow_title).lower()
                # タイトルのキーワードが格言に含まれているかチェック
                title_words = re.findall(r'\w+', title_lower)
                for word in title_words:
                    if len(word) >= 2 and word in quote_text:
                        score += weight * (len(word) / max(len(title_lower), 1))
        scores[i] = score
    
    # スコアが低い場合はランダムに近づける
    if scores.max() < 0.1:
        scores = rng.random(len(pool)) * 0.5
    else:
        # スコアにノイズを加えて多様性を持たせる
        scores = scores + rng.normal(0, scores.std() * 0.3, size=len(scores))
    
    # 上位k個を選ぶ
    top_indices = np.argsort(scores)[-k:][::-1]
    out = pool.iloc[top_indices].copy().reset_index(drop=True)
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
    # 初期状態：ランダムに1つだけ1にする（one-hot制約を満たす）
    x = np.zeros(n, dtype=int)
    x[rng.integers(0, n)] = 1

    for _ in range(max(10, int(sweeps))):
        # one-hot制約を維持しながら、別のインデックスに移動
        current_idx = np.where(x == 1)[0][0]
        # ランダムに別のインデックスを選ぶ
        candidate_idx = rng.integers(0, n)
        if candidate_idx == current_idx:
            continue
        
        # 現在の状態のエネルギー
        E_current = qubo_energy(Q, x)
        
        # 候補状態を作成
        x_candidate = np.zeros(n, dtype=int)
        x_candidate[candidate_idx] = 1
        
        # 候補状態のエネルギー
        E_candidate = qubo_energy(Q, x_candidate)
        
        # エネルギー差
        dE = E_candidate - E_current
        
        # メトロポリス判定
        if dE <= 0:
            x = x_candidate.copy()
        else:
            if rng.random() < np.exp(-beta * dE):
                x = x_candidate.copy()
    
    return x

def sample_distribution(Q: np.ndarray, n_samples: int, sweeps: int, beta: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    counts = np.zeros(n, dtype=int)
    energies = np.zeros(n_samples, dtype=float)

    for k in range(n_samples):
        # 各サンプルごとに少しシードをずらして独立性を確保
        sample_rng = np.random.default_rng(seed + k * 1000 + int(rng.random() * 10000))
        x = sa_sample(Q, sweeps=sweeps, beta=beta, rng=sample_rng)
        # one-hot制約により、必ず1つだけ1がある
        on = np.where(x==1)[0]
        if len(on) > 0:
            idx = int(on[0])  # one-hotなので1つだけ
        else:
            # フォールバック：エネルギーが最低のインデックスを選ぶ
            local_energies = []
            for i in range(n):
                xx = np.zeros(n, dtype=int)
                xx[i] = 1
                local_energies.append(qubo_energy(Q, xx))
            idx = int(np.argmin(local_energies))
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
    st.caption("STAGE_ID は『季節×時間の状態』です。Excel側の STAGE_TO_AXIS（優先）または STAGE_TO_VOW がある場合、誓願に「季節の流れ」を混ぜます。")
    # STAGE_IDの選択肢を動的に生成（ST_01～ST_16）
    stage_options = [f"ST_{i:02d}" for i in range(1, 17)]
    current_stage_idx = 0
    current_stage_val = st.session_state.get("stage_id", "ST_01")
    if current_stage_val in stage_options:
        current_stage_idx = stage_options.index(current_stage_val)
    stage_id = st.selectbox("STAGE_ID（手動上書き可）", options=stage_options, index=current_stage_idx, key="stage_id")
    st.slider("季節×時間の重み（0=無効 / 大きいほど影響大）", 0.0, 2.0, 1.0, 0.1, key="stage_weight")

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
sh_stage_to_axis_name, df_stage_to_axis = find_sheet(sheets, ["STAGE_TO_AXIS","STAGE2AXIS","STAGE-AXIS"])
sh_axis_dict_name, df_axis_dict = find_sheet(sheets, ["AXIS_DICT","AXISDICT","AXIS-DICT","AXIS"])
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
# STAGE_TO_AXISを優先的に使用、なければSTAGE_TO_VOWを使用
stage_vec = np.zeros(len(vow_cols), dtype=float)
current_stage_id = st.session_state.get("stage_id", "ST_01")

# AXIS_DICTを使って軸名とVOW_IDのマッピングを作成
axis_to_vow_map = {}  # key: 軸名（正規化済み）, value: vow_colsのインデックス
if df_axis_dict is not None and len(df_axis_dict) > 0:
    axis_cols = {norm_col(c): c for c in df_axis_dict.columns}
    axis_id_col = axis_cols.get("AXIS_ID") or axis_cols.get("AXIS") or axis_cols.get("AXIS_NAME") or axis_cols.get("NAME")
    vow_id_col = axis_cols.get("VOW_ID") or axis_cols.get("VOW") or axis_cols.get("VOW_INDEX")
    
    if axis_id_col and vow_id_col:
        for _, axis_row in df_axis_dict.iterrows():
            axis_name = str(axis_row.get(axis_id_col, "")).strip()
            vow_id = str(axis_row.get(vow_id_col, "")).strip()
            if axis_name and vow_id:
                # vow_colsの中で該当するVOW_IDのインデックスを探す
                for idx, vc in enumerate(vow_cols):
                    vc_norm = norm_col(vc)
                    vow_id_norm = norm_col(vow_id)
                    if vc_norm == vow_id_norm or (vc_norm.endswith(vow_id_norm) and vow_id_norm in vc_norm):
                        axis_to_vow_map[norm_col(axis_name)] = idx
                        break

# STAGE_TO_AXISを優先的に処理
if df_stage_to_axis is not None and len(df_stage_to_axis) > 0:
    scols = {norm_col(c): c for c in df_stage_to_axis.columns}
    sid = scols.get("STAGE_ID") or scols.get("STAGE") or scols.get("ID")
    if sid:
        tmp = df_stage_to_axis.copy()
        tmp[sid] = tmp[sid].astype(str).str.strip()
        row = tmp[tmp[sid] == current_stage_id]
        if len(row) > 0:
            row = row.iloc[0]
            sv = np.zeros(len(vow_cols), dtype=float)
            
            # STAGE_TO_AXISのカラムからAXIS_で始まるものを探す
            for col in df_stage_to_axis.columns:
                col_norm = norm_col(col)
                # AXIS_で始まるカラムを探す（例：AXIS_SE, AXIS_RYU, AXIS_MA, AXIS_MAKOTO）
                if col_norm.startswith("AXIS_"):
                    axis_name = col_norm.replace("AXIS_", "").strip()
                    # AXIS_DICTから対応するVOW_IDのインデックスを取得
                    if axis_name in axis_to_vow_map:
                        vow_idx = axis_to_vow_map[axis_name]
                        axis_value = float(row.get(col, 0.0) or 0.0)
                        sv[vow_idx] = axis_value
            
            stage_vec = sv
elif df_stage_to_vow is not None and len(df_stage_to_vow) > 0:
    # STAGE_TO_AXISがない場合はSTAGE_TO_VOWを使用
    scols = {norm_col(c): c for c in df_stage_to_vow.columns}
    sid = scols.get("STAGE_ID") or scols.get("STAGE") or scols.get("ID")
    if sid:
        tmp = df_stage_to_vow.copy()
        tmp[sid] = tmp[sid].astype(str).str.strip()
        row = tmp[tmp[sid] == current_stage_id]
        if len(row) > 0:
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
main_col, right_col = st.columns([2.4, 1.0], gap="large")

with main_col:
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
        m = re.search(r"(\d{1,2})", norm_col(c))
        idx = int(m.group(1)) if m else 0
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

# blend stage (STAGE_TO_AXIS or STAGE_TO_VOWの影響を明確に)
# 係数を1.0に変更して、季節×時間の影響を明確にする
stage_weight = float(st.session_state.get("stage_weight", 1.0))
mix_vec2 = mix_vec + stage_weight * stage_vec

# ============================================================
# Step2: 誓願ベクトル（manual/auto/mix）テーブル表示（メインカラム内）
# ============================================================
with main_col:
    st.markdown("## Step 2：誓願ベクトル（manual/auto/mix）")
    df_vow_vec = pd.DataFrame({
        "VOW_ID": [f"VOW_{int(m.group(1)):02d}" if (m := re.search(r'(\d{1,2})', norm_col(c))) else norm_col(c) for c in vow_cols],
        "TITLE": [df_vows.iloc[i]["TITLE"] if i < len(df_vows) else "" for i in range(len(vow_cols))],
        "manual(0-5)": slider_vec,
        "auto(0-5)": text_vec,
        "stage(0-5)": stage_vec * stage_weight,
        "mix(0-5)": mix_vec2
    })
    # HTMLテーブルとして表示（白文字・黒背景で確実に）
    html_table = render_dataframe_as_html_table(df_vow_vec)
    st.markdown(html_table, unsafe_allow_html=True)
    
    # 観測ボタン
    if st.button("観測する（QUBOから抽出）", type="primary", use_container_width=True):
        st.session_state["observe_triggered"] = True

# ============================================================
# QUBO calculation (outside columns for use in visualizations)
# ============================================================
# score per character = dot(mix_vec2, W_char) + noise
W = dfW[vow_cols].values.astype(float)  # shape (n_char, n_vow)
base_scores = (W @ mix_vec2.reshape(-1,1)).reshape(-1)

# シードをスライダーの値も含めて計算（スライダー変更で結果が変わるように）
slider_sum = float(slider_vec.sum())
text_hash = make_seed(user_text)
rng = np.random.default_rng(make_seed(f"{user_text}|{slider_sum:.2f}|{text_hash}"))
eps = float(st.session_state.get("eps_noise",0.08))
noisy_scores = base_scores + rng.normal(0, eps, size=len(base_scores))

# energies: lower is better
energies = -noisy_scores

# QUBO one-hot
P = float(st.session_state.get("P", 40.0))
Q = build_qubo_onehot(scores=noisy_scores, P=P)

# シードをスライダーの値も含めて計算
qubo_seed = make_seed(f"{user_text}|{slider_sum:.2f}|{mix_vec2.sum():.2f}|qubo")
prob, sampleE = sample_distribution(
    Q,
    n_samples=int(st.session_state.get("n_samples",300)),
    sweeps=int(st.session_state.get("sweeps",420)),
    beta=float(st.session_state.get("beta",2.2)),
    seed=qubo_seed,
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

# ============================================================
# 可視化セクション（メインカラム内）
# ============================================================
with main_col:
    st.markdown("## 可視化：テキストの影響・観測分布・エネルギー地形")
    
    viz_col1, viz_col2 = st.columns(2, gap="large")
    
    with viz_col1:
        st.markdown("### 1) テキスト→誓願 自動推定の影響（auto vs manual vs mix）")
        st.caption("auto(テキスト由来) と manual (スライダー) とmixの差が見える化されます。")
        
        # 可視化用データ
        vow_indices = np.arange(len(vow_cols))
        fig_text_influence = go.Figure()
        fig_text_influence.add_trace(go.Scatter(
            x=vow_indices,
            y=slider_vec,
            mode='lines+markers',
            name='manual (スライダー)',
            line=dict(color='rgba(100,200,255,0.8)', width=2),
            marker=dict(size=8)
        ))
        fig_text_influence.add_trace(go.Scatter(
            x=vow_indices,
            y=text_vec,
            mode='lines+markers',
            name='auto (テキスト)',
            line=dict(color='rgba(255,150,100,0.8)', width=2),
            marker=dict(size=8)
        ))
        fig_text_influence.add_trace(go.Scatter(
            x=vow_indices,
            y=mix_vec2,
            mode='lines+markers',
            name='mix',
            line=dict(color='rgba(150,255,150,0.8)', width=2),
            marker=dict(size=8)
        ))
        fig_text_influence.update_layout(
            xaxis=dict(title="誓願インデックス", tickmode='linear', tick0=0, dtick=1),
            yaxis=dict(title="値 (0-5)", range=[0, 5.5]),
            height=350,
            paper_bgcolor="rgba(6,8,18,1)",
            plot_bgcolor="rgba(6,8,18,0.5)",
            font=dict(color="rgba(245,245,255,0.92)"),
            legend=dict(bgcolor="rgba(10,12,26,0.8)", bordercolor="rgba(255,255,255,0.1)")
        )
        st.plotly_chart(fig_text_influence, use_container_width=True)
    
    with viz_col2:
        st.markdown("### 3) 観測分布（サンプル）")
        st.caption("同条件で複数回観測した場合の分布です。")
        
        # 観測分布の可視化
        char_names = df_chars["神"].astype(str).tolist()
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            x=char_names,
            y=prob,
            marker=dict(color='rgba(100,150,255,0.7)', line=dict(color='rgba(100,150,255,0.9)', width=1)),
            text=[f"{p:.3f}" for p in prob],
            textposition='outside'
        ))
        fig_dist.update_layout(
            xaxis=dict(title="神", tickangle=-45),
            yaxis=dict(title="確率"),
            height=350,
            paper_bgcolor="rgba(6,8,18,1)",
            plot_bgcolor="rgba(6,8,18,0.5)",
            font=dict(color="rgba(245,245,255,0.92)")
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # エネルギー地形の可視化
    st.markdown("### 2) エネルギー地形（低いほど選ばれやすい）")
    fig_energy = go.Figure()
    fig_energy.add_trace(go.Bar(
        x=df_rank["神"].head(12).tolist(),
        y=df_rank["energy（低いほど選ばれやすい）"].head(12).tolist(),
        marker=dict(
            color=df_rank["energy（低いほど選ばれやすい）"].head(12).tolist(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="エネルギー")
        ),
        text=[f"{e:.3f}" for e in df_rank["energy（低いほど選ばれやすい）"].head(12).tolist()],
        textposition='outside'
    ))
    fig_energy.update_layout(
        xaxis=dict(title="神", tickangle=-45),
        yaxis=dict(title="エネルギー（低いほど選ばれやすい）"),
        height=400,
        paper_bgcolor="rgba(6,8,18,1)",
        plot_bgcolor="rgba(6,8,18,0.5)",
        font=dict(color="rgba(245,245,255,0.92)")
    )
    st.plotly_chart(fig_energy, use_container_width=True)
    
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
            st.info("入力が短い/空のため、キーワードが抽出できません（2文字以上の語が必要です）。")
    
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
# Step3 (right): QUBO observe
# ============================================================
with right_col:
    st.markdown("## Step 3：結果（観測された神＋理由＋QUOTES神託）")

    # observed: sample from prob (single observation)
    obs_idx = int(np.argmax(prob)) if prob.sum() > 0 else int(np.argmin(energies))
    obs_char = char_ids[obs_idx]
    obs_god = str(df_chars.loc[df_chars["CHAR_ID"]==obs_char, "神"].values[0]) if (df_chars["CHAR_ID"]==obs_char).any() else obs_char

    # reason (Top VOW contributions) - calculate first
    contrib = mix_vec2 * W[obs_idx]  # elementwise
    df_top = pd.DataFrame({
        "VOW": [f"VOW_{int(m.group(1)):02d}" if (m := re.search(r'(\d{1,2})', norm_col(c))) else norm_col(c) for c in vow_cols],
        "TITLE": [df_vows.iloc[i]["TITLE"] if i < len(df_vows) else "" for i in range(len(vow_cols))],
        "mix(v)": mix_vec2,
        "W(char,v)": W[obs_idx],
        "寄与(v*w)": contrib
    }).sort_values("寄与(v*w)", ascending=False).reset_index(drop=True)

    # QUOTES (temperature) - calculate first
    qpick_temp = pick_quotes_by_temperature(
        dfQ,
        lang=st.session_state.get("lang","ja"),
        k=3,
        tau=float(st.session_state.get("quote_tau",1.2)),
        seed=make_seed(user_text + "|quotes_temp"),
    )
    
    # 選ばれた神の「役割（CHAR_MASTER）」×「寄与Top誓願（VOW）」で神託文を生成
    stage_label = st.session_state.get("stage_id","ST_01")
    oracle_green = build_oracle_from_char_master_and_vows(
        char_master=df_char_master,
        char_id=obs_char,
        god_name=obs_god,
        df_top_vows=df_top,
        stage_id=stage_label,
    )

    # 1. 一番上：今回「観測」された神の画像
    st.markdown(f'### 🌟 今回「観測」された神：{obs_god}（{obs_char}）')
    
    img_path = get_char_image_path(obs_char, st.session_state.get("img_folder","./assets/images/characters"))
    if img_path and Path(img_path).exists():
        st.image(img_path, use_container_width=True, caption=f"{obs_god}（{Path(img_path).name}）")
    else:
        st.warning(
            f"キャラクター画像が見つかりません（探索フォルダ: {st.session_state.get('img_folder')} / CHAR_ID: {obs_char}）\n"
            f"※ assets/images/characters 配下のファイル名に CHAR_01 か 1 などが含まれるようにしてください。"
        )
    
    st.markdown(
        "<div style='background:rgba(20,30,50,0.4); padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.15); margin-top:8px; margin-bottom:16px;'>"
        "<span style='color:rgba(245,245,255,0.95); font-size:0.9em; line-height:1.6;'>"
        "※ここは「単発の観測（1回抽選）」です。下の観測分布（サンプル）は「同条件で何回も観測したらどう出るか」のヒストグラムです。そのため、分布の最多と単発の観測結果が一致しないことがあります（正常挙動）。"
        "</span></div>",
        unsafe_allow_html=True
    )

    # 2. その下：選ばれた理由を表で表示
    st.markdown("### 選ばれた理由（寄与した誓願 Top）")
    # HTMLテーブルとして表示（白文字・黒背景で確実に）
    html_table_top = render_dataframe_as_html_table(df_top, max_rows=6)
    st.markdown(html_table_top, unsafe_allow_html=True)

    # 3. その下：格言を2種類表示
    # 3-1. 選ばれた神の神託（緑）：CHAR_MASTERの役割文言 + 寄与Top誓願
    st.markdown("### 選ばれた神の神託（役割×寄与Top）")
    st.markdown(
        "<div style='background:rgba(40,120,80,0.40); border:1px solid rgba(80,200,140,0.60); padding:24px; border-radius:12px; color:rgba(245,255,250,1); line-height:1.9; margin-top:12px; box-shadow: 0 4px 20px rgba(40,120,80,0.3); white-space:pre-wrap;'>"
        f"{oracle_green}"
        "</div>",
        unsafe_allow_html=True
    )
    
    # 3-2. QUOTES神託から温度付きで選んだ格言（青色）
    st.markdown("### QUOTES神託（温度付きで選択）")
    if len(qpick_temp) > 0:
        qt_temp = str(qpick_temp.loc[0,"QUOTE"])
        src_temp = str(qpick_temp.loc[0,"SOURCE"]) if "SOURCE" in qpick_temp.columns else "—"
        st.markdown(
            f"<div style='background:rgba(40,90,160,0.4); border:1px solid rgba(100,170,255,0.6); padding:24px; border-radius:12px; color:rgba(245,255,250,1); line-height:1.9; margin-top:12px; box-shadow: 0 4px 20px rgba(40,90,160,0.3);'>"
            f"<div style='color:rgba(200,230,255,1); font-size:1.15em; font-weight:500; margin-bottom:10px; line-height:1.8;'>『{qt_temp}』</div>"
            f"<div style='color:rgba(180,200,255,0.95); font-size:0.95em; margin-top:14px; font-style:italic;'>— {src_temp}</div></div>",
            unsafe_allow_html=True
        )
    else:
        st.info("QUOTES神託が見つかりませんでした。")
    
    # Debug expander for quotes
    with st.expander("格言候補Top (デバッグ)", expanded=False):
        if len(qpick_temp) > 0:
            st.markdown(
                "<div style='color:rgba(245,245,255,0.95);'><strong>QUOTES神託（温度付き）候補:</strong></div>",
                unsafe_allow_html=True
            )
            html_table_quotes = render_dataframe_as_html_table(qpick_temp[["QUOTE", "SOURCE"]])
            st.markdown(html_table_quotes, unsafe_allow_html=True)
    
    # ランキング表（参考用、折りたたみ可能に）
    with st.expander("全キャラクターランキング（参考）", expanded=False):
        html_table_rank = render_dataframe_as_html_table(df_rank.head(10))
        st.markdown(html_table_rank, unsafe_allow_html=True)


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
        "STAGE_TO_AXIS": sh_stage_to_axis_name,
        "AXIS_DICT": sh_axis_dict_name,
        "QUOTES": sh_quotes_name,
    })
    if df_axis_dict is not None and len(df_axis_dict) > 0:
        st.write("AXIS_DICT columns:")
        st.write(list(df_axis_dict.columns))
        st.write("AXIS_DICT データ:")
        st.write(df_axis_dict.to_dict('records'))
        st.write("axis_to_vow_map（軸名→VOWインデックスのマッピング）:")
        st.write(axis_to_vow_map)
    if df_stage_to_axis is not None and len(df_stage_to_axis) > 0:
        st.write("STAGE_TO_AXIS columns:")
        st.write(list(df_stage_to_axis.columns))
        st.write("STAGE_TO_AXIS データ（現在のSTAGE_IDに該当する行）:")
        current_stage_id = st.session_state.get("stage_id", "ST_01")
        scols = {norm_col(c): c for c in df_stage_to_axis.columns}
        sid = scols.get("STAGE_ID") or scols.get("STAGE") or scols.get("ID")
        if sid:
            tmp = df_stage_to_axis.copy()
            tmp[sid] = tmp[sid].astype(str).str.strip()
            row = tmp[tmp[sid] == current_stage_id]
            if len(row) > 0:
                st.write(row.to_dict('records'))
        st.write("stage_vec（現在適用されている値）:")
        st.write(stage_vec.tolist())
    elif df_stage_to_vow is not None and len(df_stage_to_vow) > 0:
        st.write("STAGE_TO_VOW columns:")
        st.write(list(df_stage_to_vow.columns))
        st.write("stage_vec（現在適用されている値）:")
        st.write(stage_vec.tolist())
    st.write("CHAR_TO_VOW columns:")
    st.write(list(df_char_to_vow.columns))
    if df_char_master is not None and len(df_char_master) > 0:
        st.write("CHAR_MASTER columns:")
        st.write(list(df_char_master.columns))
        st.write("CHAR_MASTER 先頭行（サンプル）:")
        st.write(df_char_master.head(1).to_dict('records'))
    st.write("検出された VOW列:", vow_cols)
    st.write("画像フォルダ:", st.session_state.get("img_folder"))
    st.write("スキャン結果（一部）:", dict(list(scan_character_images(st.session_state.get("img_folder")).items())[:10]))
