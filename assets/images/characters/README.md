# キャラクター画像フォルダー

このフォルダーには、12神のキャラクター画像（.pngファイル）を配置します。

## 📁 ファイル命名規則

12神のキャラクター画像は、以下の命名規則に従って配置してください：

### 推奨命名規則

```
character_01.png  # 1番目の神
character_02.png  # 2番目の神
character_03.png  # 3番目の神
...
character_12.png  # 12番目の神
```

### または、神の名前を使用

```
akiba_01.png
akiba_02.png
...
akiba_12.png
```

### または、Excelファイルの順序に合わせる

Excelファイル（`akiba12_character_list.xlsx`）の順序に合わせて命名：

```
character_01.png  # Excelの1行目（最初の神）
character_02.png  # Excelの2行目
...
character_12.png  # Excelの12行目（最後の神）
```

## 📋 画像ファイルの要件

- **形式**: PNG（推奨）またはJPG
- **サイズ**: 推奨サイズ 512x512px 以上（正方形推奨）
- **背景**: 透明背景（PNG）または単色背景
- **ファイル名**: 英数字とアンダースコアのみ（日本語不可）

## 🔗 キャラクターと画像の対応

キャラクターと画像の対応は、Excelファイル（`akiba12_character_list.xlsx`）の順序に従います：

1. Excelファイルの1行目 → `character_01.png`
2. Excelファイルの2行目 → `character_02.png`
3. ...
4. Excelファイルの12行目 → `character_12.png`

## 📝 使用方法

### Streamlitアプリでの表示

`app.py`で画像を表示する場合：

```python
import streamlit as st
from pathlib import Path

# 画像パスの取得
image_path = Path("assets/images/characters/character_01.png")

# 画像の表示
if image_path.exists():
    st.image(str(image_path), width=300)
```

### キャラクター選択時の画像表示

`app.py`の`oracle_card`関数などで、選択されたキャラクターの画像を表示できます。

## ⚠️ 注意事項

- 画像ファイルはGitHubに含まれます（`.gitignore`で除外されていません）
- ファイルサイズが大きい場合は、最適化を推奨します
- 著作権に注意してください（チームメンバーが作成した画像を使用）

## 🎨 画像の追加方法

1. このフォルダー（`assets/images/characters/`）に画像ファイルを配置
2. 命名規則に従ってファイル名を設定
3. `app.py`で画像を読み込むコードを追加（必要に応じて）

## 📚 関連ドキュメント

- `README.md` - プロジェクトのメイン説明
- `EXCEL_SETUP_GUIDE.md` - Excel設定ファイルのガイド
- `app.py` - Streamlitアプリのメインファイル