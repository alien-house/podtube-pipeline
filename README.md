# podtube-pipeline

YouTubeの動画を自動収集・文字起こし・AI解析してDBに保存するパイプライン。

## アーキテクチャ

```
キーワード検索 / チャンネル指定（Supabase管理）
　↓
crawl.py（crontab 毎時実行）
　├─ YouTube Data API で動画を収集
　├─ ルールベースフィルタリング（BGM・実況・Shorts等を除外）
　└─ Gemini でタイトルプリスクリーン → Supabaseに保存
　↓
analyze.py（PM2で常駐）
　├─ 字幕あり → Gemini で解析
　└─ 字幕なし → yt-dlp でダウンロード（並列）
　　　　　　　　→ faster-whisper で文字起こし
　　　　　　　　→ Ollama（gemma4-ctx65k）で解析
　↓
Supabaseに分析結果を保存
```

## 設計の工夫

**動画尺による処理の自動切り替え**
- 2時間未満：全文一括モード（gemma4-ctx65k, num_ctx 65536）
- 2〜3時間：時間窓モード（300秒ごとにチャンク分割して解析）
- 3時間超：スキップ

**並列ダウンロード × 直列GPU処理**
- `ThreadPoolExecutor` でyt-dlpのダウンロードを並列化（ネットワークI/Oを効率化）
- Whisper / Ollamaは直列実行（VRAM競合を回避）
- Bot対策としてDL間に3〜8秒のランダムスリープを挿入

**段階的バックオフ**
- 新着なしのチャンネルは6時間→30日と段階的にクロール間隔を伸ばしクォータを節約

**多層フィルタリング**
- ルールベース（正規表現）で明らかなノイズを除外
- Geminiのタイトルプリスクリーンでトーク系でない動画を早期除外
- GPU解析前にOllamaでタイトル判定（Whisper実行コストを削減）

**JSONL台帳**
- 全処理結果を`results.jsonl`に記録
- `jq` / `pandas`で成功率・失敗原因を集計可能

## 使用技術

| 技術 | 用途 |
|---|---|
| Python | メインスクリプト |
| YouTube Data API | 動画収集 |
| Gemini API | タイトルプリスクリーン・字幕あり動画の解析 |
| yt-dlp | 音声ダウンロード |
| faster-whisper | 音声文字起こし（GPU対応） |
| Ollama（gemma4-ctx65k） | ローカルLLMによる動画解析 |
| Supabase | データベース |
| PM2 | analyze.pyの常駐管理 |

## セットアップ

```bash
# 1. 依存パッケージをインストール
pip install -r requirements.txt

# 2. 環境変数を設定
cp .env.example .env.local
# .env.localを編集

# 3. Ollamaのモデルを準備
ollama pull gemma4:e4b
ollama create gemma4-ctx65k -f Modelfile
# Modelfileの内容:
# FROM gemma4
# PARAMETER num_ctx 65536

# 4. crawl.pyをcrontabに登録（毎時0分）
crontab -e
# 0 * * * * /usr/bin/python3 /path/to/crawl.py

# 5. analyze.pyをPM2で常駐起動
pm2 start analyze.py --interpreter python3
```