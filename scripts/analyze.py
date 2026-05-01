"""
pod tube — ローカル LLM 解析（唯一のエントリポイント: analyze.py）
=================================================================

① Supabase から未解析動画を取得
   - 2 時間未満  : 全文一括モード（gemma4-ctx65k）
   - 2〜3 時間   : 時間窓モード（従来の分析）
   - 3 時間超    : スキップ
② yt-dlp で音声ダウンロード（バッチ内は ThreadPoolExecutor で並列）
③ faster-whisper で全セグメント文字起こし（GPU は直列占有）
   [全文モード] 全セグメントを結合 → Ollama に一括投入してチャプター JSON を生成
   [窓モード]  時間窓ごとに行を間引き → Ollama(/api/chat) で各窓のチャプター JSON
④ チャプター要約から本番互換の talk 系メタ（feed_eligible 等）を 2 パス目で推定しマージ
⑤ Supabase UPDATE → 音声削除

実行例:
  cd scripts/local_analyzer && pip install -r requirements.txt && python analyze.py

依存: requirements.txt / ollama pull gemma4:e4b
  全文モード使用時は事前に下記を実行してください:
    ollama create gemma4-ctx65k -f Modelfile
  （Modelfile の内容: FROM gemma4 / PARAMETER num_ctx 65536）

環境変数:
  SUPABASE_URL, SUPABASE_SERVICE_KEY
  OLLAMA_MODEL, OLLAMA_HOST, OLLAMA_TIMEOUT, OLLAMA_MAX_RETRIES
  OLLAMA_INFER_NUM_PREDICT（2 パス目のトークン上限）
  FULLTEXT_MODE          : 後方互換のため残存（現在は尺で自動判別のため無効）
  FULLTEXT_MODEL         : 全文モード用モデル名（デフォルト: gemma4-ctx65k）
  FULLTEXT_MAX_DURATION_SEC : 全文モード上限秒数（デフォルト: 7200 = 2時間）
  FULLTEXT_MAX_CHARS     : Ollama に渡す最大文字数（デフォルト: 80000）
  CHAPTER_WINDOW_SECONDS, CHAPTER_COMPRESS_STRIDE, CHAPTER_COMPRESS_MAX_CHARS
  FAILED_RESULTS_DIR, WHISPER_MODEL, WHISPER_DEVICE, BATCH_SIZE, AUDIO_DIR
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import random
import re
import subprocess
import time
import traceback as _tb_mod
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from supabase import create_client, Client

from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# ─────────────────────────────────────────────────────────────
# ログ設定
#   - stdout  : PM2 が pm2-out.log に捕捉
#   - ファイル: ./logs/analyze.log（10MB × 5 世代ローテート）
#   - エラー  : ./logs/analyze-error.log（WARNING 以上のみ）
#   - 台帳    : ./logs/results.jsonl（成功/失敗 1 件 1 行 JSON）
# ─────────────────────────────────────────────────────────────
LOGS_DIR = Path(os.getenv("LOGS_DIR", "./logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LEDGER_FILE = LOGS_DIR / "results.jsonl"

_log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ルートロガーに手動でハンドラを追加（basicConfig を使わない）
_root = logging.getLogger()
_root.setLevel(logging.INFO)
# 既存のハンドラがあれば外す（多重起動時の二重出力を防ぐ）
for _h in list(_root.handlers):
    _root.removeHandler(_h)

_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(logging.INFO)
_stream_handler.setFormatter(_log_formatter)
_root.addHandler(_stream_handler)

_file_handler = logging.handlers.RotatingFileHandler(
    LOGS_DIR / "analyze.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_log_formatter)
_root.addHandler(_file_handler)

_err_handler = logging.handlers.RotatingFileHandler(
    LOGS_DIR / "analyze-error.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_err_handler.setLevel(logging.WARNING)
_err_handler.setFormatter(_log_formatter)
_root.addHandler(_err_handler)

log = logging.getLogger(__name__)


def _write_ledger(
    video: dict,
    status: str,          # "success" | "skip" | "failure"
    reason: str,          # 機械可読なコード (例: "audio_download_failed")
    elapsed_sec: float,
    extra: dict | None = None,
) -> None:
    """成功／失敗に関わらず 1 動画につき 1 行を ./logs/results.jsonl へ追記。
    jq / grep / pandas で集計できる JSONL フォーマット。"""
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason": reason,
        "elapsed_sec": round(float(elapsed_sec), 2),
        "youtube_id": video.get("youtube_id"),
        "video_id": video.get("id"),
        "title": (video.get("title") or "")[:120],
    }
    if extra:
        record.update(extra)
    try:
        with LEDGER_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # 台帳書き込みに失敗しても本処理は止めない
        log.error(f"  ledger 書き込み失敗: {e}")

SUPABASE_URL        = os.environ["SUPABASE_URL"]
SUPABASE_KEY        = os.environ["SUPABASE_SERVICE_KEY"]
OLLAMA_HOST         = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_TIMEOUT      = float(os.getenv("OLLAMA_TIMEOUT", "600"))
OLLAMA_MAX_RETRIES  = max(1, int(os.getenv("OLLAMA_MAX_RETRIES", "3")))
OLLAMA_INFER_NUM_PREDICT = max(256, int(os.getenv("OLLAMA_INFER_NUM_PREDICT", "512")))
CHAPTER_WINDOW_SECONDS = float(os.getenv("CHAPTER_WINDOW_SECONDS", "300"))
CHAPTER_COMPRESS_STRIDE = max(1, int(os.getenv("CHAPTER_COMPRESS_STRIDE", "10")))
CHAPTER_COMPRESS_MAX_CHARS = max(200, int(os.getenv("CHAPTER_COMPRESS_MAX_CHARS", "2000")))

# ── 全文一括モード ────────────────────────────────────────────────────────
# FULLTEXT_MODE=true のとき 1 時間未満の動画を全文で一括解析する
FULLTEXT_MODE      = os.getenv("FULLTEXT_MODE", "true").strip().lower() in ("1", "true", "yes")
FULLTEXT_MODEL     = os.getenv("FULLTEXT_MODEL", "gemma4-ctx65k")  # num_ctx 65536 設定済みモデル
FULLTEXT_MAX_CHARS = int(os.getenv("FULLTEXT_MAX_CHARS", "80000"))  # 安全上限（文字数）
FULLTEXT_MAX_DURATION_SEC = 2 * 60 * 60                                  # 2 時間未満: 全文一括, 以上: 時間窓
FAILED_RESULTS_DIR   = Path(os.getenv("FAILED_RESULTS_DIR", "./failed_results"))
FAILED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WHISPER_MODEL_SIZE  = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE      = os.getenv("WHISPER_DEVICE", "cpu")
BATCH_SIZE          = int(os.getenv("BATCH_SIZE", "5"))
AUDIO_DIR           = Path(os.getenv("AUDIO_DIR", "./podtube_audio"))
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

MAIN_FEED_MAX_DURATION_SEC = 3 * 60 * 60

CHAPTER_CHUNK_SYSTEM_PROMPT = """あなたは動画に目次（チャプター）を付ける編集者です。
与えられるのは、その時間帯の音声文字起こしを「行おきに間引いた」断片だけです。
語が飛び飛びでも、動画タイトルと断片から内容を推定してください。

必ず次のキーだけを含む JSON を返す（マークダウンやコードブロック禁止）:
- chapter_title   : 22 文字以内の見出し
- chapter_summary : 50 文字以内の 1 行説明
- topic_tags      : 2〜6 文字の名詞を最大 5 個の配列
- inferred_theme  : 大まかな分類を 1 語（雑談・投資・恋愛・芸能・スポーツ・料理 など）
- is_highlight    : この時間帯が動画全体の「見どころ・盛り上がり・重要な話題」なら true、そうでなければ false

不明な場合も推測して必ずすべてのキーを埋めてください。"""

# チャプター要約 → 本番 AiMetadata 互換の talk 系フィールドのみ
INFER_STANDARD_FIELDS_PROMPT = """あなたは動画メタデータの推定専門家です。
入力は「動画タイトル」と「チャプター目次の要約（JSON テキスト）」のみです。全文トランスクリプトはありません。
チャプター内容から推定し、次のキーだけを含む JSON を返す（マークダウン禁止）:

- feed_eligible: boolean
  true = 会話・解説・雑談・実況トークが主役。false = 作業BGM・無言ゲーム中心・実況が極端に薄い等。
- talk_purity: 0.0-1.0（BGM・SE が少なく声が主なら高め）
- talk_density: 0.0-1.0（チャプター全体として喋りの割合が高そうなら高め）
- speaker_info: { gender: "男"|"女"|"混合", age_group: "10代"|"20代"|"30-40代"|"50代以上", count: 整数 }
- emotional_scores: { laughter, moving, sadness, joy: 各 0.0-1.0 }

推測で構いません。すべて埋めてください。"""

_GENDER_OK = frozenset({"男", "女", "混合"})
_AGE_OK = frozenset({"10代", "20代", "30-40代", "50代以上"})


def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def iso8601_duration_seconds(iso: str | None) -> int:
    if not iso:
        return 0
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not m:
        return 0
    return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + int(m.group(3) or 0)


def is_within_main_feed_duration(iso: str | None) -> bool:
    sec = iso8601_duration_seconds(iso)
    if sec <= 0:
        return True
    return sec < MAIN_FEED_MAX_DURATION_SEC


def fetch_unanalyzed(supabase: Client, limit: int) -> list[dict]:
    """ai_metadata が NULL または {}（空オブジェクト）の動画を新しい順に取得。

    背景: 全動画は先行で Gemini（字幕ベース）が解析するため analyzed=true になる。
    ただし字幕が無い動画は Gemini でメタデータを作れず ai_metadata が null / {} で残る。
    → そこだけを Whisper + Gemma で解析する、というフロー。
    """
    fetch_n = min(max(limit * 8, limit), 200)
    res = (
        supabase.table("videos")
        .select("id, youtube_id, title, duration, ai_metadata")
        .or_("ai_metadata.is.null,ai_metadata.eq.{}")
        .order("published_at", desc=True)
        .limit(fetch_n)
        .execute()
    )
    rows = res.data or []
    out: list[dict] = []
    for row in rows:
        # Python 側でも二重チェック（PostgREST の or フィルタの取りこぼし対策）
        md = row.get("ai_metadata")
        if md is not None and md != {}:
            continue
        dur_sec = iso8601_duration_seconds(row.get("duration"))
        if not is_within_main_feed_duration(row.get("duration")):
            log.info(
                "  キューから除外（尺 %ds ≥ %dh）: %s",
                dur_sec,
                MAIN_FEED_MAX_DURATION_SEC // 3600,
                row.get("youtube_id"),
            )
            continue
        out.append(row)
        if len(out) >= limit:
            break
    return out


def download_audio(youtube_id: str, output_dir: Path) -> Path | None:
    out_path = output_dir / f"{youtube_id}.mp3"
    if out_path.exists():
        log.info(f"  音声キャッシュ使用: {out_path.name}")
        return out_path

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    cmd = [
        "yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "9",
        "--no-playlist", "--quiet",
        "--remote-components", "ejs:github",
        "--cookies", "./www.youtube.com_cookies.txt",
        "--js-runtimes", "node",
        "--output", str(output_dir / f"{youtube_id}.%(ext)s"), url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log.warning(f"  yt-dlp 失敗: {result.stderr[:200]}")
            return None
        candidates = list(output_dir.glob(f"{youtube_id}.*"))
        if not candidates:
            log.warning(f"  音声ファイルが見つかりません: {youtube_id}")
            return None
        for c in candidates:
            if c.suffix == ".mp3":
                return c
        return candidates[0]
    except subprocess.TimeoutExpired:
        log.warning(f"  yt-dlp タイムアウト: {youtube_id}")
        return None
    except Exception as e:
        log.warning(f"  yt-dlp エラー: {e}")
        return None


_whisper_model: WhisperModel | None = None


def get_whisper() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        log.info(f"Whisper モデルをロード: {WHISPER_MODEL_SIZE} ({WHISPER_DEVICE})")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type="int8",
        )
    return _whisper_model


@dataclass(frozen=True)
class TimedSegment:
    start: float
    end: float
    text: str


def _fmt_ts(sec: float) -> str:
    s = int(max(0.0, sec))
    h, r = divmod(s, 3600)
    m, s2 = divmod(r, 60)
    if h:
        return f"{h:d}:{m:02d}:{s2:02d}"
    return f"{m:d}:{s2:02d}"


def transcribe_segments(audio_path: Path) -> list[TimedSegment]:
    model = get_whisper()
    segments, info = model.transcribe(
        str(audio_path),
        language="ja",
        beam_size=1,
        vad_filter=True,
    )
    log.info(f"  言語: {info.language} (確率 {info.language_probability:.2f})")
    out: list[TimedSegment] = []
    for seg in segments:
        t = seg.text.strip()
        if not t:
            continue
        out.append(TimedSegment(start=float(seg.start), end=float(seg.end), text=t))
    log.info(f"  Whisper セグメント数: {len(out)}")
    return out


def bucket_timed_segments(segs: list[TimedSegment], window_sec: float) -> list[list[TimedSegment]]:
    if not segs or window_sec <= 0:
        return []
    mb = max(int(s.start // window_sec) for s in segs)
    buckets: list[list[TimedSegment]] = [[] for _ in range(mb + 1)]
    for s in segs:
        b = int(s.start // window_sec)
        if b < 0:
            b = 0
        while b >= len(buckets):
            buckets.append([])
        buckets[b].append(s)
    return [b for b in buckets if b]


def lines_from_window(window: list[TimedSegment]) -> list[str]:
    return [f"[{_fmt_ts(s.start)}] {s.text}" for s in window]


def compress_stride(lines: list[str], every: int, max_chars: int) -> str:
    if not lines:
        return ""
    stride = max(1, every)
    while True:
        picked = lines[::stride]
        text = "\n".join(picked)
        if len(text) <= max_chars or stride >= len(lines):
            break
        stride = min(stride * 2, len(lines))
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


# ── 全文一括モード用プロンプト ─────────────────────────────────────────────
FULLTEXT_CHAPTER_SYSTEM_PROMPT = """あなたは動画に目次（チャプター）を付ける編集者です。
与えられるのは動画の全文文字起こし（タイムスタンプ付き）です。

必ず次のキーだけを含む JSON を返す（マークダウンやコードブロック禁止）:
- chapters: 以下の構造を持つオブジェクトの配列
    - chapter_title   : 22 文字以内の見出し
    - chapter_summary : 50 文字以内の 1 行説明
    - topic_tags      : 2〜6 文字の名詞を最大 5 個の配列
    - inferred_theme  : 大まかな分類を 1 語（雑談・投資・恋愛・芸能・スポーツ・料理 など）
    - start_time      : そのチャプターの開始タイムスタンプ（文字起こし中の [mm:ss] 形式をそのまま）
    - end_time        : そのチャプターの終了タイムスタンプ
    - is_highlight    : このチャプターが動画の「見どころ・盛り上がり・重要な話題」なら true、そうでなければ false

内容の区切れ目ごとにチャプターを分けてください（目安: 5〜15 分に 1 チャプター）。
不明な場合も推測して必ずすべてのキーを埋めてください。
is_highlight は動画全体を通じて true が 2〜4 個になるよう選んでください（全部 true や全部 false は避ける）。"""

# 全文トランスクリプトから通し要約を生成するプロンプト
FULLTEXT_SUMMARY_SYSTEM_PROMPT = """あなたは動画コンテンツの要約専門家です。
与えられるのは動画の全文文字起こし（タイムスタンプ付き）です。
動画全体の内容を 250〜300 字程度の自然な日本語で要約してください。

【ルール】
- 章立て・番号・箇条書きは使わず、流れるような文章で書く
- 話者の名前・固有名詞はできるだけ残す
- 「この動画では」など前置き不要、内容を直接書く
- 要約文のみ出力（説明・前置き不要）"""


def fulltext_from_segments(segments: list[TimedSegment]) -> str:
    """全セグメントをタイムスタンプ付きテキストに結合。
    FULLTEXT_MAX_CHARS を超える場合は末尾を切り捨て（安全弁）。"""
    lines = [f"[{_fmt_ts(s.start)}] {s.text}" for s in segments]
    text = "\n".join(lines)
    if len(text) > FULLTEXT_MAX_CHARS:
        log.warning(
            f"  全文テキスト {len(text)} 文字 > 上限 {FULLTEXT_MAX_CHARS} 文字 → 切り捨て"
        )
        text = text[:FULLTEXT_MAX_CHARS]
    return text


def analyze_fulltext_with_ollama(title: str, fulltext: str) -> list[dict] | None:
    """全文字起こしを Ollama に一括投入してチャプターリストを返す。失敗時は None。"""
    user_content = f"動画タイトル: {title}\n\n全文文字起こし:\n{fulltext}"
    payload = {
        "model": FULLTEXT_MODEL,
        "think": False,
        "messages": [
            {"role": "system", "content": FULLTEXT_CHAPTER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "num_predict": 4096,
            "think": False,
        },
    }
    raw = ""
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        try:
            resp = httpx.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            msg = (resp.json().get("message") or {})
            raw = (msg.get("content") or "").strip()
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip().strip("`").strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("レスポンスが dict ではありません")
            chapters = parsed.get("chapters")
            if not isinstance(chapters, list) or len(chapters) == 0:
                raise ValueError("chapters フィールドが空または存在しません")
            log.info(f"  全文解析完了: {len(chapters)} チャプター取得")
            return chapters
        except json.JSONDecodeError as e:
            log.warning(f"  全文解析 JSON パース失敗 (attempt {attempt}): {e} — 先頭: {raw[:200]}")
        except Exception as e:
            log.warning(f"  全文解析エラー (attempt {attempt}): {e}")
        if attempt < OLLAMA_MAX_RETRIES:
            time.sleep(min(2.0, 0.5 * attempt))
    return None


def summarize_fulltext_with_ollama(title: str, fulltext: str) -> str | None:
    """全文字起こしから動画全体の通し要約を生成する。
    失敗時は None を返す（呼び出し元で chapters 連結にフォールバックする）。"""
    # 要約用には長すぎるテキストを圧縮（先頭・中間・末尾を均等サンプリング）
    MAX_CHARS_FOR_SUMMARY = 40000
    if len(fulltext) > MAX_CHARS_FOR_SUMMARY:
        third = MAX_CHARS_FOR_SUMMARY // 3
        sampled = (
            fulltext[:third]
            + "\n\n[... 中略 ...]\n\n"
            + fulltext[len(fulltext)//2 - third//2 : len(fulltext)//2 + third//2]
            + "\n\n[... 中略 ...]\n\n"
            + fulltext[-third:]
        )
        log.info(f"  要約用テキスト: {len(fulltext)} 文字 → {len(sampled)} 文字にサンプリング")
        fulltext = sampled

    user_content = f"動画タイトル: {title}\n\n全文文字起こし:\n{fulltext}"
    payload = {
        "model": FULLTEXT_MODEL,
        "think": False,
        "messages": [
            {"role": "system", "content": FULLTEXT_SUMMARY_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
            "think": False,
        },
    }
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        try:
            resp = httpx.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            msg = resp.json().get("message") or {}
            text = (msg.get("content") or "").strip()
            if text:
                log.info(f"  通し要約生成完了: {len(text)} 字")
                return text
            log.warning(f"  通し要約: 空レスポンス (attempt {attempt})")
        except Exception as e:
            log.warning(f"  通し要約エラー (attempt {attempt}): {e}")
        if attempt < OLLAMA_MAX_RETRIES:
            time.sleep(min(2.0, 0.5 * attempt))
    return None


def _parse_ts(ts_str: str | None) -> float:
    """'mm:ss' または 'h:mm:ss' → 秒数。パース失敗時は 0.0。"""
    if not ts_str:
        return 0.0
    ts_str = ts_str.strip().lstrip("[").rstrip("]")
    parts = ts_str.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, IndexError):
        pass
    return 0.0


def build_chapters_metadata_from_fulltext(raw_chapters: list[dict]) -> dict:
    """全文モードの生チャプターリストを build_chapters_metadata と同じ構造の dict に変換。"""
    def _build_summary_long(chapters: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for c in chapters:
            s = (c.get("summary") or "").strip()
            t = (c.get("title") or "").strip()
            if s:
                parts.append(s)
            elif t:
                parts.append(t)
        text = " ".join(parts)
        if len(text) > 380:
            text = text[:377] + "..."
        return text

    chapters: list[dict[str, Any]] = []
    themes: list[str] = []
    merged_tags: list[str] = []

    for j in raw_chapters:
        tags = j.get("topic_tags") or []
        if not isinstance(tags, list):
            tags = []
        theme = (j.get("inferred_theme") or "").strip()
        if theme:
            themes.append(theme)
        for t in tags:
            t = str(t).strip()
            if t and t not in merged_tags:
                merged_tags.append(t)
        chapters.append({
            "start_seconds": int(_parse_ts(j.get("start_time"))),
            "end_seconds":   int(_parse_ts(j.get("end_time"))),
            "title":         (j.get("chapter_title")   or "").strip(),
            "summary":       (j.get("chapter_summary") or "").strip(),
            "topic_tags":    [str(x).strip() for x in tags if str(x).strip()][:8],
            "inferred_theme": theme,
            "is_highlight":  bool(j.get("is_highlight", False)),
        })

    topic_category = Counter(themes).most_common(1)[0][0] if themes else "不明"
    titles = [c["title"] for c in chapters if c["title"]]
    summary_glue = " / ".join(titles[:4])
    if len(summary_glue) > 90:
        summary_glue = summary_glue[:87] + "…"
    summary_long = _build_summary_long(chapters)

    return {
        "analysis_mode": "chapters",
        "chapter_window_seconds": 0,  # 全文モードでは窓なし
        "topic_category": topic_category,
        "topics": merged_tags[:12],
        "summary": summary_glue or (chapters[0]["summary"] if chapters else ""),
        "summary_long": summary_long,
        "chapters": chapters,
    }


def _ollama_chat_json(system: str, user: str, num_predict: int) -> dict | None:
    payload = {
        "model": OLLAMA_MODEL,
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2, "num_predict": num_predict, "think": False},
    }
    raw = ""
    try:
        resp = httpx.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        msg = (resp.json().get("message") or {})
        raw = (msg.get("content") or "").strip()
        if not raw:
            think = (msg.get("thinking") or "").strip()
            m = re.search(r"\{[\s\S]*\}", think)
            if m:
                raw = m.group(0)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().strip("`").strip()
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError as e:
        log.warning(f"  JSON パース失敗: {e} — 先頭: {raw[:200]}")
        return None
    except Exception as e:
        log.warning(f"  Ollama エラー: {e}")
        return None


PRESCREEN_SYSTEM_PROMPT = """あなたは動画コンテンツの分類専門家です。
動画タイトルだけから、その動画が「人が喋ることがメインのコンテンツ（雑談・トーク・
ポッドキャスト・実況トーク・インタビューなど）」かどうかを判定してください。

判定基準:
- true  : 雑談・トーク・ラジオ・実況トーク・解説がメイン（ゲームでも喋りが主なら true）
- false : 作業用BGM・演奏・歌ってみた・ASMR・料理レシピ・無言ゲームプレイ等

必ず {"eligible": true} または {"eligible": false} だけを返してください。"""


def prescreen_title_with_gemma(title: str) -> bool:
    """タイトルだけで雑談・トーク向きかを判定。
    Gemma への軽量呼び出しで音声DL前に非対象を除外する。
    判定失敗時は True を返して後段に委ねる（誤除外を避ける）。
    """
    result = _ollama_chat_json(
        system=PRESCREEN_SYSTEM_PROMPT,
        user=f"動画タイトル: {title}",
        num_predict=32,
    )
    if result is None:
        log.warning("  プリスクリーン失敗 → 通過扱い")
        return True
    eligible = result.get("eligible")
    if not isinstance(eligible, bool):
        log.warning(f"  プリスクリーン: eligible フィールドが不正 ({eligible!r}) → 通過扱い")
        return True
    return eligible


def analyze_chapter_chunk_with_ollama(
    title: str,
    idx: int,
    total_windows: int,
    window_start: float,
    window_end: float,
    compressed: str,
) -> dict | None:
    t0, t1 = _fmt_ts(window_start), _fmt_ts(window_end)
    attempt_text = compressed
    num_predict = 512
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        user_content = (
            f"動画タイトル: {title}\n"
            f"チャプター {idx + 1}/{total_windows} — 音声の目安区間 {t0} 〜 {t1}\n\n"
            f"間引きトランスクリプト:\n{attempt_text}"
        )
        result = _ollama_chat_json(
            CHAPTER_CHUNK_SYSTEM_PROMPT, user_content, num_predict=num_predict
        )
        if result is not None:
            return result
        if attempt < OLLAMA_MAX_RETRIES:
            next_len = max(120, len(attempt_text) // 2)
            attempt_text = attempt_text[:next_len]
            num_predict = max(256, num_predict - 128)
            log.warning(
                f"  窓 {idx + 1}: Ollama 再試行 {attempt + 1}/{OLLAMA_MAX_RETRIES} "
                f"(input={len(attempt_text)} chars, num_predict={num_predict})"
            )
            time.sleep(min(2.0, 0.5 * attempt))
    return None


def build_chapters_metadata(
    windows: list[list[TimedSegment]],
    chunk_jsons: list[dict],
) -> dict:
    def _build_summary_long(chapters: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for c in chapters:
            s = (c.get("summary") or "").strip()
            t = (c.get("title") or "").strip()
            if s:
                parts.append(s)
            elif t:
                parts.append(t)
        text = " ".join(parts)
        if len(text) > 380:
            text = text[:377] + "..."
        return text

    chapters: list[dict[str, Any]] = []
    for w, j in zip(windows, chunk_jsons):
        tags = j.get("topic_tags") or []
        if not isinstance(tags, list):
            tags = []
        chapters.append({
            "start_seconds": int(w[0].start),
            "end_seconds": int(w[-1].end),
            "title": (j.get("chapter_title") or "").strip(),
            "summary": (j.get("chapter_summary") or "").strip(),
            "topic_tags": [str(x).strip() for x in tags if str(x).strip()][:8],
            "inferred_theme": (j.get("inferred_theme") or "").strip(),
            "is_highlight": bool(j.get("is_highlight", False)),
        })
    themes = [c["inferred_theme"] for c in chapters if c["inferred_theme"]]
    topic_category = Counter(themes).most_common(1)[0][0] if themes else "不明"
    merged_tags: list[str] = []
    for c in chapters:
        for t in c["topic_tags"]:
            if t and t not in merged_tags:
                merged_tags.append(t)
    titles = [c["title"] for c in chapters if c["title"]]
    summary_glue = " / ".join(titles[:4])
    if len(summary_glue) > 90:
        summary_glue = summary_glue[:87] + "…"
    summary_long = _build_summary_long(chapters)
    return {
        "analysis_mode": "chapters",
        "chapter_window_seconds": CHAPTER_WINDOW_SECONDS,
        "topic_category": topic_category,
        "topics": merged_tags[:12],
        "summary": summary_glue or (chapters[0]["summary"] if chapters else ""),
        "summary_long": summary_long,
        "chapters": chapters,
    }


def validate_chapters_metadata(data: dict) -> bool:
    if data.get("analysis_mode") != "chapters":
        log.warning("  analysis_mode が chapters ではありません")
        return False
    ch = data.get("chapters")
    if not isinstance(ch, list) or len(ch) == 0:
        log.warning("  chapters が空です")
        return False
    for i, c in enumerate(ch):
        if not isinstance(c, dict):
            log.warning(f"  chapter[{i}] がオブジェクトではありません")
            return False
        for key in ("start_seconds", "end_seconds", "title", "summary"):
            if key not in c:
                log.warning(f"  chapter[{i}] に {key} がありません")
                return False
    return True


def chapter_digest_for_infer(base: dict) -> str:
    parts = [
        f"topic_category: {base.get('topic_category')}",
        f"topics: {base.get('topics')}",
        f"summary: {base.get('summary')}",
        "chapters:",
    ]
    for c in base.get("chapters") or []:
        if isinstance(c, dict):
            parts.append(
                json.dumps({
                    "title": c.get("title"),
                    "summary": c.get("summary"),
                    "tags": c.get("topic_tags"),
                    "theme": c.get("inferred_theme"),
                }, ensure_ascii=False)
            )
    return "\n".join(parts)


def infer_standard_fields_from_chapters(title: str, base: dict) -> dict | None:
    user = f"動画タイトル: {title}\n\nチャプター要約:\n{chapter_digest_for_infer(base)}"
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        out = _ollama_chat_json(
            INFER_STANDARD_FIELDS_PROMPT,
            user,
            num_predict=OLLAMA_INFER_NUM_PREDICT,
        )
        if out is not None:
            normalize_ai_metadata(out)
            if _infer_fields_complete(out):
                return out
            log.warning(f"  talk系メタ推定: 不完全な応答、再試行 {attempt}/{OLLAMA_MAX_RETRIES}")
        else:
            log.warning(f"  talk系メタ推定失敗、再試行 {attempt}/{OLLAMA_MAX_RETRIES}")
        time.sleep(min(2.0, 0.5 * attempt))
    return None


def _infer_fields_complete(d: dict) -> bool:
    if not isinstance(d.get("feed_eligible"), bool):
        return False
    for k in ("talk_purity", "talk_density"):
        if not isinstance(d.get(k), (int, float)):
            return False
    si = d.get("speaker_info")
    if not isinstance(si, dict):
        return False
    if not all(k in si for k in ("gender", "age_group", "count")):
        return False
    es = d.get("emotional_scores")
    if not isinstance(es, dict):
        return False
    return all(isinstance(es.get(k), (int, float)) for k in ("laughter", "moving", "sadness", "joy"))


def normalize_ai_metadata(data: dict) -> None:
    fe = data.get("feed_eligible")
    if isinstance(fe, str):
        low = fe.strip().lower()
        if low in ("true", "1", "yes"):
            data["feed_eligible"] = True
        elif low in ("false", "0", "no"):
            data["feed_eligible"] = False

    for key in ("talk_purity", "talk_density"):
        v = data.get(key)
        if isinstance(v, (int, float)):
            data[key] = float(v)

    es = data.get("emotional_scores")
    if isinstance(es, dict):
        for k in ("laughter", "moving", "sadness", "joy"):
            v = es.get(k)
            if isinstance(v, (int, float)):
                es[k] = float(v)

    si = data.get("speaker_info")
    if isinstance(si, dict) and "count" in si:
        c = si["count"]
        if isinstance(c, float):
            si["count"] = int(round(c))
        elif isinstance(c, str) and c.isdigit():
            si["count"] = int(c)


def highlights_from_chapters(chapters: list[dict], max_n: int = 5) -> list[dict]:
    """is_highlight=true のチャプターをハイライトとして返す。
    フラグが全て false / 未設定の場合は先頭 max_n 件にフォールバック。"""

    def _to_highlight(c: dict) -> dict:
        title = (c.get("title") or "").strip()
        text = (c.get("summary") or title)[:30] or "見どころ"
        ts = c.get("start_seconds", 0)
        if isinstance(ts, float):
            ts = int(round(ts))
        elif not isinstance(ts, int):
            ts = 0
        return {"time_seconds": ts, "text": text}

    # is_highlight=true のものを優先抽出
    flagged = [c for c in chapters if isinstance(c, dict) and c.get("is_highlight") is True]

    if flagged:
        return [_to_highlight(c) for c in flagged[:max_n]]

    # フォールバック: フラグなし or 全 false → 先頭 max_n 件
    out = []
    for c in chapters[:max_n]:
        if not isinstance(c, dict):
            continue
        title = (c.get("title") or "").strip()
        if not title:
            continue
        out.append(_to_highlight(c))
    return out


def merge_chapters_with_standard(base: dict, infer: dict) -> dict:
    ch = base["chapters"]
    topics = [str(t).strip() for t in (base.get("topics") or []) if str(t).strip()][:5]
    if not topics:
        topics = ["その他"]

    summary = (base.get("summary") or "").strip()
    if len(summary) > 30:
        summary = summary[:30]
    if not summary and ch:
        summary = ((ch[0].get("title") if isinstance(ch[0], dict) else "") or "要約")[:30]
    summary_long = (base.get("summary_long") or "").strip()
    if not summary_long:
        chunks: list[str] = []
        for c in ch:
            if not isinstance(c, dict):
                continue
            s = (c.get("summary") or "").strip()
            t = (c.get("title") or "").strip()
            if s:
                chunks.append(s)
            elif t:
                chunks.append(t)
        summary_long = " ".join(chunks)
        if len(summary_long) > 380:
            summary_long = summary_long[:377] + "..."

    highlights = highlights_from_chapters(ch)
    if not highlights:
        highlights = [{"time_seconds": 0, "text": (summary or "見どころ")[:15]}]

    return {
        "feed_eligible": infer["feed_eligible"],
        "talk_purity": infer["talk_purity"],
        "talk_density": infer["talk_density"],
        "speaker_info": infer["speaker_info"],
        "emotional_scores": infer["emotional_scores"],
        "topic_category": (base.get("topic_category") or "不明").strip() or "不明",
        "topics": topics,
        "summary": summary,
        "summary_long": summary_long,
        "highlights": highlights,
        "analysis_mode": "chapters",
        "chapter_window_seconds": CHAPTER_WINDOW_SECONDS,
        "chapters": ch,
    }


def validate_app_metadata(data: dict) -> bool:
    if not isinstance(data.get("feed_eligible"), bool):
        log.warning("  feed_eligible が bool ではありません")
        return False
    for key in ("talk_purity", "talk_density"):
        v = data.get(key)
        if not isinstance(v, (int, float)):
            log.warning(f"  {key} が数値ではありません")
            return False
    si = data.get("speaker_info", {})
    if not isinstance(si, dict) or not all(k in si for k in ("gender", "age_group", "count")):
        log.warning("  speaker_info が不完全")
        return False
    if si.get("gender") not in _GENDER_OK:
        log.warning(f"  speaker_info.gender が不正: {si.get('gender')!r}")
        return False
    if si.get("age_group") not in _AGE_OK:
        log.warning(f"  speaker_info.age_group が不正: {si.get('age_group')!r}")
        return False
    if not isinstance(si.get("count"), int):
        log.warning("  speaker_info.count が整数ではありません")
        return False
    es = data.get("emotional_scores", {})
    if not isinstance(es, dict):
        return False
    for k in ("laughter", "moving", "sadness", "joy"):
        if not isinstance(es.get(k), (int, float)):
            log.warning(f"  emotional_scores.{k} が数値ではありません")
            return False
    if not isinstance(data.get("topic_category"), str) or not str(data["topic_category"]).strip():
        return False
    topics = data.get("topics")
    if not isinstance(topics, list) or not topics:
        return False
    if not all(isinstance(t, str) and t.strip() for t in topics):
        return False
    if not isinstance(data.get("summary"), str) or not str(data["summary"]).strip():
        return False
    hl = data.get("highlights")
    if not isinstance(hl, list) or not hl:
        return False
    for i, item in enumerate(hl):
        if not isinstance(item, dict):
            return False
        if not isinstance(item.get("time_seconds"), int) or not isinstance(item.get("text"), str):
            log.warning(f"  highlights[{i}] の型が不正")
            return False
        if not str(item["text"]).strip():
            return False
    if data.get("analysis_mode") != "chapters":
        return False
    ch = data.get("chapters")
    if not isinstance(ch, list) or len(ch) == 0:
        return False
    return True


def _gemma_provenance(status: str = "ok", reason: str | None = None) -> dict:
    """Gemma 解析の出所タグ（ai_metadata に混ぜて DB に書き込む）。

    クエリ例:
      WHERE ai_metadata->>'_source' = 'gemma'          -- Gemma が書いた動画だけ
      WHERE ai_metadata->>'_status' = 'failed'          -- Gemma 失敗マーカー
      WHERE ai_metadata->>'_reason' = 'empty_transcript' -- 失敗理由で絞り込み

    失敗動画を再挑戦させたいとき:
      UPDATE videos SET ai_metadata = NULL
      WHERE ai_metadata->>'_status' = 'failed'
        AND ai_metadata->>'_reason' = 'empty_transcript';
    """
    tag = {
        "_source": "gemma",
        "_model": OLLAMA_MODEL,
        "_processed_at": datetime.now(timezone.utc).isoformat(),
    }
    if status != "ok":
        tag["_status"] = status
        if reason:
            tag["_reason"] = reason
    return tag


def save_to_supabase(supabase: Client, video_id: str, metadata: dict) -> bool:
    # Gemma 由来であることを明示する出所タグを自動付与
    metadata = {**metadata, **_gemma_provenance(status="ok")}
    try:
        supabase.table("videos").update({
            "ai_metadata": metadata,
            "analyzed": True,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", video_id).execute()
        return True
    except Exception as e:
        log.error(f"  Supabase 保存エラー: {e}")
        return False


def save_failure_marker(supabase: Client, video_id: str, reason: str) -> None:
    """失敗した動画の ai_metadata に「Gemma で失敗した」マーカーを書き込む。
    これで ai_metadata が空 ({} / null) でなくなり、次バッチで再取得されない。
    analyzed も true にする（「処理は試みた」という意味合い）。
    後から再挑戦したい場合は、該当動画の ai_metadata を手動で NULL に戻す。"""
    marker = _gemma_provenance(status="failed", reason=reason)
    try:
        supabase.table("videos").update({
            "ai_metadata": marker,
            "analyzed": True,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", video_id).execute()
    except Exception as e:
        log.error(f"  failure marker 書き込み失敗: {e}")


def save_failed_result(video: dict, metadata: dict, reason: str) -> Path | None:
    youtube_id = video.get("youtube_id", "unknown")
    vid_id = video.get("id", "unknown")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = FAILED_RESULTS_DIR / f"{ts}_{youtube_id}_{vid_id}.json"
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "video": {"id": video.get("id"), "youtube_id": youtube_id, "title": video.get("title")},
        "ai_metadata": metadata,
    }
    try:
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log.warning(f"  退避保存: {out_path}")
        return out_path
    except Exception as e:
        log.error(f"  退避保存エラー: {e}")
        return None


def process_video(supabase: Client, video: dict) -> bool:
    vid_id     = video["id"]
    youtube_id = video["youtube_id"]
    title      = video["title"]
    _t0 = time.time()
    log.info(f"▶ [{youtube_id}] {title[:50]}")

    # ── Gemma プリスクリーン（音声DL前にタイトルだけで判定） ──────────
    log.info("  Gemma プリスクリーン（タイトル判定）…")
    if not prescreen_title_with_gemma(title):
        log.info("  ✗ 非対象（雑談・トークでない）→ feed_eligible=false でスキップ")
        save_to_supabase(supabase, vid_id, {"feed_eligible": False})
        _write_ledger(video, "skip", "prescreened_not_talk", time.time() - _t0)
        return True  # DB 更新は成功

    log.info("  ✓ トーク向き → 音声解析へ")
    audio_path = download_audio(youtube_id, AUDIO_DIR)
    if audio_path is None:
        log.warning(f"  スキップ（音声取得失敗）: {youtube_id}")
        save_failed_result(video, {}, "audio_download_failed")
        save_failure_marker(supabase, vid_id, "audio_download_failed")
        _write_ledger(video, "failure", "audio_download_failed", time.time() - _t0)
        return False

    try:
        log.info("  Whisper（全長・セグメント単位）…")
        segments = transcribe_segments(audio_path)
        if not segments:
            log.warning("  トランスクリプトが空 → スキップ")
            save_failed_result(video, {}, "empty_transcript")
            save_failure_marker(supabase, vid_id, "empty_transcript")
            _write_ledger(video, "failure", "empty_transcript", time.time() - _t0)
            return False

        # ── チャプター解析: 尺で自動判別（2時間未満→全文一括, 以上→時間窓）──
        dur_sec = iso8601_duration_seconds(video.get("duration"))
        use_fulltext = dur_sec < FULLTEXT_MAX_DURATION_SEC

        if use_fulltext:
            # ── 全文一括モード ────────────────────────────────────────────
            fulltext = fulltext_from_segments(segments)
            log.info(
                f"  全文一括モード: {len(segments)} セグメント / "
                f"{len(fulltext)} 文字 → {FULLTEXT_MODEL}"
            )
            raw_chapters = analyze_fulltext_with_ollama(title, fulltext)
            if raw_chapters is None:
                log.warning("  全文チャプター解析失敗 → スキップ")
                save_failed_result(video, {}, "chapter_analysis_failed")
                save_failure_marker(supabase, vid_id, "chapter_analysis_failed")
                _write_ledger(video, "failure", "chapter_analysis_failed", time.time() - _t0)
                return False
            base = build_chapters_metadata_from_fulltext(raw_chapters)

            # ── 全文から通し要約を生成（chapters 連結より高品質）──────────
            log.info("  Ollama: 全文から通し要約を生成…")
            summary_long_fulltext = summarize_fulltext_with_ollama(title, fulltext)
            if summary_long_fulltext:
                base["summary_long"] = summary_long_fulltext
                base["_summary_long_source"] = "fulltext_resummary"
                base["_summary_long_model"] = FULLTEXT_MODEL
            else:
                log.warning("  通し要約生成失敗 → chapters 連結フォールバック（処理は続行）")
        else:
            # ── 時間窓モード（従来）───────────────────────────────────────
            windows = bucket_timed_segments(segments, CHAPTER_WINDOW_SECONDS)
            log.info(
                f"  チャプター窓 {len(windows)} 本（約 {CHAPTER_WINDOW_SECONDS:.0f} 秒ごと、"
                f"{CHAPTER_COMPRESS_STRIDE} 行に 1 行・最大 {CHAPTER_COMPRESS_MAX_CHARS} 文字で圧縮）"
            )
            chunk_jsons: list[dict] = []
            for i, win in enumerate(windows):
                lines = lines_from_window(win)
                compressed = compress_stride(lines, CHAPTER_COMPRESS_STRIDE, CHAPTER_COMPRESS_MAX_CHARS)
                log.info(
                    f"  窓 {i + 1}/{len(windows)} — 行 {len(lines)} → 圧縮 {len(compressed)} 文字 / {OLLAMA_MODEL}"
                )
                j = analyze_chapter_chunk_with_ollama(
                    title, i, len(windows), win[0].start, win[-1].end, compressed
                )
                if j is None:
                    log.warning(f"  チャプター {i + 1}/{len(windows)} 窓の解析失敗 → スキップ")
                    save_failed_result(
                        video,
                        {"windows_total": len(windows), "failed_at_window": i + 1},
                        "chapter_analysis_failed",
                    )
                    save_failure_marker(supabase, vid_id, "chapter_analysis_failed")
                    _write_ledger(
                        video, "failure", "chapter_analysis_failed", time.time() - _t0,
                        extra={"windows_total": len(windows), "failed_at_window": i + 1},
                    )
                    return False
                chunk_jsons.append(j)
            base = build_chapters_metadata(windows, chunk_jsons)
        if not validate_chapters_metadata(base):
            log.warning("  チャプターメタのバリデーション失敗 → スキップ")
            save_failed_result(video, base, "chapters_validation_failed")
            save_failure_marker(supabase, vid_id, "chapters_validation_failed")
            _write_ledger(video, "failure", "chapters_validation_failed", time.time() - _t0)
            return False

        log.info("  Ollama: 本番互換メタ（feed_eligible 等）をチャプターから推定…")
        infer = infer_standard_fields_from_chapters(title, base)
        if infer is None:
            log.warning("  talk系メタ推定失敗 → スキップ")
            save_failed_result(video, base, "infer_standard_failed")
            save_failure_marker(supabase, vid_id, "infer_standard_failed")
            _write_ledger(video, "failure", "infer_standard_failed", time.time() - _t0)
            return False

        metadata = merge_chapters_with_standard(base, infer)
        normalize_ai_metadata(metadata)
        if not validate_app_metadata(metadata):
            log.warning("  結合メタのバリデーション失敗 → スキップ")
            save_failed_result(video, metadata, "merged_validation_failed")
            save_failure_marker(supabase, vid_id, "merged_validation_failed")
            _write_ledger(video, "failure", "merged_validation_failed", time.time() - _t0)
            return False

        ok = save_to_supabase(supabase, vid_id, metadata)
        if ok:
            log.info(
                f"  ✓ 保存完了: {len(metadata['chapters'])} チャプター — "
                f"{metadata.get('summary', '')} (feed_eligible={metadata.get('feed_eligible')})"
            )
            _write_ledger(
                video, "success", "ok", time.time() - _t0,
                extra={
                    "chapters_count": len(metadata.get("chapters", [])),
                    "feed_eligible": metadata.get("feed_eligible"),
                },
            )
        else:
            save_failed_result(video, metadata, "supabase_save_failed")
            # save_to_supabase 自体が失敗しているので Supabase への追加書き込みは試みない
            _write_ledger(video, "failure", "supabase_save_failed", time.time() - _t0)
        return ok

    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass


POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "60"))


def _download_one(video: dict) -> tuple[dict, "Path | None"]:
    """音声ダウンロードだけを担う軽量タスク（ThreadPoolExecutor から呼ばれる）。
    ネットワーク I/O のみで GPU を使わないため並列実行して問題ない。"""
    youtube_id = video["youtube_id"]
    log.info(f"  [DL] 開始: {youtube_id} — {video.get('title', '')[:40]}")
    path = download_audio(youtube_id, AUDIO_DIR)
    if path:
        log.info(f"  [DL] 完了: {youtube_id}")
    else:
        log.warning(f"  [DL] 失敗: {youtube_id}")
    return video, path


def _analyze_one(supabase: Client, video: dict, audio_path: "Path") -> bool:
    """Whisper 文字起こし → Ollama 解析 → Supabase 保存 を直列で実行する。
    GPU リソースを占有するフェーズはすべてここに集約する。"""
    vid_id     = video["id"]
    youtube_id = video["youtube_id"]
    title      = video["title"]
    _t0 = time.time()

    try:
        log.info(f"  [GPU] Whisper 開始: {youtube_id}")
        segments = transcribe_segments(audio_path)
        if not segments:
            log.warning("  トランスクリプトが空 → スキップ")
            save_failed_result(video, {}, "empty_transcript")
            save_failure_marker(supabase, vid_id, "empty_transcript")
            _write_ledger(video, "failure", "empty_transcript", time.time() - _t0)
            return False

        # ── チャプター解析: 尺で自動判別（2時間未満→全文一括, 以上→時間窓）──
        dur_sec = iso8601_duration_seconds(video.get("duration"))
        use_fulltext = dur_sec < FULLTEXT_MAX_DURATION_SEC

        if use_fulltext:
            fulltext = fulltext_from_segments(segments)
            log.info(
                f"  全文一括モード: {len(segments)} セグメント / "
                f"{len(fulltext)} 文字 → {FULLTEXT_MODEL}"
            )
            raw_chapters = analyze_fulltext_with_ollama(title, fulltext)
            if raw_chapters is None:
                log.warning("  全文チャプター解析失敗 → スキップ")
                save_failed_result(video, {}, "chapter_analysis_failed")
                save_failure_marker(supabase, vid_id, "chapter_analysis_failed")
                _write_ledger(video, "failure", "chapter_analysis_failed", time.time() - _t0)
                return False
            base = build_chapters_metadata_from_fulltext(raw_chapters)
        else:
            windows = bucket_timed_segments(segments, CHAPTER_WINDOW_SECONDS)
            log.info(
                f"  チャプター窓 {len(windows)} 本（約 {CHAPTER_WINDOW_SECONDS:.0f} 秒ごと、"
                f"{CHAPTER_COMPRESS_STRIDE} 行に 1 行・最大 {CHAPTER_COMPRESS_MAX_CHARS} 文字で圧縮）"
            )
            chunk_jsons: list[dict] = []
            for i, win in enumerate(windows):
                lines = lines_from_window(win)
                compressed = compress_stride(lines, CHAPTER_COMPRESS_STRIDE, CHAPTER_COMPRESS_MAX_CHARS)
                log.info(
                    f"  窓 {i + 1}/{len(windows)} — 行 {len(lines)} → 圧縮 {len(compressed)} 文字 / {OLLAMA_MODEL}"
                )
                j = analyze_chapter_chunk_with_ollama(
                    title, i, len(windows), win[0].start, win[-1].end, compressed
                )
                if j is None:
                    log.warning(f"  チャプター {i + 1}/{len(windows)} 窓の解析失敗 → スキップ")
                    save_failed_result(
                        video,
                        {"windows_total": len(windows), "failed_at_window": i + 1},
                        "chapter_analysis_failed",
                    )
                    save_failure_marker(supabase, vid_id, "chapter_analysis_failed")
                    _write_ledger(
                        video, "failure", "chapter_analysis_failed", time.time() - _t0,
                        extra={"windows_total": len(windows), "failed_at_window": i + 1},
                    )
                    return False
                chunk_jsons.append(j)
            base = build_chapters_metadata(windows, chunk_jsons)

        if not validate_chapters_metadata(base):
            log.warning("  チャプターメタのバリデーション失敗 → スキップ")
            save_failed_result(video, base, "chapters_validation_failed")
            save_failure_marker(supabase, vid_id, "chapters_validation_failed")
            _write_ledger(video, "failure", "chapters_validation_failed", time.time() - _t0)
            return False

        log.info("  Ollama: 本番互換メタ（feed_eligible 等）をチャプターから推定…")
        infer = infer_standard_fields_from_chapters(title, base)
        if infer is None:
            log.warning("  talk系メタ推定失敗 → スキップ")
            save_failed_result(video, base, "infer_standard_failed")
            save_failure_marker(supabase, vid_id, "infer_standard_failed")
            _write_ledger(video, "failure", "infer_standard_failed", time.time() - _t0)
            return False

        metadata = merge_chapters_with_standard(base, infer)
        normalize_ai_metadata(metadata)
        if not validate_app_metadata(metadata):
            log.warning("  結合メタのバリデーション失敗 → スキップ")
            save_failed_result(video, metadata, "merged_validation_failed")
            save_failure_marker(supabase, vid_id, "merged_validation_failed")
            _write_ledger(video, "failure", "merged_validation_failed", time.time() - _t0)
            return False

        ok = save_to_supabase(supabase, vid_id, metadata)
        if ok:
            log.info(
                f"  ✓ 保存完了: {len(metadata['chapters'])} チャプター — "
                f"{metadata.get('summary', '')} (feed_eligible={metadata.get('feed_eligible')})"
            )
            _write_ledger(
                video, "success", "ok", time.time() - _t0,
                extra={
                    "chapters_count": len(metadata.get("chapters", [])),
                    "feed_eligible": metadata.get("feed_eligible"),
                },
            )
        else:
            save_failed_result(video, metadata, "supabase_save_failed")
            _write_ledger(video, "failure", "supabase_save_failed", time.time() - _t0)
        return ok

    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass


def _run_one_batch(supabase: Client) -> int:
    """1 バッチ分を並列（ダウンロード）と直列（GPU 解析）の組み合わせで処理する。

    パイプライン:
      Step 1 — yt-dlp ダウンロード: ThreadPoolExecutor で並列（ネットワーク I/O）
      Step 2 — プリスクリーン → Whisper → Ollama: 直列（GPU リソースを競合させない）
    """
    videos = fetch_unanalyzed(supabase, BATCH_SIZE)
    if not videos:
        return 0

    log.info(f"=== バッチ開始: {len(videos)} 件を並列パイプラインで処理します ===")
    success = 0
    fail = 0
    reasons: Counter = Counter()

    # ── Step 1: 音声ダウンロードを並列で先行実行 ─────────────────────────
    # DL_WORKERS = min(1, len(videos))  # Bot対策: 最大2並列
    DL_WORKERS=1
    log.info(f"Step 1: 音声ダウンロード並列開始 ({DL_WORKERS} threads)…")
    downloaded: list[tuple[dict, "Path | None"]] = []
    with ThreadPoolExecutor(max_workers=DL_WORKERS) as ex:
        futures: dict = {}
        for i, v in enumerate(videos):
            futures[ex.submit(_download_one, v)] = v
            # Bot対策: DL投入の間に 3〜8 秒ランダムスリープ（最後の1件は不要）
            if i < len(videos) - 1:
                wait = random.uniform(3, 8)
                log.debug(f"  次のDL投入まで {wait:.1f}s 待機…")
                time.sleep(wait)
        for fut in futures:
            try:
                downloaded.append(fut.result())
            except Exception as e:
                video = futures[fut]
                log.exception(f"  [DL] 予期しない例外: {video.get('youtube_id')}: {e}")
                downloaded.append((video, None))

    log.info(
        f"Step 1 完了: ダウンロード成功 "
        f"{sum(1 for _, p in downloaded if p)} / {len(downloaded)} 件"
    )

    # ── Step 2: GPU 処理（プリスクリーン → Whisper → Ollama）を直列実行 ──
    log.info("Step 2: GPU 解析（直列）開始…")
    for i, (video, audio_path) in enumerate(downloaded, 1):
        vid_id     = video["id"]
        youtube_id = video["youtube_id"]
        title      = video["title"]
        t0 = time.time()
        log.info(f"[{i}/{len(downloaded)}] ▶ [{youtube_id}] {title[:50]}")

        # ダウンロード失敗
        if audio_path is None:
            log.warning(f"  スキップ（音声取得失敗）: {youtube_id}")
            save_failed_result(video, {}, "audio_download_failed")
            save_failure_marker(supabase, vid_id, "audio_download_failed")
            _write_ledger(video, "failure", "audio_download_failed", time.time() - t0)
            fail += 1
            reasons["audio_download_failed"] += 1
            continue

        # プリスクリーン（タイトルだけで判定・GPU 使用なし）
        log.info("  Gemma プリスクリーン（タイトル判定）…")
        if not prescreen_title_with_gemma(title):
            log.info("  ✗ 非対象（雑談・トークでない）→ feed_eligible=false でスキップ")
            save_to_supabase(supabase, vid_id, {"feed_eligible": False})
            _write_ledger(video, "skip", "prescreened_not_talk", time.time() - t0)
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass
            success += 1  # DB 更新は成功
            continue

        log.info("  ✓ トーク向き → GPU 解析へ")
        try:
            ok = _analyze_one(supabase, video, audio_path)
        except Exception as e:
            elapsed = time.time() - t0
            log.exception(f"  ✗ 予期しない例外: {type(e).__name__}: {e}")
            _write_ledger(
                video, "failure", "unexpected_exception", elapsed,
                extra={"error_class": type(e).__name__, "error_message": str(e),
                       "traceback": _tb_mod.format_exc()},
            )
            save_failed_result(
                video,
                {"error_class": type(e).__name__, "error_message": str(e),
                 "traceback": _tb_mod.format_exc()},
                "unexpected_exception",
            )
            try:
                save_failure_marker(supabase, video["id"], "unexpected_exception")
            except Exception:
                pass
            fail += 1
            reasons["unexpected_exception"] += 1
            log.info(f"  失敗 ({time.time() - t0:.1f}s)")
            continue

        elapsed = time.time() - t0
        if ok:
            success += 1
            log.info(f"  完了 ({elapsed:.1f}s)")
        else:
            fail += 1
            log.info(f"  失敗 ({elapsed:.1f}s)")

    # ── バッチサマリ ─────────────────────────────────────────────────────
    log.info(f"=== バッチ完了: 成功 {success}件 / 失敗 {fail}件 ===")
    if reasons:
        breakdown = ", ".join(f"{k}={v}" for k, v in reasons.most_common())
        log.info(f"    失敗内訳: {breakdown}")
    log.info(f"    台帳: {LEDGER_FILE}  /  退避JSON: {FAILED_RESULTS_DIR}")
    return len(videos)


def main() -> None:
    """内部で無限ループ。キューが空なら POLL_INTERVAL_SEC 秒眠ってから再試行。
    これにより PM2 の restart_delay / min_uptime / max_restarts に依存せず、
    予期せぬ停止を防ぐ。"""
    supabase = get_supabase()
    mode_label = (
        f"全文一括({FULLTEXT_MODEL}, 1h未満)" if FULLTEXT_MODE
        else f"時間窓({OLLAMA_MODEL}, {CHAPTER_WINDOW_SECONDS:.0f}s)"
    )
    log.info(
        f"=== pod tube ローカル解析 起動 "
        f"(batch={BATCH_SIZE}, mode={mode_label}, poll={POLL_INTERVAL_SEC}s) ==="
    )
    while True:
        try:
            processed = _run_one_batch(supabase)
        except KeyboardInterrupt:
            log.info("中断シグナルを受信。終了します。")
            return
        except Exception as e:
            # バッチ取得中などの想定外エラー。5 分休んでリトライ。
            log.exception(f"バッチ実行中に例外: {type(e).__name__}: {e}")
            processed = 0
            time.sleep(300)
            continue

        if processed == 0:
            log.info(f"未解析動画はありません。{POLL_INTERVAL_SEC} 秒後に再チェックします。")
            time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()