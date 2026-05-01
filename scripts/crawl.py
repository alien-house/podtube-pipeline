#!/usr/bin/env python3
"""
pod tube ローカルクロール
route.ts と同等のロジックを Python で実装。crontab で毎時0分に実行する。

crontab 設定例:
  0 * * * * /usr/bin/python3 /path/to/scripts/local_crawler/crawl.py >> ~/crawl.log 2>&1
"""

import os
import re
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests
from supabase import create_client, Client
from google import genai
from google.genai import types

# ── ログ設定 ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

# ── .env.local 読み込み ───────────────────────────────────────────────────────
def load_env_local():
    env_path = Path(__file__).parent / '.env.local'
    if not env_path.exists():
        log.warning(f'.env.local が見つかりません: {env_path}')
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key not in os.environ:
            os.environ[key] = value

load_env_local()

# ── 設定 ──────────────────────────────────────────────────────────────────────
SUPABASE_URL        = os.environ.get('NEXT_PUBLIC_SUPABASE_URL', '')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY', '')
ADMIN_YOUTUBE_KEY   = os.environ.get('YOUTUBE_API_KEY', '')
GEMINI_API_KEY      = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL        = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash').strip()

YOUTUBE_API_BASE    = 'https://www.googleapis.com/youtube/v3'
CHANNELS_PER_RUN    = 30
KEYWORDS_PER_RUN    = 2
SEARCH_ORDERS       = ['date', 'rating', 'relevance']
MIN_DURATION_SEC    = 5 * 60       # 5分
MAX_DURATION_SEC    = 3 * 60 * 60  # 3時間
BACKOFF_SOFT_SEC    = 6 * 60 * 60  # 6時間
BACKOFF_HARD_SEC    = 30 * 24 * 60 * 60  # 30日
EMPTY_THRESHOLD     = 3

# ── タイトルフィルター ─────────────────────────────────────────────────────────
BLOCKED_PATTERNS = [
    re.compile(r'作業用\s*bgm', re.I),
    re.compile(r'作業\s*bgm', re.I),
    re.compile(r'睡眠\s*bgm', re.I),
    re.compile(r'睡眠用\s*bgm', re.I),
    re.compile(r'睡眠音楽'),
    re.compile(r'勉強用\s*bgm', re.I),
    re.compile(r'集中用\s*bgm', re.I),
    re.compile(r'lo[-\s]?fi', re.I),
    re.compile(r'ローファイ'),
    re.compile(r'弾いてみた'),
    re.compile(r'歌ってみた'),
    re.compile(r'\basmr\b', re.I),
    re.compile(r'実況'),

    # ── 災害・ニュース・報道 ──
    re.compile(r'津波'),
    re.compile(r'地震'),
    re.compile(r'噴火'),
    re.compile(r'台風'),
    re.compile(r'避難'),
    re.compile(r'緊急地震速報'),

    # ── 音楽・演奏系 ──
    re.compile(r'\bmv\b', re.I),
    re.compile(r'music\s*video', re.I),
    re.compile(r'ミュージックビデオ'),
    re.compile(r'ライブ映像'),
    re.compile(r'\blive\s*(at|in|映像|version)', re.I),
    re.compile(r'カラオケ'),
    re.compile(r'フルコーラス'),
    re.compile(r'フル\s*ver', re.I),
    re.compile(r'\bcover\b', re.I),

    # ── 切り抜き・まとめ系 ──
    # re.compile(r'切り抜き'),
    # re.compile(r'名場面'),
    # re.compile(r'神回'),
    # re.compile(r'ハイライト'),
    # re.compile(r'ダイジェスト'),

    # ── 宣伝・PV・CM系 ──
    re.compile(r'\bcm\b', re.I),
    re.compile(r'tvcm', re.I),
    re.compile(r'\bpv\b', re.I),
    re.compile(r'予告編'),
    re.compile(r'トレーラー'),
    re.compile(r'\bteaser\b', re.I),
    re.compile(r'\btrailer\b', re.I),

    # ── ゲーム関連（実況の補完）──
    re.compile(r'ゲーム配信'),
    re.compile(r'\brta\b', re.I),
    re.compile(r'ゆっくり実況'),
    re.compile(r'ゆっくり解説'),

    # ── アニメ・映像制作系 ──
    re.compile(r'\bmmd\b', re.I),
    re.compile(r'声真似'),

    # ── Shorts・短尺 ──
    re.compile(r'#shorts', re.I),
    re.compile(r'ショート動画'),
]

JAPANESE_KANA = re.compile(r'[\u3040-\u30FF\uFF66-\uFF9F]')

def is_japanese_title(title: str) -> bool:
    return bool(JAPANESE_KANA.search(title))

def is_blocked_title(title: str) -> bool:
    return any(p.search(title) for p in BLOCKED_PATTERNS)

def is_channel_not_found(status_code: int) -> bool:
    return status_code == 404

def is_key_invalid(status_code: int, text: str) -> bool:
    return status_code in (400, 403) or 'quotaExceeded' in text or 'keyInvalid' in text

# ── ISO 8601 duration → 秒 ────────────────────────────────────────────────────
def duration_to_seconds(iso: str) -> int:
    if not iso or iso == 'P0D':
        return 0
    m = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', iso)
    if not m:
        return 0
    h = int(m.group(1) or 0)
    mi = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + mi * 60 + s

# ── YouTube API ───────────────────────────────────────────────────────────────
def fetch_video_details(video_ids: list[str], api_key: str) -> list[dict]:
    if not video_ids:
        return []
    params = {
        'key': api_key,
        'id': ','.join(video_ids),
        'part': 'snippet,contentDetails,statistics',
    }
    r = requests.get(f'{YOUTUBE_API_BASE}/videos', params=params, timeout=30)
    r.raise_for_status()
    videos = []
    for item in r.json().get('items', []):
        sn = item.get('snippet') or {}
        cd = item.get('contentDetails') or {}
        thumbs = sn.get('thumbnails', {})
        thumb = (thumbs.get('maxres') or thumbs.get('high') or thumbs.get('default') or {}).get('url', '')
        st = item.get('statistics', {})

        # 必須フィールドの null ガード（YouTube API が稀に部分レスポンスを返すため）
        youtube_id   = item.get('id')
        channel_id   = sn.get('channelId')
        published_at = sn.get('publishedAt')
        duration     = cd.get('duration')
        if not youtube_id or not channel_id or not published_at or not duration:
            log.warning(
                f'  必須フィールド欠損でスキップ: id={youtube_id} '
                f'channel_id={channel_id} published_at={published_at} duration={duration}'
            )
            continue

        videos.append({
            'youtube_id':    youtube_id,
            'title':         sn.get('title') or '',
            'channel_id':    channel_id,
            'channel_title': sn.get('channelTitle') or '',
            'thumbnail_url': thumb,
            'published_at':  published_at,
            'duration':      duration,
            'view_count':    int(st['viewCount'])    if 'viewCount'    in st else None,
            'like_count':    int(st['likeCount'])    if 'likeCount'    in st else None,
            'comment_count': int(st['commentCount']) if 'commentCount' in st else None,
            'analyzed':      False,
        })
    return videos

def fetch_channel_videos(channel_id: str, api_key: str, max_results: int = 20) -> list[dict]:
    playlist_id = channel_id.replace('UC', 'UU', 1)
    params = {
        'key': api_key,
        'playlistId': playlist_id,
        'part': 'contentDetails',
        'maxResults': max_results,
    }
    r = requests.get(f'{YOUTUBE_API_BASE}/playlistItems', params=params, timeout=30)
    if r.status_code == 404:
        raise ValueError(f'404 channelNotFound: {channel_id}')
    if not r.ok:
        raise ValueError(f'{r.status_code} YouTube API error: {r.text[:200]}')
    video_ids = [item['contentDetails']['videoId'] for item in r.json().get('items', [])]
    return fetch_video_details(video_ids, api_key)

def fetch_videos_by_keyword(keyword: str, api_key: str, max_results: int = 50, order: str = 'date') -> list[dict]:
    params = {
        'key': api_key,
        'q': keyword,
        'part': 'snippet',
        'type': 'video',
        'order': order,
        'maxResults': max_results,
        'regionCode': 'JP',
        'relevanceLanguage': 'ja',
        'videoDuration': 'long',
    }
    r = requests.get(f'{YOUTUBE_API_BASE}/search', params=params, timeout=30)
    if not r.ok:
        raise ValueError(f'{r.status_code} YouTube API error: {r.text[:200]}')
    video_ids = [item['id']['videoId'] for item in r.json().get('items', [])]
    return fetch_video_details(video_ids, api_key)

# ── Gemini プリスクリーン ──────────────────────────────────────────────────────
def prescreen_titles(titles: list[str]) -> list[bool]:
    if not titles or not GEMINI_API_KEY:
        return [True] * len(titles)
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        numbered = '\n'.join(f'[{i+1}] {t}' for i, t in enumerate(titles))
        prompt = (
            f'以下の YouTube 動画タイトルについて、それぞれが「人が喋ることがメインの動画'
            f'（雑談・トーク・ポッドキャスト・インタビューなど）」かを判定してください。\n'
            f'作業用BGM・演奏・歌ってみた・ASMR・料理レシピ解説・ゲーム配信・ゲーム実況は false。\n\n'
            f'{numbered}\n\n'
            f'結果を JSON 配列で返してください（例: [true, false, true]）。配列の長さは {len(titles)} にすること。'
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                temperature=0.1,
            ),
        )
        parsed = json.loads(response.text.strip())
        if isinstance(parsed, list) and len(parsed) == len(titles):
            return [bool(v) for v in parsed]
        log.warning(f'prescreenTitles: 長さ不一致 (期待 {len(titles)}, 実際 {len(parsed)})')
        return [True] * len(titles)
    except Exception as e:
        log.warning(f'prescreenTitles 失敗（全件通過）: {e}')
        return [True] * len(titles)

# ── メイン処理 ────────────────────────────────────────────────────────────────
def main():
    if not SUPABASE_URL or 'your-project' in SUPABASE_URL:
        log.error('Supabase が設定されていません')
        sys.exit(1)
    if not ADMIN_YOUTUBE_KEY or ADMIN_YOUTUBE_KEY == 'your-youtube-api-key':
        log.error('YOUTUBE_API_KEY が設定されていません')
        sys.exit(1)

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    now_iso = datetime.now(timezone.utc).isoformat()
    log.info('=== pod tube ローカルクロール 開始 ===')

    # ── ブロック済みチャンネル ──
    blocked_res = supabase.from_('blocked_channels').select('youtube_channel_id').execute()
    blocked_ids = {r['youtube_channel_id'] for r in (blocked_res.data or [])}

    # ── チャンネルキュー構築 ──
    admin_res = (supabase.from_('channels')
                 .select('youtube_channel_id')
                 .eq('active', True)
                 .order('youtube_channel_id')
                 .execute())
    user_res = (supabase.from_('user_youtube_channels')
                .select('youtube_channel_id')
                .or_(f'next_crawl_at.is.null,next_crawl_at.lte.{now_iso}')
                .order('last_crawled_at', desc=False)
                .limit(CHANNELS_PER_RUN * 3)
                .execute())

    channel_set: list[str] = []
    seen: set[str] = set()
    for row in (admin_res.data or []):
        cid = row['youtube_channel_id']
        if cid not in seen:
            seen.add(cid)
            channel_set.append(cid)
    for row in (user_res.data or []):
        cid = row['youtube_channel_id']
        if cid not in seen:
            seen.add(cid)
            channel_set.append(cid)

    target_channels = [c for c in channel_set[:CHANNELS_PER_RUN] if c not in blocked_ids]
    log.info(f'クロール対象チャンネル: {len(target_channels)} 件')

    # ── キーワード取得 ──
    kw_res = (supabase.from_('crawl_keywords')
              .select('id, keyword, crawl_count')
              .eq('active', True)
              .order('last_crawled_at', desc=False, nullsfirst=True)
              .limit(KEYWORDS_PER_RUN)
              .execute())
    keywords = kw_res.data or []

    # ── API キープール ──
    contrib_res = (supabase.from_('user_settings')
                   .select('user_id, youtube_api_key, api_key_failed_at')
                   .eq('plan', 'contributor')
                   .not_.is_('youtube_api_key', 'null')
                   .order('api_key_failed_at', desc=False, nullsfirst=True)
                   .execute())
    key_pool = []
    for row in (contrib_res.data or []):
        if row.get('youtube_api_key'):
            key_pool.append({'user_id': row['user_id'], 'key': row['youtube_api_key']})
    key_pool.append({'user_id': None, 'key': ADMIN_YOUTUBE_KEY})

    log.info(f'APIキープール: {len(key_pool)} 本')

    # ── 状態管理 ──
    total_crawled = 0   # 新規 + 更新の合計（後方互換）
    new_inserted = 0    # 新規追加のみ
    stats_updated = 0   # 既存動画の統計更新のみ
    errors: list[str] = []
    failed_user_ids: set[str] = set()
    succeeded_user_ids: set[str] = set()
    channels_crawled = 0
    keywords_crawled = 0
    channels_with_saves: list[str] = []
    channels_with_empty: list[str] = []
    not_found_channels: list[str] = []

    # ── upsertVideos ──────────────────────────────────────────────────────────
    def upsert_videos(videos: list[dict]) -> int:
        nonlocal total_crawled, new_inserted, stats_updated

        # ① ルールベースフィルター
        candidates = []
        for v in videos:
            if v['channel_id'] in blocked_ids:
                continue
            if not is_japanese_title(v['title']):
                continue
            if is_blocked_title(v['title']):
                continue
            if v['duration'] == 'P0D':
                continue
            sec = duration_to_seconds(v['duration'])
            if sec > 0 and sec < MIN_DURATION_SEC:
                continue
            if sec >= MAX_DURATION_SEC:
                continue
            candidates.append(v)

        if not candidates:
            return 0

        # ② 既存動画を除外
        youtube_ids = [v['youtube_id'] for v in candidates]
        existing_res = (supabase.from_('videos')
                        .select('youtube_id')
                        .in_('youtube_id', youtube_ids)
                        .execute())
        existing_ids = {r['youtube_id'] for r in (existing_res.data or [])}

        new_candidates = [v for v in candidates if v['youtube_id'] not in existing_ids]
        existing_candidates = [v for v in candidates if v['youtube_id'] in existing_ids]

        # ③ Gemini プリスクリーン（新規のみ）
        flags = prescreen_titles([v['title'] for v in new_candidates]) if new_candidates else []
        new_eligible = [v for v, f in zip(new_candidates, flags) if f]
        skipped = len(new_candidates) - len(new_eligible)
        if skipped > 0:
            log.info(f'  プリスクリーン除外: {skipped}/{len(new_candidates)} 件')

        # ④ DB 保存（1件ずつ try/except で囲み、1件失敗でバッチ全体を落とさない）
        saved = 0

        # 新規動画: analyzed=False を付けて INSERT
        for v in new_eligible:
            try:
                payload = {**v, 'analyzed': False}
                res = (supabase.from_('videos')
                       .upsert(payload, on_conflict='youtube_id')
                       .execute())
                if res.data:
                    total_crawled += 1
                    new_inserted += 1
                    saved += 1
            except Exception as e:
                msg = f'upsert(new) 失敗 youtube_id={v.get("youtube_id")}: {e}'
                log.warning(f'  {msg}')
                errors.append(msg)

        # 既存動画: stats のみ update。analyzed/ai_metadata は触らない
        # ※ upsert は内部的に INSERT ... ON CONFLICT のため、NOT NULL カラム（channel_id 等）
        #   を payload に含めないと INSERT 段階で弾かれる。そのため update() を使う。
        stats_keys = {'view_count', 'like_count', 'comment_count',
                      'thumbnail_url', 'title', 'channel_title'}
        for v in existing_candidates:
            try:
                payload = {k: v[k] for k in stats_keys if k in v}
                res = (supabase.from_('videos')
                       .update(payload)
                       .eq('youtube_id', v['youtube_id'])
                       .execute())
                if res.data:
                    total_crawled += 1
                    stats_updated += 1
                    saved += 1
            except Exception as e:
                msg = f'update(existing) 失敗 youtube_id={v.get("youtube_id")}: {e}'
                log.warning(f'  {msg}')
                errors.append(msg)

        return saved

    # ── tryWithKeys ───────────────────────────────────────────────────────────
    def try_with_keys(fn) -> dict:
        for entry in key_pool:
            try:
                videos = fn(entry['key'])
                saved = upsert_videos(videos)
                if entry['user_id']:
                    succeeded_user_ids.add(entry['user_id'])
                return {'ok': True, 'saved': saved, 'not_found': False}
            except Exception as e:
                msg = str(e)
                if '404' in msg or 'channelNotFound' in msg or 'not found' in msg.lower():
                    errors.append(msg)
                    return {'ok': False, 'saved': 0, 'not_found': True}
                errors.append(msg)
                if entry['user_id'] and re.search(r'400|403|quotaExceeded|keyInvalid|forbidden', msg, re.I):
                    failed_user_ids.add(entry['user_id'])
        return {'ok': False, 'saved': 0, 'not_found': False}

    # ① チャンネルクロール
    for channel_id in target_channels:
        result = try_with_keys(lambda key, cid=channel_id: fetch_channel_videos(cid, key, 20))
        if result['not_found']:
            not_found_channels.append(channel_id)
            log.info(f'  チャンネル削除（404）: {channel_id}')
        elif result['ok']:
            channels_crawled += 1
            if result['saved'] > 0:
                channels_with_saves.append(channel_id)
                log.info(f'  ✓ {channel_id}: {result["saved"]} 件保存')
            else:
                channels_with_empty.append(channel_id)

    # 404 チャンネルを両テーブルから削除
    if not_found_channels:
        supabase.from_('user_youtube_channels').delete().in_('youtube_channel_id', not_found_channels).execute()
        supabase.from_('channels').delete().in_('youtube_channel_id', not_found_channels).execute()

    # 保存ありチャンネル: バックオフリセット
    if channels_with_saves:
        supabase.from_('user_youtube_channels').update({
            'last_crawled_at': now_iso,
            'consecutive_empty_crawls': 0,
            'next_crawl_at': None,
        }).in_('youtube_channel_id', channels_with_saves).execute()

    # 保存なしチャンネル: 段階的バックオフ
    if channels_with_empty:
        empty_res = (supabase.from_('user_youtube_channels')
                     .select('youtube_channel_id, consecutive_empty_crawls')
                     .in_('youtube_channel_id', channels_with_empty)
                     .execute())
        for row in (empty_res.data or []):
            new_count = (row.get('consecutive_empty_crawls') or 0) + 1
            delay_sec = BACKOFF_HARD_SEC if new_count >= EMPTY_THRESHOLD else BACKOFF_SOFT_SEC
            next_crawl = (datetime.now(timezone.utc) + timedelta(seconds=delay_sec)).isoformat()
            supabase.from_('user_youtube_channels').update({
                'last_crawled_at': now_iso,
                'consecutive_empty_crawls': new_count,
                'next_crawl_at': next_crawl,
            }).eq('youtube_channel_id', row['youtube_channel_id']).execute()

    # ② キーワードクロール
    for kw in keywords:
        order = SEARCH_ORDERS[(kw.get('crawl_count') or 0) % 3]
        result = try_with_keys(lambda key, k=kw['keyword'], o=order: fetch_videos_by_keyword(k, key, 50, o))
        if result['ok']:
            keywords_crawled += 1
            log.info(f'  キーワード「{kw["keyword"]}」: {result["saved"]} 件保存')
        supabase.from_('crawl_keywords').update({
            'last_crawled_at': now_iso,
            'crawl_count': (kw.get('crawl_count') or 0) + 1,
        }).eq('id', kw['id']).execute()

    # ── キー健康状態更新 ──
    if failed_user_ids:
        (supabase.from_('user_settings')
         .update({'api_key_failed_at': now_iso})
         .in_('user_id', list(failed_user_ids))
         .is_('api_key_failed_at', None)
         .execute())
    if succeeded_user_ids:
        (supabase.from_('user_settings')
         .update({'api_key_failed_at': None})
         .in_('user_id', list(succeeded_user_ids))
         .not_.is_('api_key_failed_at', None)
         .execute())

    # ── crawl_runs 記録 ──
    estimated_quota = channels_crawled * 2 + keywords_crawled * 101
    run_summary = {
        'ran_at':           now_iso,
        'channels_target':  len(target_channels),
        'channels_crawled': channels_crawled,
        'keywords_target':  len(keywords),
        'keywords_crawled': keywords_crawled,
        'videos_upserted':  total_crawled,
        'estimated_quota':  estimated_quota,
        'key_pool_size':    len(key_pool),
        'keys_failed':      len(failed_user_ids),
        'keys_succeeded':   len(succeeded_user_ids),
        'error_count':      len(errors),
        'sample_error':     errors[0] if errors else None,
    }
    supabase.from_('crawl_runs').insert(run_summary).execute()

    log.info(
        f'=== 完了: 新規 {new_inserted} 件 / 統計更新 {stats_updated} 件 / '
        f'CH {channels_crawled}/{len(target_channels)} / KW {keywords_crawled}/{len(keywords)} / '
        f'Quota ~{estimated_quota} / エラー {len(errors)} 件 ==='
    )
    if errors:
        log.warning(f'最初のエラー: {errors[0]}')


if __name__ == '__main__':
    main()
