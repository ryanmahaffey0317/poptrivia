# poptrivia

Pop-Up Video style trivia for Plex movies, delivered to a Discord channel
while you watch. When someone starts a movie, poptrivia checks whether it
has a pre-generated **trivia track** (a JSON list of timestamped cards
about the production, the cast, easter eggs, goofs, etc.). If yes, it
posts the cards to Discord as the corresponding moments play. If no, it
queues a generation job: scrape IMDB + Wikipedia + TMDB, pull subtitles,
run a two-stage LLM pipeline against an Ollama server, save the track,
and notify you when it's ready.

---

## Required services

- **Plex** — your media server. Used as the playback source.
- **Tautulli** — the playback notifier. poptrivia receives webhooks from
  Tautulli when a movie starts, and polls Tautulli for live playback
  position. Must be reachable from the poptrivia container.
- **Ollama** — runs the LLM for fact extraction and timestamp alignment.
  Expected to live on a separate (LAN) machine; the queue worker tolerates
  it being offline and resumes when it comes back.
- **Discord webhook** — at least one channel webhook URL. Optionally a
  second webhook for system notifications (prep complete, prep failed).
- **TMDB API key** *(optional)* — used to pull credits/keywords as
  additional context for the LLM.

## Deploy

1. Copy `.env.example` to `.env` and fill in the values.
2. Build and start the container:

   ```bash
   docker compose up -d
   ```

3. Confirm it's healthy:

   ```bash
   curl http://localhost:8765/health
   # {"status":"ok"}
   ```

The container exposes port `8765` for the Tautulli webhook. Tautulli must
be able to reach this port.

## Tautulli notification template

In Tautulli, go to **Settings → Notification Agents → Add → Webhook** and
create an entry with:

- **Webhook URL:** `http://<unraid-ip>:8765/tautulli-webhook`
- **Webhook Method:** `POST`
- **Triggers:** check **Playback Start**, **Playback Resume**, and
  **Playback Stop** at minimum. (Pause is handled by the poll loop, not
  the webhook.)
- **Conditions:** *(optional)* `Media Type is movie`. Saves us an entry
  in the logs for every TV play.

Under **Data → Playback Start → JSON Data**, paste:

```json
{
  "event": "{action}",
  "username": "{username}",
  "media_type": "{media_type}",
  "title": "{title}",
  "year": "{year}",
  "imdb_id": "{imdb_id}",
  "tmdb_id": "{themoviedb_id}",
  "plex_guid": "{guid}",
  "file": "{file}",
  "duration_ms": {duration_ms},
  "session_key": "{session_key}",
  "view_offset_ms": {view_offset}
}
```

Note that `duration_ms`, `session_key`, and `view_offset_ms` are
**unquoted** — they need to arrive as numbers. The rest are strings.

Use the same body for the other triggers; only `{action}` varies.

> Tautulli variable names have drifted across versions. If your install
> doesn't recognize a placeholder above, check the up-to-date list at
> https://github.com/Tautulli/Tautulli/wiki/Custom-Notifications. The
> webhook handler is tolerant of common variants (`{action}`/`event`,
> `{user}`/`username`, `{view_offset}`/`view_offset_ms`,
> `{imdb_id}`/`imdbId`, etc.).

## `.env` reference

See `.env.example` for the canonical list. Key fields:

| variable                          | purpose                                              |
|-----------------------------------|------------------------------------------------------|
| `MONITORED_USERS`                 | Comma-separated Plex usernames; everyone else ignored. |
| `OLLAMA_URL`                      | e.g. `http://192.168.1.50:11434`                     |
| `OLLAMA_MODEL`                    | Default `qwen2.5:32b-instruct-q5_K_M`. Any JSON-mode-capable model. |
| `DISCORD_WEBHOOK_URL`             | Required. Trivia cards post here.                    |
| `DISCORD_SYSTEM_WEBHOOK_URL`      | Optional. Prep-complete/failure notifications.       |
| `TMDB_API_KEY`                    | Optional. Skips TMDB metadata fetch if blank.        |
| `CARD_LEAD_TIME_SECONDS`          | Fire cards this far before their timestamp. Default 7. |
| `MISSED_CARD_THRESHOLD_SECONDS`   | On mid-movie join, cards older than this are silenced. Default 30. |
| `TARGET_CARDS_PER_MOVIE`          | Stage 1 produces ~1.5× this many candidate facts. Default 50. |
| `MAX_CARDS_PER_MOVIE`             | Hard cap on the final track size. Default 70.        |

## Manually prepping a movie

Useful for planning ahead — generate a track before you play the movie.

```bash
# By GUID (the movie must already exist in the DB)
docker compose exec poptrivia python scripts/prep_movie.py \
    --guid "plex://movie/5d776b59ad5437001f7936f0"

# By title (creates a row from scratch; --file is required)
docker compose exec poptrivia python scripts/prep_movie.py \
    --title "The Shining" --year 1980 \
    --imdb tt0081505 --tmdb 694 \
    --file "/media/movies/The Shining (1980)/The.Shining.1980.mkv"
```

## Inspecting a generated track

```bash
docker compose exec poptrivia python scripts/inspect_track.py \
    "plex://movie/5d776b59ad5437001f7936f0"
```

## Testing the Discord webhook

```bash
docker compose exec poptrivia python scripts/test_discord.py
```

Sends a sample trivia card and a system message. Useful after first
configuration to confirm the webhook URLs are correct.

## How playback monitoring works

When Tautulli sends a `Playback Start` event for a monitored user
watching a movie:

- If the movie already has a `ready` track, poptrivia starts a
  **SessionMonitor** task tagged with the Plex `session_key`. The
  monitor polls Tautulli every `SESSION_POLL_INTERVAL_SECONDS` and
  fires cards whose timestamp falls in
  `[current_offset, current_offset + CARD_LEAD_TIME_SECONDS]`.
- **Pause** — the monitor keeps polling, but doesn't fire while
  `state != playing`.
- **Seek forward** — detected by playback-delta vs wall-clock divergence
  > 10 s. Cards in the skipped range are silenced (marked fired without
  posting), so re-entering the lookahead window after the seek doesn't
  flood the channel.
- **Seek backward** — already-fired cards stay fired; they don't re-post.
- **Stop / session disappears** — after 3 consecutive missing polls,
  the monitor posts a summary (`X of Y cards fired`) and exits.
- **Mid-movie join** — any cards more than
  `MISSED_CARD_THRESHOLD_SECONDS` before the current offset are
  silenced on first poll.

## Troubleshooting

### "No cards fire"

1. Confirm the movie's status in the DB:

   ```bash
   docker compose exec poptrivia sqlite3 /config/poptrivia.db \
       "SELECT plex_guid, title, status, track_path FROM movies;"
   ```

   Status must be `ready` for cards to fire.

2. Confirm a SessionMonitor started. Check the container logs for
   `SessionMonitor starting | session=…`.

3. Verify `TAUTULLI_URL` and `TAUTULLI_API_KEY` are correct — the
   monitor needs to poll `get_activity` and find a matching
   `session_key`. From inside the container:

   ```bash
   curl "$TAUTULLI_URL/api/v2?apikey=$TAUTULLI_API_KEY&cmd=get_activity"
   ```

4. Confirm the Plex username is in `MONITORED_USERS` (case-sensitive).

### "Prep job stuck"

Run:

```bash
docker compose exec poptrivia sqlite3 /config/poptrivia.db \
    "SELECT * FROM prep_jobs ORDER BY id DESC LIMIT 5;"
```

- `status='pending'` and `last_error` mentions Ollama → the main PC
  isn't reachable. Start Ollama; the worker retries every 60 s.
- `status='running'` for a long time → check container logs for the
  current stage. Stage 2 (timestamp alignment) is the slowest, often
  several minutes against a 32B model.
- `status='failed'` → `last_error` and the corresponding movie row's
  `error_message` describe what went wrong. Common causes:
  no IMDB ID + no Wikipedia match, no embedded or sidecar subtitles
  (and Whisper not installed), Ollama returned malformed JSON.

### "Ollama unreachable"

The container needs to resolve `OLLAMA_URL` from inside Docker. If
Ollama is on the host, use the host's LAN IP, not `localhost`. From
inside the container:

```bash
docker compose exec poptrivia curl -s $OLLAMA_URL/api/tags
```

Should return a JSON list of installed models.

### "Whisper transcription is slow"

The Whisper fallback only runs if both embedded subtitle streams and
sidecar `.srt` files are missing. Best fix: extract or download
sidecar subs ahead of time. To enable the Whisper fallback at all,
install the `whisper` extra (heavy GPU/CPU dependency) and accept the
multi-minute wait per movie.

## Anti-features (deliberately out of scope)

This is a personal project. The following are **not** part of v1 and
won't be added without a clear use case:

- No web UI. Discord *is* the UI.
- No reaction handling / Discord bot. Webhook output only.
- No additional sources beyond IMDB trivia/goofs + Wikipedia + TMDB.
- No multi-modal live mode (vision-LLM trivia from the video stream).
- No manual review UI for generated cards.

If you want any of these, fork it.
