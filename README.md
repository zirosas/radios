# radios — Terminal Radio Player

A retro Winamp-style TUI for playing NZ radio stations from the MediaWorks network (George FM, The Edge, The Rock, etc.), plus 30,000+ internet stations via [Radio Browser](https://www.radio-browser.info/).

The TUI is inspired by [cliamp](https://github.com/bjarneo/cliamp): a keyboard-driven station picker, a now-playing screen with elapsed time and volume bar, a live FFT spectrum visualizer, and a Radio Browser search screen.

## How it works

- Fetches each station's live HLS stream URL from `radio-api.mediaworks.nz` on every play (no auth required; streams are hosted publicly by StreamGuys).
- Decodes the stream to raw PCM with `ffmpeg` and plays it through `sounddevice` (PortAudio). Because playback goes through Python, volume can be changed at runtime and the PCM data feeds a real FFT spectrum analyzer.

## Requirements

- Python 3.11+ (project venv at `.venv` with `textual`, `sounddevice`, `numpy` — already set up)
- `ffmpeg`
- `libportaudio2` (usually preinstalled on Ubuntu desktop)

To recreate the venv from scratch:

```sh
python3 -m venv .venv
.venv/bin/pip install textual sounddevice numpy
```

## Usage

```sh
./radios            # TUI: arrow keys to pick a station, enter to play
./radios 2          # jump straight into The Edge
./radios -v 30 1    # George FM at 30% volume
```

Keys in the menu: `↑/↓` + `enter` to play, `R` to open the Radio Browser search, `q` quit.

Keys in the Radio Browser screen: type a name + `enter` to search (empty search = top NZ stations by votes), `tab` to the results list, `enter` to play, `b`/`esc` back.

Keys in the player: `←/→` or `n`/`p` switch station within the current list (wraps around), `↑/↓` volume, `b`/`esc` back, `q` quit.

## Included stations (as of 2026-09)

| # | Station    | API slug  |
|---|------------|-----------|
| 1 | George FM  | georgefm  |
| 2 | The Edge   | theedge   |
| 3 | The Rock   | therock   |
| 4 | Mai FM     | maifm     |
| 5 | More FM    | morefm    |
| 6 | Magic      | magic     |
| 7 | The Breeze | thebreeze |
| 8 | The Sound  | thesound  |

## Adding a station

Add one tuple to `STATIONS` in `radios.py`:

```python
("Station Name", "apislug"),
```

API slugs are lowercase with no hyphens (e.g. the-edge → theedge). Find the correct slug by checking which API endpoint the station's page on rova.nz actually calls.

## Files

- `radios` — launcher (runs `radios.py` in the project venv)
- `radios.py` — the TUI app

## Limitations

- The fixed station list only covers the MediaWorks/rova.nz network; everything else (RNZ, NZME, international) comes from Radio Browser, whose data is community-maintained — stream quality and availability vary per station.
- If StreamGuys changes its URL scheme, MediaWorks playback may break (though since the URL is fetched from the API on each run, it should track any URL changes for existing stations).
