#!/usr/bin/env python3
"""radios — Terminal radio player with a retro Winamp-style TUI.

Plays NZ radio stations from the MediaWorks network. Audio is decoded by
ffmpeg to raw PCM and played through sounddevice, which enables runtime
volume control and a real FFT spectrum visualizer.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import threading
import time
import urllib.parse
import urllib.request

import numpy as np
import sounddevice as sd
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Input, Label, OptionList, Static
from textual.widgets.option_list import Option

API_BASE = "https://radio-api.mediaworks.nz/radio-api/v3/station"
RADIO_BROWSER_BASE = "https://de1.api.radio-browser.info/json"
USER_AGENT = "radios-cli/1.0 (github.com/user/radios)"

STATIONS = [
    ("George FM", "georgefm"),
    ("The Edge", "theedge"),
    ("The Rock", "therock"),
    ("Mai FM", "maifm"),
    ("More FM", "morefm"),
    ("Magic", "magic"),
    ("The Breeze", "thebreeze"),
    ("The Sound", "thesound"),
]

SAMPLE_RATE = 44100
CHANNELS = 2
BLOCK_FRAMES = 2048
CHUNK_BYTES = BLOCK_FRAMES * CHANNELS * 2  # s16le
NUM_BANDS = 24
VIS_ROWS = 8

GREEN = "#33ff33"
AMBER = "#ffbf00"
DIM = "#1a7a1a"


def fetch_stream_url(slug: str) -> str | None:
    url = f"{API_BASE}/{slug}/digital/web?unique=true"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.load(resp)
        return data.get("audioRenditions", [{}])[0].get("url")
    except Exception:
        return None


def search_radio_browser(
    name: str = "", countrycode: str = "NZ", limit: int = 50
) -> list[tuple[str, str]]:
    """Search Radio Browser for stations, most voted first.

    Returns a list of (station_name, resolved_stream_url) tuples.
    """
    params = f"hidebroken=true&order=votes&reverse=true&limit={limit}"
    params += f"&countrycode={urllib.parse.quote(countrycode)}"
    if name:
        params += f"&name={urllib.parse.quote(name)}"
    req = urllib.request.Request(
        f"{RADIO_BROWSER_BASE}/stations/search?{params}",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
        results: list[tuple[str, str]] = []
        seen_urls: set[str] = set()
        for st in data:
            url = st.get("url_resolved") or st.get("url")
            title = (st.get("name") or "").strip()
            if not url or not title or "[IP_ADDRESS]" in url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append((title, url))
        return results
    except Exception:
        return []


class AudioEngine:
    """ffmpeg -> PCM pipe -> sounddevice, with FFT band analysis."""

    def __init__(self) -> None:
        self.volume = 0.5
        self.bands = np.zeros(NUM_BANDS)
        self.proc: subprocess.Popen | None = None
        self.stream: sd.OutputStream | None = None
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self.audio_started = threading.Event()

    def start(self, url: str) -> None:
        self.stop()
        self._stop.clear()
        self.audio_started.clear()
        self._queue = queue.Queue(maxsize=64)
        cmd = [
            "ffmpeg", "-loglevel", "error",
            "-reconnect", "1", "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            "-i", url,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE),
            "-",
        ]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=BLOCK_FRAMES,
            callback=self._callback,
        )
        self.stream.start()

    def _read_loop(self) -> None:
        assert self.proc and self.proc.stdout
        buf = b""
        while not self._stop.is_set():
            data = self.proc.stdout.read(8192)
            if not data:
                break
            buf += data
            while len(buf) >= CHUNK_BYTES:
                chunk, buf = buf[:CHUNK_BYTES], buf[CHUNK_BYTES:]
                try:
                    self._queue.put(chunk, timeout=0.5)
                except queue.Full:
                    pass  # drop audio rather than block the decoder

    def _callback(self, outdata, frames, time_info, status) -> None:
        try:
            data = self._queue.get_nowait()
        except queue.Empty:
            outdata[:] = 0
            return
        self.audio_started.set()
        samples = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)
        self._analyze(samples)
        if self.volume < 1.0:
            samples = (samples.astype(np.float32) * self.volume).astype(np.int16)
        n = min(len(samples), frames)
        outdata[:n] = samples[:n]
        if n < frames:
            outdata[n:] = 0

    def _analyze(self, samples: np.ndarray) -> None:
        mono = samples.astype(np.float32).mean(axis=1) / 32768.0
        if len(mono) < 64:
            return
        windowed = mono * np.hanning(len(mono))
        mag = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(mono), 1.0 / SAMPLE_RATE)
        edges = np.geomspace(40, 16000, NUM_BANDS + 1)
        vals = np.zeros(NUM_BANDS)
        for i in range(NUM_BANDS):
            mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
            if mask.any():
                db = 20 * np.log10(mag[mask].mean() + 1e-7)
                vals[i] = np.clip((db + 55) / 55, 0.0, 1.0)
        # attack fast, decay slow — classic spectrum feel
        self.bands = np.maximum(vals, self.bands * 0.72)

    def set_volume(self, value: float) -> None:
        self.volume = max(0.0, min(1.0, value))

    def stop(self) -> None:
        self._stop.set()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None
        self.bands = np.zeros(NUM_BANDS)


class Visualizer(Static):
    """Spectrum analyzer rendered as colored block columns."""

    def render_bands(self, bands: np.ndarray) -> None:
        heights = (bands * VIS_ROWS).astype(int)
        lines: list[Text] = []
        for row in range(VIS_ROWS - 1, -1, -1):
            line = Text()
            for h in heights:
                if h > row:
                    if row >= VIS_ROWS - 2:
                        color = "#ff3333"
                    elif row >= VIS_ROWS - 4:
                        color = AMBER
                    else:
                        color = GREEN
                    line.append("██", style=color)
                else:
                    line.append("██", style="#0a2a0a")
                line.append(" ")
            lines.append(line)
        self.update(Text("\n").join(lines))


def volume_bar(volume: float, width: int = 20) -> Text:
    filled = round(volume * width)
    bar = Text()
    bar.append("█" * filled, style=GREEN)
    bar.append("░" * (width - filled), style=DIM)
    bar.append(f" {round(volume * 100):3d}%", style=AMBER)
    return bar


class StationScreen(Screen):
    BINDINGS = [
        Binding("q", "quit_app", "Quit"),
        Binding("r", "browse", "Browse"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-box"):
            yield Label("R A D I O S", id="title")
            yield Label("── NZ · MediaWorks ──", id="subtitle")
            yield OptionList(
                *[Option(f" {name}", id=slug) for name, slug in STATIONS],
                Option(" 🌐 browse radio browser…", id="__browse__"),
                id="stations",
            )
            yield Label("↑/↓ select · enter play · R browse · q quit", id="hint")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id == "__browse__":
            self.action_browse()
            return
        slug = event.option.id
        name = next(n for n, s in STATIONS if s == slug)
        playlist = [(n, None, s) for n, s in STATIONS]
        index = next(i for i, (_, s) in enumerate(STATIONS) if s == slug)
        self.app.start_playlist(playlist, index)

    def action_browse(self) -> None:
        self.app.push_screen(BrowseScreen())

    def action_quit_app(self) -> None:
        self.app.exit()


class BrowseScreen(Screen):
    """Search and play stations from radio-browser.info (30k+ stations)."""

    BINDINGS = [
        Binding("b", "back", "Back"),
        Binding("escape", "back", "Back"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._results: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="browse-box"):
            yield Label("RADIO BROWSER", id="title")
            yield Label("── radio-browser.info ──", id="subtitle")
            yield Input(
                placeholder="search name (empty = top NZ stations)…",
                id="browse-search",
            )
            yield Label("loading…", id="browse-status")
            yield OptionList(id="browse-results")
            yield Label("enter search/play · b back · q quit", id="hint")

    def on_mount(self) -> None:
        self.run_worker(lambda: self._search(""), thread=True)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.run_worker(lambda: self._search(event.value.strip()), thread=True)

    def _search(self, name: str) -> None:
        self.app.call_from_thread(
            self.query_one("#browse-status", Label).update, "searching…"
        )
        results = search_radio_browser(name=name)
        self._results = results
        self.app.call_from_thread(self._render_results, name, results)

    def _render_results(self, name: str, results: list[tuple[str, str]]) -> None:
        status = self.query_one("#browse-status", Label)
        options = self.query_one("#browse-results", OptionList)
        options.clear_options()
        if not results:
            status.update("no stations found")
            return
        label = f"top {len(results)} NZ stations" if not name else f"{len(results)} results for '{name}'"
        status.update(label)
        options.add_options([Option(f" {title}", id=str(i)) for i, (title, _) in enumerate(results)])
        options.highlighted = 0

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        playlist = [(name, url, None) for name, url in self._results]
        self.app.start_playlist(playlist, event.option_index)

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_quit_app(self) -> None:
        self.app.exit()


class PlayerScreen(Screen):
    BINDINGS = [
        Binding("up", "volume_up", "Vol +"),
        Binding("down", "volume_down", "Vol -"),
        Binding("right,n", "next_station", "Next"),
        Binding("left,p", "prev_station", "Prev"),
        Binding("b", "back", "Stations"),
        Binding("escape", "back", "Stations"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(
        self,
        name: str,
        url: str | None = None,
        slug: str | None = None,
        playlist: list[tuple[str, str | None, str | None]] | None = None,
        index: int = 0,
    ) -> None:
        """Play a station.

        `slug` = MediaWorks station (URL resolved via their API).
        `url`  = direct stream URL (e.g. from Radio Browser).
        `playlist` = list of (name, url, slug) tuples for prev/next switching.
        """
        super().__init__()
        self.playlist = playlist or [(name, url, slug)]
        self.index = index
        self.station_name, self.url, self.slug = self.playlist[index]
        self.started_at: float | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="player-box"):
            yield Label("♪ NOW PLAYING", id="np-label")
            yield Label(self.station_name, id="np-station")
            yield Label("connecting…", id="np-status")
            yield Label("", id="np-time")
            yield Static(id="np-volume")
            yield Visualizer(id="np-vis")
            yield Label("←/→ station · ↑/↓ volume · b list · q quit", id="hint")

    def on_mount(self) -> None:
        self._refresh_volume()
        self.set_interval(1 / 15, self._tick_vis)
        self.set_interval(1.0, self._tick_time)
        self.run_worker(self._connect, thread=True)

    def _connect(self) -> None:
        if self.slug is not None:
            url = fetch_stream_url(self.slug)
        else:
            url = self.url
        if not url:
            self.app.call_from_thread(self._show_error, "could not get stream URL")
            return
        self.app.call_from_thread(self._set_status, "buffering…")
        try:
            self.app.engine.start(url)
        except Exception as exc:
            self.app.call_from_thread(self._show_error, f"audio error: {exc}")
            return
        self.started_at = time.monotonic()
        self.app.call_from_thread(self._set_status, "live")

    def _switch_station(self, offset: int) -> None:
        self.index = (self.index + offset) % len(self.playlist)
        self.station_name, self.url, self.slug = self.playlist[self.index]
        self.app.engine.stop()
        self.started_at = None
        self.query_one("#np-station", Label).update(self.station_name)
        self._set_status("connecting…")
        self.run_worker(self._connect, thread=True)

    def action_next_station(self) -> None:
        self._switch_station(1)

    def action_prev_station(self) -> None:
        self._switch_station(-1)

    def _set_status(self, text: str) -> None:
        self.query_one("#np-status", Label).update(f"[{AMBER}]{text}[/]")

    def _show_error(self, text: str) -> None:
        self._set_status(f"error: {text}")

    def _tick_vis(self) -> None:
        self.query_one("#np-vis", Visualizer).render_bands(self.app.engine.bands)

    def _tick_time(self) -> None:
        if self.started_at is None:
            return
        elapsed = int(time.monotonic() - self.started_at)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        stamp = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        self.query_one("#np-time", Label).update(
            Text("⏱ ", style=DIM) + Text(stamp, style=GREEN)
        )

    def _refresh_volume(self) -> None:
        bar = Text("VOL ", style=AMBER) + volume_bar(self.app.engine.volume)
        self.query_one("#np-volume", Static).update(bar)

    def action_volume_up(self) -> None:
        self.app.engine.set_volume(self.app.engine.volume + 0.05)
        self._refresh_volume()

    def action_volume_down(self) -> None:
        self.app.engine.set_volume(self.app.engine.volume - 0.05)
        self._refresh_volume()

    def action_back(self) -> None:
        self.app.engine.stop()
        self.app.pop_screen()

    def action_quit_app(self) -> None:
        self.app.exit()


class RadiosApp(App):
    CSS = """
    Screen {
        background: #050505;
    }
    #menu-box, #player-box {
        width: 64;
        height: auto;
        border: heavy #1a7a1a;
        padding: 1 2;
        margin: 2 4;
        background: #0a0a0a;
    }
    #title {
        color: #33ff33;
        text-style: bold;
        text-align: center;
    }
    #subtitle {
        color: #ffbf00;
        text-align: center;
        margin-bottom: 1;
    }
    #stations {
        background: #0a0a0a;
        border: none;
        height: auto;
    }
    #stations .option-list--option-highlighted {
        background: #143314;
        color: #33ff33;
        text-style: bold;
    }
    #hint {
        color: #1a7a1a;
        text-align: center;
        margin-top: 1;
    }
    #np-label {
        color: #ffbf00;
        text-style: bold;
    }
    #np-station {
        color: #33ff33;
        text-style: bold;
        margin-bottom: 1;
    }
    #np-status, #np-time, #np-volume {
        margin-bottom: 1;
    }
    #np-vis {
        height: 9;
        margin: 1 0;
    }
    #browse-box {
        width: 76;
        height: auto;
        max-height: 90%;
        border: heavy #1a7a1a;
        padding: 1 2;
        margin: 2 4;
        background: #0a0a0a;
    }
    #browse-search {
        background: #0a0a0a;
        border: solid #1a7a1a;
        color: #33ff33;
        margin-bottom: 1;
    }
    #browse-search:focus {
        border: solid #33ff33;
    }
    #browse-status {
        color: #ffbf00;
        margin-bottom: 1;
    }
    #browse-results {
        background: #0a0a0a;
        border: none;
        height: 14;
    }
    #browse-results .option-list--option-highlighted {
        background: #143314;
        color: #33ff33;
        text-style: bold;
    }
    """

    def __init__(self, initial_volume: float = 0.5, direct_slug: str | None = None) -> None:
        super().__init__()
        self.engine = AudioEngine()
        self.engine.set_volume(initial_volume)
        self._direct_slug = direct_slug

    def on_mount(self) -> None:
        self.push_screen(StationScreen())
        if self._direct_slug:
            name = next((n for n, s in STATIONS if s == self._direct_slug), None)
            if name:
                self.start_station(name, self._direct_slug)

    def start_station(self, name: str, slug: str) -> None:
        self.push_screen(PlayerScreen(name, slug=slug))

    def start_playlist(
        self, playlist: list[tuple[str, str | None, str | None]], index: int
    ) -> None:
        name, url, slug = playlist[index]
        self.push_screen(PlayerScreen(name, url=url, slug=slug, playlist=playlist, index=index))

    def on_unmount(self) -> None:
        self.engine.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal radio player (NZ MediaWorks stations)")
    parser.add_argument("station", nargs="?", type=int, help="station number (skip the menu)")
    parser.add_argument("-v", "--volume", type=int, default=50, help="volume 0-100 (default: 50)")
    args = parser.parse_args()

    direct_slug = None
    if args.station is not None:
        if not 1 <= args.station <= len(STATIONS):
            parser.error(f"station must be 1-{len(STATIONS)}")
        direct_slug = STATIONS[args.station - 1][1]

    app = RadiosApp(initial_volume=args.volume / 100, direct_slug=direct_slug)
    app.run()


if __name__ == "__main__":
    main()
