#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║   SONAR STATION — LOFAR / DEMON                                     ║
║   PyQt6 · PyQtGraph · sounddevice · NumPy · SciPy · pyFFTW · Numba ║
║   ──────────────────────────────────────────────────────────────    ║
║   INSTALL:                                                           ║
║     pip install PyQt6 pyqtgraph sounddevice numpy scipy pyfftw     ║
║     pip install numba  (optional — JIT-compiles the OS-CFAR loop)  ║
║                                                                      ║
║   LOFAR  – narrowband machinery tonals (STFT + robust/TPSW norm)    ║
║   DEMON  – propeller / drone blade-rate (envelope demodulation)     ║
║   ALE    – adaptive line enhancer (optional pre-process)            ║
║   NORM   – Off / TPSW / Robust / OS-CFAR (with guard cells)         ║
║   EIGEN  – cross-frame PCA/SVD subspace denoiser (optional)         ║
║                                                                      ║
║   CONTROLS:                                                          ║
║     Scroll waterfall       – zoom freq axis (each window separate)  ║
║     Horizontal scroll      – pan freq axis                          ║
║     Double-click waterfall – reset zoom to full range               ║
║     Bandpass reticule      – drag / scroll to set analysis band     ║
║     AUDIO OUT              – bandpass-filtered signal (reticule)    ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import sys
import time
import queue
import threading
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.signal as sp
from scipy.signal import ZoomFFT
from scipy.ndimage import percentile_filter, uniform_filter1d
import scipy.io.wavfile as wavfile
import sounddevice as sd

# ── pyFFTW — drop-in faster FFT with FFTW planning; falls back to NumPy ───────
try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as _np_fft
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(30.0)
    _PYFFTW = True
except ImportError:
    import numpy.fft as _np_fft   # type: ignore[assignment]
    _PYFFTW = False

# ── Numba — JIT-compiles the OS-CFAR inner loop; falls back silently ───────────
try:
    import numba as _numba
    _NUMBA = True
except ImportError:
    _NUMBA = False

# prange → parallel range inside @njit(parallel=True); degrades to range otherwise.
_prange = _numba.prange if _NUMBA else range

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QComboBox,
    QGroupBox, QFileDialog, QStatusBar,
    QMenu, QCheckBox, QDoubleSpinBox, QScrollBar,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QAction, QColor
import pyqtgraph as pg


# ═══════════════════════════════════════════════════════════════════════════════
#  LOCK-FREE AUDIO QUEUE
# ═══════════════════════════════════════════════════════════════════════════════
class _AudioQueue:
    """
    Low-lock drop-in for queue.Queue in the audio path.

    put() / put_nowait() never block and never raise queue.Full — the deque
    silently discards the oldest unread chunk when at capacity.  This removes
    the threading.Condition acquire/release cycle that queue.Queue performs on
    every put AND get (contributing ~19 s of lock.acquire time in profiling).

    get(timeout=…) blocks via a lightweight threading.Event instead of the
    heavier Condition machinery, cutting lock acquisitions from O(N×retries)
    to O(N) for the happy path.

    Supports put, put_nowait, get (with optional block/timeout), get_nowait,
    and a put(None) sentinel for shutdown — compatible with all call sites.
    """

    def __init__(self, maxsize: int = 128) -> None:
        self._d  = deque(maxlen=maxsize)
        self._ev = threading.Event()

    def put(self, item, block: bool = True, timeout=None) -> None:
        # deque.appendleft is GIL-atomic for a single writer; never raises Full.
        self._d.appendleft(item)
        self._ev.set()

    def put_nowait(self, item) -> None:
        self._d.appendleft(item)
        self._ev.set()

    def get(self, block: bool = True, timeout=None):
        if block:
            self._ev.wait(timeout)
        try:
            item = self._d.pop()
        except IndexError:
            raise queue.Empty
        if not self._d:
            self._ev.clear()
        return item

    def get_nowait(self):
        return self.get(block=False)

    @property
    def maxsize(self) -> int:
        return self._d.maxlen  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
CHUNK    = 512
FFT_N    = 4096
HISTORY  = 300
BG       = "#050c05"

AUDIO_SR  = 48_000            # capture / bandpass-filter / audio-out rate
LOFAR_SR  = 16_000            # LOFAR analysis rate (Nyquist = 8 kHz = DISP_HI)
SR_DEF    = AUDIO_SR          # kept as default arg for Waterfall()

DISP_LO   = 0.0
DISP_HI   = 8_000.0
DISP_SPAN = DISP_HI - DISP_LO

BP_LO  = 0.0
BP_HI  = float(AUDIO_SR // 2)   # 24 000 Hz — full audio Nyquist for bandpass reticule

# DEMON — envelope modulation spectrum (ships + drones)
DEMON_FFT_N   = 4096
DEMON_HISTORY = 300
DEMON_DISP_HI = 500.0          # Hz — default covers ships + most small drones
DEMON_DS_SR   = 2000.0         # target envelope rate (Nyquist 1 kHz → supports 0–1 kHz preset)

# ZoomFFT: ring buffer holds this many × FFT_N samples.
# At Nx zoom, window grows Nx, giving true Nx frequency resolution improvement.
# 8  →  1.49 s window @ 44.1 kHz / ds=2  → 0.67 Hz/bin in 0-1 kHz zoom
# 32 →  5.95 s window @ 44.1 kHz / ds=2  → 0.17 Hz/bin  (ship classification)
ZOOM_BUF_MULT = 32

# EIGEN — cross-frame PCA/SVD subspace denoiser (LOFAR + DEMON, see
# _EigenDenoiser below).  HIST = how many past frames form the window the
# eigen-spectra are estimated from; RANK = how many leading singular
# vectors are kept per reconstruction; RECALC = re-factor the basis only
# every N frames (the eigen-spectra of rotating-machinery tonals barely
# move over a few tens of ms, so this trades imperceptible staleness for
# an N× cut in SVD calls).
PCA_HIST   = 24
PCA_RANK   = 4
PCA_RECALC = 6


# ═══════════════════════════════════════════════════════════════════════════════
#  COLOUR MAPS  — tactical first, analysis last
# ═══════════════════════════════════════════════════════════════════════════════
def cmap_green_phosphor() -> pg.ColorMap:
    """Classic green phosphor — dark background, bright green lines."""
    return pg.ColorMap(
        pos  =[0.00, 0.15, 0.40, 0.70, 1.00],
        color=[(  0,  10,   0, 255),
               (  0,  40,   0, 255),
               (  0, 140,  20, 255),
               ( 40, 220,  60, 255),
               (180, 255, 180, 255)],
    )


def cmap_amber() -> pg.ColorMap:
    """Amber / orange phosphor — low eye strain."""
    return pg.ColorMap(
        pos  =[0.00, 0.20, 0.50, 0.80, 1.00],
        color=[(  8,   4,   0, 255),
               ( 40,  20,   0, 255),
               (160,  80,   0, 255),
               (240, 160,  20, 255),
               (255, 230, 140, 255)],
    )


def cmap_hot() -> pg.ColorMap:
    """Hot — black → red → yellow → white."""
    return pg.ColorMap(
        pos  =[0.00, 0.25, 0.55, 0.80, 1.00],
        color=[(  0,   0,   0, 255),
               (120,   0,   0, 255),
               (230,  60,   0, 255),
               (255, 200,   0, 255),
               (255, 255, 220, 255)],
    )


def cmap_gray() -> pg.ColorMap:
    """Pure grayscale."""
    return pg.ColorMap(
        pos  =[0.00, 0.30, 0.60, 1.00],
        color=[(  0,   0,   0, 255),
               ( 50,  50,  50, 255),
               (140, 140, 140, 255),
               (255, 255, 255, 255)],
    )


def cmap_ice() -> pg.ColorMap:
    """Ice — deep blue floor → cyan → white.  Darker than jet."""
    return pg.ColorMap(
        pos  =[0.00, 0.25, 0.55, 0.80, 1.00],
        color=[(  0,   0,  20, 255),
               (  0,  40, 100, 255),
               (  0, 140, 200, 255),
               (100, 220, 255, 255),
               (230, 250, 255, 255)],
    )


def cmap_bone() -> pg.ColorMap:
    """Bone — cool blue-gray, subtle and easy on the eyes."""
    return pg.ColorMap(
        pos  =[0.00, 0.30, 0.60, 1.00],
        color=[(  0,   0,  10, 255),
               ( 40,  50,  70, 255),
               (120, 140, 160, 255),
               (230, 235, 240, 255)],
    )


def cmap_copper() -> pg.ColorMap:
    """Copper — black → brown → orange → pale yellow."""
    return pg.ColorMap(
        pos  =[0.00, 0.30, 0.60, 1.00],
        color=[(  0,   0,   0, 255),
               ( 80,  40,  10, 255),
               (180, 100,  30, 255),
               (255, 220, 160, 255)],
    )


def cmap_night() -> pg.ColorMap:
    """Night-vision green — higher contrast, deeper blacks."""
    return pg.ColorMap(
        pos  =[0.00, 0.20, 0.50, 0.80, 1.00],
        color=[(  0,   0,   0, 255),
               (  0,  25,   0, 255),
               (  0, 110,  15, 255),
               ( 20, 200,  40, 255),
               (160, 255, 160, 255)],
    )


def cmap_crimson() -> pg.ColorMap:
    """Crimson / red phosphor — high urgency look."""
    return pg.ColorMap(
        pos  =[0.00, 0.25, 0.55, 0.85, 1.00],
        color=[( 10,   0,   0, 255),
               ( 60,   0,   0, 255),
               (160,  20,  20, 255),
               (240,  80,  60, 255),
               (255, 200, 180, 255)],
    )


def cmap_jet() -> pg.ColorMap:
    """MATLAB jet — classic analysis rainbow."""
    return pg.ColorMap(
        pos  =[0.000, 0.125, 0.375, 0.625, 0.875, 1.000],
        color=[(  0,   0, 127, 255),
               (  0,   0, 255, 255),
               (  0, 255, 255, 255),
               (255, 255,   0, 255),
               (255,   0,   0, 255),
               (127,   0,   0, 255)],
    )


# Default tactical map
cmap_phosphor = cmap_green_phosphor
cmap_demon    = cmap_green_phosphor

CMAPS = {
    "Green Phosphor": cmap_green_phosphor,
    "Night Vision":   cmap_night,
    "Amber":          cmap_amber,
    "Hot":            cmap_hot,
    "Crimson":        cmap_crimson,
    "Ice":            cmap_ice,
    "Bone":           cmap_bone,
    "Copper":         cmap_copper,
    "Gray":           cmap_gray,
    "Jet (analysis)": cmap_jet,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  TPSW — Two-Pass Split-Window noise-floor estimate (classic sonar)
# ═══════════════════════════════════════════════════════════════════════════════
def tpsw(spectrum: np.ndarray, window: int = 21, alpha: float = 0.3) -> np.ndarray:
    """
    Two-Pass Split-Window normalizer.

    Estimates a smooth noise floor by local averaging while excluding
    strong bins (tonals) on the second pass.  Returns normalized excess
    spectrum in dB-friendly linear ratio form (signal / floor).
    """
    x = np.asarray(spectrum, dtype=np.float64)
    n = len(x)
    if n < 3:
        return np.ones_like(x)
    w = max(3, window | 1)          # odd

    # Pass 1 — O(N) sliding-window local mean.
    # uniform_filter1d uses a running-sum kernel: O(N) vs O(N×w) for
    # np.convolve.  mode="nearest" replicates the edge, matching the
    # old np.pad(x, half, mode="edge") + convolve(valid) pattern.
    mean1 = uniform_filter1d(x, size=w, mode="nearest")

    # Clip peaks above mean*(1+alpha) so tonals don't inflate the floor
    clipped = np.minimum(x, mean1 * (1.0 + alpha))

    # Pass 2 — local mean of clipped spectrum = noise-floor estimate
    floor = uniform_filter1d(clipped, size=w, mode="nearest")
    np.maximum(floor, 1e-12, out=floor)
    return x / floor


def robust_floor(spectrum: np.ndarray, window: int = 51,
                 percentile: float = 30.0) -> np.ndarray:
    """
    Order-statistic (percentile) noise-floor normalizer.

    Far more robust to strong tonals than classic TPSW because the floor is
    taken from a low percentile of the local window — lines almost never
    pull the estimate up.  Returns signal / floor (linear ratio).
    """
    x = np.asarray(spectrum, dtype=np.float64)
    n = len(x)
    if n < 5:
        return np.ones_like(x)
    w = max(5, int(window) | 1)
    # percentile_filter is fast enough for real-time FFT_N ≤ 8192
    floor = percentile_filter(x, percentile, size=w, mode="nearest")
    floor = np.maximum(floor, 1e-12)
    return x / floor


def _os_cfar_kernel_py(x: np.ndarray, train: int, guard: int,
                        rank: float) -> np.ndarray:
    """
    OS-CFAR inner loop — pure-NumPy/Python fallback kept for Numba JIT.

    Uses a pre-allocated scratch buffer and in-place sort instead of
    Python lists / np.concatenate / np.partition so that the same source
    can be decorated with @numba.njit (every operation is nopython-safe).
    """
    n     = len(x)
    floor = np.empty(n, dtype=np.float64)

    for i in _prange(n):
        buf = np.empty(2 * train, dtype=np.float64)   # per-iteration: safe for parallel
        left_end    = i - guard
        left_start  = left_end - train
        right_start = i + guard + 1
        right_end   = right_start + train

        count = 0

        # Left training cells (guard zone excluded)
        ls = max(0, left_start)
        le = left_end
        if le > 0 and ls < le:
            for j in range(ls, le):
                buf[count] = x[j]
                count += 1

        # Right training cells (guard zone excluded)
        rs = right_start
        re = min(n, right_end)
        if rs < n and rs < re:
            for j in range(rs, re):
                buf[count] = x[j]
                count += 1

        if count == 0:
            floor[i] = x[i]
            continue

        # k-th order statistic via in-place sort of the occupied slice
        k    = int(round(rank * (count - 1)))
        view = buf[:count]
        view.sort()
        floor[i] = view[k]

    return floor


def _os_cfar_kernel_vec(x: np.ndarray, train: int, guard: int,
                         rank: float) -> np.ndarray:
    """
    Fully-vectorised OS-CFAR kernel — no Python-level loops.

    Builds all training windows at once with sliding_window_view, strips
    the guard zone via a boolean mask, then finds the k-th order statistic
    across all bins in a single np.partition call (C-level, O(n * 2*train)).

    ~10-30× faster than the Python loop version when Numba is unavailable.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n     = len(x)
    w     = train + guard          # one-sided half-window
    pad   = np.pad(x, w, mode='edge')
    # views: (n, 2*w+1) — every bin's symmetric window including guard zone
    views = sliding_window_view(pad, 2 * w + 1)

    # Build a mask that removes the central 2*guard+1 guard cells, keeping
    # exactly 2*train training cells per row.
    mask = np.ones(2 * w + 1, dtype=bool)
    mask[w - guard: w + guard + 1] = False   # zero out guard zone
    training = np.ascontiguousarray(views[:, mask])  # (n, 2*train)

    # k-th order statistic across training cells — single C call
    n_train = training.shape[1]
    k = int(round(rank * (n_train - 1)))
    k = max(0, min(k, n_train - 1))
    np.partition(training, k, axis=1, out=training)
    floor = training[:, k].astype(np.float64)
    return floor


if _NUMBA:
    _os_cfar_kernel = _numba.njit(
        cache=True, fastmath=True, boundscheck=False, parallel=True
    )(_os_cfar_kernel_py)
    # Warm up with the actual runtime spectrum sizes so Numba compiles and
    # caches both before the first real audio frame (avoids mid-stream stall).
    _os_cfar_kernel(np.ones(FFT_N // 2 + 1,       dtype=np.float64), 40, 3, 0.75)
    _os_cfar_kernel(np.ones(DEMON_FFT_N // 2 + 1, dtype=np.float64), 28, 2, 0.70)
else:
    # Vectorised NumPy fallback — ~10-30× faster than the Python loop.
    _os_cfar_kernel = _os_cfar_kernel_vec


def os_cfar_floor(spectrum: np.ndarray, train: int = 40, guard: int = 3,
                  rank: float = 0.75) -> np.ndarray:
    """
    Ordered-Statistic CFAR noise-floor estimator with guard cells.

    For every bin under test (CUT):
      • exclude a guard zone on each side (so the tonal itself does not
        bias the estimate)
      • collect the remaining training cells
      • take the `rank`-th order statistic as the local noise floor

    Returns signal / floor (linear ratio).  The inner loop is JIT-compiled
    with Numba when available (pip install numba); falls back to pure
    Python/NumPy transparently if Numba is not installed.
    """
    x = np.asarray(spectrum, dtype=np.float64)
    n = len(x)
    if n < 2 * (train + guard) + 1:
        # Fall back to a simple global percentile when the spectrum is tiny
        floor_val = float(np.percentile(x, 30.0))
        return x / max(floor_val, 1e-12)

    train = max(4, int(train))
    guard = max(0, int(guard))
    rank  = float(np.clip(rank, 0.05, 0.95))

    floor = _os_cfar_kernel(x, train, guard, rank)
    np.maximum(floor, 1e-12, out=floor)
    return x / floor


# ═══════════════════════════════════════════════════════════════════════════════
#  EIGEN — cross-frame PCA/SVD subspace denoiser
# ═══════════════════════════════════════════════════════════════════════════════
class _EigenDenoiser:
    """
    Rolling low-rank (PCA/SVD) denoiser for a stream of spectra.

    TPSW / Robust / OS-CFAR all estimate a noise floor from a single
    frame — they only look *across frequency*.  This looks *across time*
    instead, which is a genuinely different and complementary source of
    signal: a persistent tonal sits in the same bin frame after frame, so
    across a short window of history it is almost entirely captured by
    the first few singular vectors of the frame-by-frame data matrix.
    Broadband noise that is incoherent from one frame to the next spreads
    its energy roughly evenly across every singular value. Reconstructing
    each new frame from only the top `rank` eigen-spectra of that window
    keeps the persistent structure and drops most of the incoherent tail.

    Expects its input already floor-normalized (TPSW/Robust/OS-CFAR
    output, signal/floor ratio ~1 at the noise floor) — that puts every
    bin on a comparable scale so the SVD reflects genuine temporal
    correlation rather than just which bins happen to carry more raw
    energy (e.g. low frequencies in a 1/f-ish spectrum).

    Trade-off worth knowing: because rank is fixed and small, a tonal
    that is weak or only lasts a frame or two may not make the cut and
    gets smoothed away along with the noise. This sharpens strong,
    stable lines; it isn't a strict superset of the per-frame floor
    normalizers and can be turned off if you need every faint transient
    preserved.

    The basis is refreshed only every `recalc` frames (not every frame)
    since it barely changes over a few tens of milliseconds — this is
    the same trade AUTO LVL already makes for its percentile recompute.
    """

    def __init__(self, n_bins: int, hist: int = PCA_HIST,
                 rank: int = PCA_RANK, recalc: int = PCA_RECALC) -> None:
        self.hist   = max(4, int(hist))
        self.rank   = max(1, int(rank))
        self.recalc = max(1, int(recalc))
        self._buf   = np.zeros((self.hist, n_bins), dtype=np.float64)
        self._n     = 0     # rows filled so far (saturates at hist)
        self._pos   = 0     # circular write index
        self._vt    = None  # cached (rank, n_bins) eigen-spectra basis
        self._age   = 0     # frames since the basis was last refreshed

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._n = 0
        self._pos = 0
        self._vt = None
        self._age = 0

    def denoise(self, row: np.ndarray) -> np.ndarray:
        """Push one new spectrum in, return its low-rank reconstruction."""
        self._buf[self._pos] = row
        self._pos = (self._pos + 1) % self.hist
        self._n = min(self._n + 1, self.hist)

        # Not enough history yet to trust a subspace estimate — pass through.
        if self._n < max(4, self.rank + 1):
            return row

        self._age += 1
        if self._vt is None or self._age >= self.recalc:
            self._age = 0
            if self._n < self.hist:
                mat = self._buf[:self._n]
            else:
                # Oldest→newest ordering (doesn't affect SVD, kept for clarity)
                mat = np.concatenate(
                    (self._buf[self._pos:], self._buf[:self._pos]), axis=0
                )
            try:
                _, _, vt = np.linalg.svd(mat, full_matrices=False)
            except np.linalg.LinAlgError:
                return row
            r = min(self.rank, vt.shape[0])
            self._vt = vt[:r]

        coeffs = row @ self._vt.T
        return coeffs @ self._vt


# ═══════════════════════════════════════════════════════════════════════════════
#  ALE — Adaptive Line Enhancer (LMS)
# ═══════════════════════════════════════════════════════════════════════════════
class AdaptiveLineEnhancer:
    """
    Fast frequency-domain Adaptive Line Enhancer (FDAF-style, one FFT/block).

    Delay decorrelates broadband noise; adaptive weights lock onto narrow
    spectral peaks (tonals). Output is the enhanced line estimate.
    Fully vectorized — real-time safe.
    """

    def __init__(self, n_fft: int = 512, mu: float = 0.3, delay: float = 16) -> None:
        self.n_fft = int(n_fft)
        self.mu = float(mu)
        self.delay = max(1, int(delay))
        self.W = np.zeros(self.n_fft // 2 + 1, dtype=np.complex128)
        self._overlap = np.zeros(self.n_fft, dtype=np.float64)
        self._in_hist = np.zeros(self.n_fft + self.delay, dtype=np.float64)
        # Periodic Hanning for analysis: 0.5*(1-cos(2πn/N)).
        # Unlike np.hanning(N) which uses denominator N-1, the periodic form
        # satisfies w[n] + w[n+hop] = 1 exactly at 50% overlap (COLA).
        # With rectangular synthesis (win_s = 1), the OLA condition simplifies to
        # sum_k win_a[n - k*hop] = 1, which periodic Hanning satisfies exactly.
        # This eliminates the hop-rate tonal (48000/hop Hz) completely.
        _n = np.arange(self.n_fft, dtype=np.float64)
        self._win = 0.5 * (1.0 - np.cos(2.0 * np.pi * _n / self.n_fft))
        self._P = np.ones(self.n_fft // 2 + 1, dtype=np.float64)  # power estimate
        self._U_mag2 = np.empty(self.n_fft // 2 + 1, dtype=np.float64)  # scratch for P update

        # ── Pre-planned FFTW transforms (skipped gracefully without pyFFTW) ───
        # A single forward plan is reused for both the desired (d) and reference
        # (u) frames; the output is copied after the first call so the second
        # call does not overwrite it.  The inverse plan is used for y_time.
        if _PYFFTW:
            _a = pyfftw.empty_aligned(self.n_fft, dtype='float64')
            _b = pyfftw.empty_aligned(self.n_fft // 2 + 1, dtype='complex128')
            self._fftw_fwd = pyfftw.builders.rfft(
                _a, planner_effort='FFTW_MEASURE', threads=1
            )
            self._fftw_inv = pyfftw.builders.irfft(
                _b, n=self.n_fft, planner_effort='FFTW_MEASURE', threads=1
            )
        else:
            self._fftw_fwd = None
            self._fftw_inv = None

    def reset(self) -> None:
        self.W[:] = 0.0
        self._overlap[:] = 0.0
        self._in_hist[:] = 0.0
        self._P[:] = 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        out = np.zeros_like(x)
        hop = self.n_fft // 2
        # Stream through overlapping frames
        pos = 0
        while pos < len(x):
            take = min(hop, len(x) - pos)
            # Push into history — in-place shift, no Python allocation
            self._in_hist[:-take] = self._in_hist[take:]
            self._in_hist[-take:] = x[pos:pos + take]

            # Desired = recent frame; reference = delayed frame
            d = self._in_hist[-self.n_fft:] * self._win
            u = self._in_hist[-self.n_fft - self.delay: -self.delay] * self._win
            if len(u) != self.n_fft:
                pos += take
                continue

            if self._fftw_fwd is not None:
                D = self._fftw_fwd(d).copy()   # copy: same plan reused for U
                U = self._fftw_fwd(u).copy()   # copy: U used after y_time call
            else:
                D = _np_fft.rfft(d)
                U = _np_fft.rfft(u)
            Y = self.W * U
            if self._fftw_inv is not None:
                y_time = self._fftw_inv(Y)
            else:
                y_time = _np_fft.irfft(Y, n=self.n_fft)
            # Overlap-add with rectangular synthesis (no synthesis window).
            # The analysis window (periodic Hanning) already satisfies COLA at
            # 50% overlap: w[n]+w[n+hop]=1.  Adding win_s=Hanning on top would
            # create Hanning² whose OLA sum is 0.5+0.5cos²θ — not constant —
            # which is what was generating the residual 187.5 Hz artifact.
            self._overlap += y_time
            n_out = min(hop, len(out) - pos)
            if n_out > 0:
                out[pos:pos + n_out] = self._overlap[:n_out]
            # Shift overlap buffer by one hop
            self._overlap[:hop] = self._overlap[hop:]
            self._overlap[hop:] = 0.0

            # NLMS weight update in frequency domain
            E = D - Y
            np.abs(U, out=self._U_mag2)       # |U|        — no alloc
            self._U_mag2 **= 2                 # |U|²       — in-place
            self._P *= 0.85                    # decay      — in-place
            self._P += 0.15 * self._U_mag2     # accumulate — one temp (scalar mul)
            self.W += self.mu * E * np.conj(U) / (self._P + 1e-6)

            pos += take

        # Blend enhanced lines with a fraction of the original so broadband
        # context remains visible.  No per-chunk amplitude rescaling here —
        # that caused AM modulation at the CHUNK rate (48000/512 = 93.75 Hz)
        # which appeared as a solid bar in DEMON.  The NLMS weight update
        # already normalises gain via the per-bin power estimate _P.
        return (0.75 * out + 0.25 * x).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════
class AudioOut:
    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._stream = None
        self.enabled: bool = True
        self._n_ch: int = 1

    def start(self, sr: int, channels: int = 1) -> None:
        self.stop()
        self._n_ch = max(1, int(channels))
        self._stream = sd.OutputStream(
            samplerate=sr, channels=self._n_ch, dtype="float32",
            blocksize=CHUNK, callback=self._cb, latency="low",
        )
        self._stream.start()

    def _cb(self, out: np.ndarray, frames: int, _t, _st) -> None:
        if not self.enabled:
            out[:] = 0.0
            return
        try:
            chunk = self._q.get_nowait()
            n = min(len(chunk), frames)
            if chunk.ndim == 1:
                # mono source — broadcast to all output channels
                out[:n] = chunk[:n, np.newaxis]
            else:
                ch = min(chunk.shape[1], self._n_ch)
                out[:n, :ch] = chunk[:n, :ch]
                if ch < self._n_ch:
                    out[:n, ch:] = 0.0
            if n < frames:
                out[n:] = 0.0
        except queue.Empty:
            out[:] = 0.0

    def push(self, chunk: np.ndarray) -> None:
        try:
            self._q.put_nowait(chunk.astype(np.float32))
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait(chunk.astype(np.float32))
            except queue.Empty:
                pass

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


# ═══════════════════════════════════════════════════════════════════════════════
#  DSP WORKER — LOFAR + DEMON
# ═══════════════════════════════════════════════════════════════════════════════
class DSPWorker(QObject):
    # (lofar_db, demon_db)
    spectrum_ready = pyqtSignal(object, object)

    def __init__(self) -> None:
        super().__init__()
        self.audio_q: _AudioQueue = _AudioQueue(maxsize=128)
        self.sr: int = AUDIO_SR          # capture / bandpass-filter / audio-out rate
        self.lofar_sr: float = LOFAR_SR  # LOFAR analysis rate (post-decimate)
        self._lofar_ds: int  = AUDIO_SR // LOFAR_SR   # decimation factor = 3

        # Anti-alias LP state for the LOFAR decimation stage
        self._lofar_aa_sos = None
        self._lofar_aa_zi  = None
        self._lofar_aa_stale: bool = True

        self.aout: AudioOut | None = None

        self._running = False
        self._lo = 300.0
        self._hi = 3000.0
        self._sos = None
        self._zi_list = None   # list of per-channel zi states for audio-out bandpass
        self._filter_stale = True
        self._last_sample = 0.0

        # Normalization mode: "off" | "tpsw" | "robust" | "oscfar"
        self._norm_mode = "robust"
        self._use_ale = False
        self._ale = AdaptiveLineEnhancer(n_fft=512, mu=0.3, delay=16)

        # Cross-frame PCA/SVD subspace denoiser — applied to the
        # floor-normalized ratio (TPSW/Robust/OS-CFAR), before dB
        # conversion. No effect when NORM is Off.
        self._use_eigen = False
        self._lofar_eigen = _EigenDenoiser(FFT_N // 2 + 1)
        self._demon_eigen = _EigenDenoiser(DEMON_FFT_N // 2 + 1)

        # DEMON: bandpass → envelope → decimate to fine bin spacing in 0..DEMON_DISP_HI
        self._demon_bp_lo = 800.0
        self._demon_bp_hi = 6000.0
        self._demon_sos = None
        self._demon_zi = None
        self._demon_env_lp = None
        self._demon_env_zi = None
        self._demon_stale = True
        self._demon_ds = 1              # decimation factor
        self._demon_ds_sr = DEMON_DS_SR # target envelope sample rate
        self._demon_ds_phase = 0        # decimation phase counter
        self._env_buf = np.zeros(DEMON_FFT_N, dtype=np.float64)
        self._env_pos = 0
        # FIX: gate FFT on a hop counter so we only push a new DEMON row when
        # enough fresh envelope data has accumulated.  Every chunk contributes
        # only ~5 decimated samples; without this, 99 % of the FFT window is
        # identical frame-to-frame, producing permanent vertical pinstripes.
        self._demon_hop_acc = 0
        self._demon_hop     = 32   # higher overlap → faster DEMON updates
        self._demon_smooth: np.ndarray | None = None

        # Multi-sub-band DEMON ("SUB" mode).
        # The BP region is split into _N_SUB independent sub-bands.  Each
        # sub-band produces its own DEMON power spectrum per hop; averaging N
        # power spectra then taking √ gives √N better amplitude SNR for
        # coherent modulations (shaft / blade rates) because noise is
        # incoherent across bands while the propeller modulation is not.
        # With _N_SUB=8 that is a 2.8× amplitude gain — actually visible.
        _N_SUB = 8
        self._demon_sub_enabled: bool = True
        self._demon_sub_sos:   list = [None] * _N_SUB   # per-band BP SOS
        self._demon_sub_bp_zi: list = [None] * _N_SUB   # per-band BP filter state
        self._demon_sub_lp_zi: list = [None] * _N_SUB   # per-band LP filter state
        self._demon_sub_buf:   np.ndarray = np.zeros((_N_SUB, DEMON_FFT_N), dtype=np.float64)
        self._demon_n_sub:     int = _N_SUB
        self._demon_sub_stale: bool = True

        self._win = np.hanning(FFT_N).astype(np.float32)
        self._demon_win = np.hanning(DEMON_FFT_N).astype(np.float64)

        # ── ZoomFFT state ───────────────────────────────────────
        # Audio ring buffer: ZOOM_BUF_MULT × FFT_N samples at lofar_sr.
        # When zoomed we feed a longer slice to ZoomFFT so resolution
        # scales with zoom factor.
        self._audio_ring = np.zeros(FFT_N * ZOOM_BUF_MULT, dtype=np.float32)
        self._win_cache: dict = {}          # cached hanning windows by length

        # ── Pre-planned FFTW transforms for the two fixed FFT sizes ───────────
        # LOFAR: FFT_N float32 samples → FFT_N//2+1 complex64 magnitudes
        # DEMON: DEMON_FFT_N float64 envelope samples → complex128 magnitudes
        # Both builders are called with the windowed input array directly;
        # pyfftw copies the data into its aligned buffer before executing.
        if _PYFFTW:
            _lofar_in = pyfftw.empty_aligned(FFT_N, dtype='float32')
            self._lofar_fftw = pyfftw.builders.rfft(
                _lofar_in, planner_effort='FFTW_MEASURE', threads=1
            )
            _demon_in = pyfftw.empty_aligned(DEMON_FFT_N, dtype='float64')
            self._demon_fftw = pyfftw.builders.rfft(
                _demon_in, planner_effort='FFTW_MEASURE', threads=1
            )
        else:
            self._lofar_fftw = None
            self._demon_fftw = None

        self._lofar_zoom_lo: float | None = None
        self._lofar_zoom_hi: float | None = None
        self._lofar_zoom_stale: bool = False
        self._lofar_zoom_n: int = FFT_N
        self._lofar_zoom_obj = None

        self._demon_zoom_lo: float | None = None
        self._demon_zoom_hi: float | None = None
        self._demon_zoom_stale: bool = False
        self._demon_zoom_obj = None

    def set_band(self, lo: float, hi: float) -> None:
        self._lo, self._hi = float(lo), float(hi)
        # FIX: keep DEMON bandpass in sync with the reticule so moving the
        # green overlay actually shifts the demodulation band.
        self._demon_bp_lo, self._demon_bp_hi = float(lo), float(hi)
        self._filter_stale  = True
        self._demon_stale   = True
        self._demon_sub_stale = True   # sub-band limits derived from bp_lo/hi
        self._demon_eigen.reset()      # carrier band moved — history is stale

    def set_sr(self, sr: int) -> None:
        self.sr = sr
        # Recompute LOFAR decimation: keep lofar_sr just above 2×DISP_HI.
        # e.g. 48000 → ds=3, lofar_sr=16000; 44100 → ds=2, lofar_sr=22050;
        #      19200 → ds=1, lofar_sr=19200 (no decimation needed).
        self._lofar_ds   = max(1, sr // int(2 * DISP_HI))
        self.lofar_sr    = sr / self._lofar_ds
        self._lofar_aa_stale = True
        self._lofar_aa_zi    = None
        self._filter_stale = True
        self._demon_stale = True
        self._ale.reset()
        self._lofar_eigen.reset()
        self._demon_eigen.reset()
        self._env_buf[:] = 0.0
        self._env_pos = 0
        self._demon_ds_phase = 0
        self._demon_hop_acc  = 0
        self._demon_smooth   = None
        # Invalidate zoom objects — they embed the sample rate
        self._audio_ring[:] = 0.0
        self._lofar_zoom_stale = True
        self._lofar_zoom_obj   = None
        self._demon_zoom_stale = True
        self._demon_zoom_obj   = None

    def _build_lofar_antialias(self) -> None:
        """Butter LP anti-alias filter at 90 % of the LOFAR Nyquist,
        normalised to the full audio rate.  Skipped when ds == 1."""
        self._lofar_aa_stale = False
        if self._lofar_ds <= 1:
            self._lofar_aa_sos = None
            self._lofar_aa_zi  = None
            return
        nyq    = self.sr / 2.0
        cutoff = min((self.lofar_sr / 2.0 * 0.9) / nyq, 0.99)
        self._lofar_aa_sos = sp.butter(8, cutoff, btype='low', output='sos')
        self._lofar_aa_zi  = None

    def set_norm(self, mode: str) -> None:
        """mode: 'off' | 'tpsw' | 'robust' | 'oscfar'"""
        mode = str(mode).lower().strip()
        if mode not in ("off", "tpsw", "robust", "oscfar"):
            mode = "robust"
        self._norm_mode = mode
        # A different normalizer rescales the ratio differently — the eigen
        # history would otherwise mix incompatible scales.
        self._lofar_eigen.reset()
        self._demon_eigen.reset()

    def set_ale(self, on: bool) -> None:
        self._use_ale = bool(on)
        if on:
            self._ale.reset()

    def set_eigen(self, on: bool) -> None:
        """Toggle the cross-frame PCA/SVD subspace denoiser (LOFAR + DEMON)."""
        self._use_eigen = bool(on)
        if on:
            self._lofar_eigen.reset()
            self._demon_eigen.reset()

    def set_demon_sub(self, on: bool) -> None:
        """Toggle multi-sub-band DEMON coherent averaging (SUB button).

        When enabled the broadband DEMON bandpass is split into _N_SUB
        independent sub-bands; their power spectra are averaged per hop and
        √-ed back to magnitude.  Noise is incoherent across bands; propeller
        modulation is coherent → √N SNR gain in amplitude.
        """
        self._demon_sub_enabled = bool(on)
        if on:
            self._demon_sub_stale = True   # rebuild filters if band changed

    def _build_demon_sub_filters(self) -> None:
        """Build the N sub-band BP SOS filters spanning [_demon_bp_lo, _demon_bp_hi]."""
        n   = self._demon_n_sub
        lo  = self._demon_bp_lo
        hi  = self._demon_bp_hi
        bw  = (hi - lo) / n
        nyq = self.sr / 2.0
        for i in range(n):
            f_lo = lo + i * bw
            f_hi = f_lo + bw
            if 0 < f_lo < nyq and 0 < f_hi <= nyq and f_hi > f_lo + 10:
                try:
                    # Floor the low edge the same way _build_demon_filters()
                    # does — a bandpass whose low cutoff sits right on DC
                    # normalizes to coefficients so close to a pole at z=1
                    # that sp.butter() succeeds but sosfilt_zi() later raises
                    # ValueError (this is what a narrow, near-0 Hz reticule
                    # band used to crash the DSP thread with).
                    lo_n = max(f_lo / nyq, 0.001)
                    hi_n = min(f_hi / nyq, 0.9999)
                    if hi_n <= lo_n:
                        raise ValueError("degenerate band after normalization")
                    sos = sp.butter(4, [lo_n, hi_n], btype="bandpass", output="sos")
                    sp.sosfilt_zi(sos)   # validate stability before storing it
                    self._demon_sub_sos[i] = sos
                except Exception:
                    self._demon_sub_sos[i] = None
            else:
                self._demon_sub_sos[i] = None
            self._demon_sub_bp_zi[i] = None
            self._demon_sub_lp_zi[i] = None
        self._demon_sub_buf[:] = 0.0
        self._demon_sub_stale  = False

    # ── ZoomFFT control ─────────────────────────────────────────────────────

    def set_lofar_zoom(self, lo: float, hi: float) -> None:
        """Called by the GUI whenever the LOFAR waterfall zoom changes.
        At full display range the standard FFT is used; otherwise a ZoomFFT
        with an adaptively longer window gives resolution ∝ zoom factor."""
        if abs(lo - DISP_LO) < 1.0 and abs(hi - DISP_HI) < 1.0:
            self._lofar_zoom_lo = None
            self._lofar_zoom_hi = None
        else:
            self._lofar_zoom_lo = float(lo)
            self._lofar_zoom_hi = float(hi)
        self._lofar_zoom_stale = True
        self._lofar_zoom_obj   = None
        self._lofar_eigen.reset()   # bin↔frequency mapping just changed

    def set_demon_zoom(self, lo: float, hi: float) -> None:
        """Called when the DEMON waterfall zoom changes.  The existing long
        envelope buffer already provides true resolution improvement."""
        if lo < 0.5 and abs(hi - DEMON_DISP_HI) < 1.0:
            self._demon_zoom_lo = None
            self._demon_zoom_hi = None
        else:
            self._demon_zoom_lo = float(lo)
            self._demon_zoom_hi = float(hi)
        self._demon_zoom_stale = True
        self._demon_zoom_obj   = None
        self._demon_eigen.reset()   # bin↔frequency mapping just changed
        # Sub-band buffers are in the audio-frequency domain, not the display
        # frequency axis, so they don't need flushing on zoom changes.
        # But reset LP state so there are no transients after zoom shift.
        self._demon_sub_lp_zi = [None] * self._demon_n_sub

    def _hanning(self, n: int) -> np.ndarray:
        """Return a cached Hanning window of length n."""
        if n not in self._win_cache:
            self._win_cache[n] = np.hanning(n).astype(np.float32)
        return self._win_cache[n]

    def _build_lofar_zoom(self, n: int) -> None:
        lo, hi = self._lofar_zoom_lo, self._lofar_zoom_hi
        m = FFT_N // 2 + 1      # same bin count as rfft so Waterfall._buf fits
        try:
            # fs is the rate of the ring buffer data (lofar_sr, not audio sr)
            self._lofar_zoom_obj = ZoomFFT(n, [lo, hi], m=m, fs=self.lofar_sr)
            self._lofar_zoom_n   = n
        except Exception:
            self._lofar_zoom_obj = None
        self._lofar_zoom_stale = False

    def _build_demon_zoom(self) -> None:
        lo, hi = self._demon_zoom_lo, self._demon_zoom_hi
        m = DEMON_FFT_N // 2 + 1
        try:
            self._demon_zoom_obj = ZoomFFT(
                DEMON_FFT_N, [lo, hi], m=m, fs=self._demon_ds_sr
            )
        except Exception:
            self._demon_zoom_obj = None
        self._demon_zoom_stale = False

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                chunk = self.audio_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if chunk is None:
                break

            raw  = chunk                      # original (stereo or mono) for audio out
            mono = (raw.mean(axis=1) if raw.ndim > 1 else raw).astype(np.float32)
            if len(mono) == 0:
                continue
            self._last_sample = float(mono[-1])

            # Optional ALE (block LMS) — runs at full AUDIO_SR
            proc = self._ale.process(mono) if self._use_ale else mono

            # ── LOFAR — anti-alias + decimate to lofar_sr, then FFT ────────
            # Build (or rebuild) the anti-alias filter when the rate changes.
            if self._lofar_aa_stale:
                self._build_lofar_antialias()

            # Apply LP anti-alias filter then decimate (skipped when ds == 1).
            # LOFAR reads raw mono — not the ALE output — so that the adaptive
            # spectral coloring from the LMS weights doesn't paint vertical
            # stripes on the waterfall.  DEMON keeps the ALE path (proc) where
            # it genuinely helps by enhancing blade-rate harmonics.
            if self._lofar_ds > 1 and self._lofar_aa_sos is not None:
                if self._lofar_aa_zi is None:
                    self._lofar_aa_zi = (
                        sp.sosfilt_zi(self._lofar_aa_sos) * float(mono[0])
                    )
                aa, self._lofar_aa_zi = sp.sosfilt(
                    self._lofar_aa_sos, mono, zi=self._lofar_aa_zi
                )
                proc_lofar = aa[::self._lofar_ds]
            else:
                proc_lofar = mono     # ds == 1, audio_sr already ≤ 2×DISP_HI

            # Fill ring buffer with lofar_sr data so ZoomFFT can reach back
            # further than a single chunk when the window needs to be longer.
            n_new = len(proc_lofar)
            self._audio_ring[:-n_new] = self._audio_ring[n_new:]
            self._audio_ring[-n_new:] = proc_lofar

            if self._lofar_zoom_lo is not None:
                # Zoomed: grow the analysis window proportional to zoom factor
                # so frequency resolution scales as 1/zoom_span (true CZT gain,
                # not just interpolation).  Capped at ZOOM_BUF_MULT × FFT_N.
                zoom_span   = max(1.0, self._lofar_zoom_hi - self._lofar_zoom_lo)
                zoom_factor = DISP_SPAN / zoom_span
                n_zoom = int(2 ** round(np.log2(FFT_N * zoom_factor)))
                n_zoom = max(FFT_N, min(n_zoom, len(self._audio_ring)))
                if self._lofar_zoom_stale or n_zoom != self._lofar_zoom_n:
                    self._build_lofar_zoom(n_zoom)
                win = self._hanning(n_zoom)
                if self._lofar_zoom_obj is not None:
                    mag = np.abs(self._lofar_zoom_obj(
                        self._audio_ring[-n_zoom:] * win
                    )) + 1e-12
                else:
                    buf = self._audio_ring[-FFT_N:]
                    _win_buf = buf * self._win
                    mag = np.abs(self._lofar_fftw(_win_buf)
                                 if self._lofar_fftw is not None
                                 else _np_fft.rfft(_win_buf)) + 1e-12
                lofar_f_lo = self._lofar_zoom_lo
                lofar_f_hi = self._lofar_zoom_hi
            else:
                # Full range: FFT over the last FFT_N real lofar_sr samples.
                # Using the ring buffer (not zero-padded proc) gives true
                # SR/FFT_N resolution rather than SR/CHUNK.
                if self._lofar_zoom_stale:
                    self._lofar_zoom_obj   = None
                    self._lofar_zoom_stale = False
                buf = self._audio_ring[-FFT_N:]
                _win_buf = buf * self._win
                mag = np.abs(self._lofar_fftw(_win_buf)
                             if self._lofar_fftw is not None
                             else _np_fft.rfft(_win_buf)) + 1e-12
                lofar_f_lo = 0.0
                lofar_f_hi = float(self.lofar_sr / 2.0)

            if self._norm_mode == "robust":
                # Wider window + low percentile → robust against strong tonals
                norm = robust_floor(mag, window=71, percentile=28.0)
            elif self._norm_mode == "oscfar":
                # Full OS-CFAR with guard cells (best line isolation)
                norm = os_cfar_floor(mag, train=36, guard=3, rank=0.72)
            elif self._norm_mode == "tpsw":
                norm = tpsw(mag, window=41, alpha=0.25)
            else:
                norm = None
            if norm is not None:
                if self._use_eigen:
                    norm = np.maximum(self._lofar_eigen.denoise(norm), 1e-9)
                # Reuse norm's fresh buffer: saves 2 temporaries per frame
                np.log10(norm, out=norm)
                norm *= 20.0
                lofar_db = norm
            else:
                np.log10(mag, out=mag)
                mag *= 20.0
                lofar_db = mag - float(np.median(mag))

            # ── DEMON ──────────────────────────────────────────────
            # Classical chain: BP → |·| → LP → decimate → STFT of envelope.
            # When SUB is ON the BP region is split into _N_SUB independent
            # sub-bands; their power spectra are averaged per hop and √-ed back
            # to magnitude.  Noise is incoherent across bands; propeller
            # modulation is coherent → √N_SUB amplitude SNR gain.
            if self._demon_stale:
                self._build_demon_filters()
            if self._demon_sub_enabled and self._demon_sub_stale:
                self._build_demon_sub_filters()

            if self._demon_sos is not None:
                try:
                    if self._demon_zi is None:
                        self._demon_zi = sp.sosfilt_zi(self._demon_sos) * float(proc[0])
                    bp, self._demon_zi = sp.sosfilt(self._demon_sos, proc, zi=self._demon_zi)
                except Exception:
                    # A pathological band (e.g. right on DC) can pass sp.butter()
                    # but fail sosfilt_zi()'s stability check — disable this
                    # filter rather than take the whole DSP thread down with it.
                    self._demon_sos = None
                    self._demon_zi = None
                    bp = proc
            else:
                bp = proc

            env = np.abs(bp)
            if self._demon_env_lp is not None:
                try:
                    if self._demon_env_zi is None:
                        self._demon_env_zi = sp.sosfilt_zi(self._demon_env_lp) * float(env[0])
                    env, self._demon_env_zi = sp.sosfilt(
                        self._demon_env_lp, env, zi=self._demon_env_zi
                    )
                except Exception:
                    self._demon_env_lp = None
                    self._demon_env_zi = None

            # Decimate envelope to ~demon_ds_sr
            ds = max(1, self._demon_ds)
            old_ds_phase = self._demon_ds_phase   # sub-bands need the same phase
            env_ds = env[old_ds_phase::ds]
            self._demon_ds_phase = (self._demon_ds_phase + len(env)) % ds

            n_new = len(env_ds)
            demon_pkg = None
            if n_new > 0:
                self._env_buf[:-n_new] = self._env_buf[n_new:]
                self._env_buf[-n_new:] = env_ds
                self._demon_hop_acc += n_new

                # ── Sub-band accumulation (parallel with main band) ─────
                if self._demon_sub_enabled:
                    for i, sub_sos in enumerate(self._demon_sub_sos):
                        if sub_sos is None:
                            continue
                        try:
                            if self._demon_sub_bp_zi[i] is None:
                                self._demon_sub_bp_zi[i] = (
                                    sp.sosfilt_zi(sub_sos) * float(bp[0])
                                )
                            sub_bp, self._demon_sub_bp_zi[i] = sp.sosfilt(
                                sub_sos, bp, zi=self._demon_sub_bp_zi[i]
                            )
                            sub_env = np.abs(sub_bp)
                            if self._demon_env_lp is not None:
                                if self._demon_sub_lp_zi[i] is None:
                                    self._demon_sub_lp_zi[i] = (
                                        sp.sosfilt_zi(self._demon_env_lp)
                                        * float(sub_env[0])
                                    )
                                sub_env, self._demon_sub_lp_zi[i] = sp.sosfilt(
                                    self._demon_env_lp, sub_env,
                                    zi=self._demon_sub_lp_zi[i],
                                )
                        except Exception:
                            # Same defense as the main DEMON filter above —
                            # drop this one sub-band instead of crashing the
                            # DSP thread; _build_demon_sub_filters() rebuilds
                            # it clean next time the band changes.
                            self._demon_sub_sos[i] = None
                            self._demon_sub_bp_zi[i] = None
                            self._demon_sub_lp_zi[i] = None
                            continue
                        # Same decimation phase as the main band
                        sub_ds = sub_env[old_ds_phase::ds]
                        n_sub = len(sub_ds)
                        if n_sub > 0:
                            self._demon_sub_buf[i, :-n_sub] = self._demon_sub_buf[i, n_sub:]
                            self._demon_sub_buf[i, -n_sub:] = sub_ds

                if self._demon_hop_acc >= self._demon_hop:
                    self._demon_hop_acc %= self._demon_hop

                    # Subtract mean before windowing to remove DC (mean
                    # envelope level).  The absolute-value envelope is always
                    # positive so bin 0 is huge; without this it bleeds into
                    # bins 1-4 via the Hanning main lobe → solid 0-2 Hz bar.
                    windowed = (self._env_buf - self._env_buf.mean()) * self._demon_win

                    if self._demon_zoom_lo is not None:
                        # Zoomed DEMON: ZoomFFT over [d_lo, d_hi].
                        # The env buffer is already DEMON_FFT_N long, so we
                        # get true resolution improvement with no extra cost.
                        if self._demon_zoom_stale:
                            self._build_demon_zoom()
                        if self._demon_zoom_obj is not None:
                            demon_mag = np.abs(
                                self._demon_zoom_obj(windowed)
                            ) + 1e-12
                            demon_f_lo = self._demon_zoom_lo
                            demon_f_hi = self._demon_zoom_hi
                        else:
                            demon_mag  = np.abs(self._demon_fftw(windowed)
                                                if self._demon_fftw is not None
                                                else _np_fft.rfft(windowed)) + 1e-12
                            demon_f_lo = 0.0
                            demon_f_hi = self._demon_ds_sr / 2.0
                    else:
                        if self._demon_zoom_stale:
                            self._demon_zoom_obj   = None
                            self._demon_zoom_stale = False
                        demon_mag  = np.abs(self._demon_fftw(windowed)
                                            if self._demon_fftw is not None
                                            else _np_fft.rfft(windowed)) + 1e-12
                        demon_f_lo = 0.0
                        demon_f_hi = self._demon_ds_sr / 2.0

                    # ── Multi-band coherent averaging (SUB mode) ────
                    # Average power across all valid sub-bands then √ back
                    # to magnitude.  Zoom is not applied here — the sub-bands
                    # already live in the audio frequency domain and the STFT
                    # frequency axis (0…demon_ds_sr/2) is the same regardless.
                    if self._demon_sub_enabled:
                        valid = [i for i, s in enumerate(self._demon_sub_sos)
                                 if s is not None]
                        if valid:
                            w_all  = ((self._demon_sub_buf[valid]
                                       - self._demon_sub_buf[valid].mean(
                                           axis=1, keepdims=True))
                                      * self._demon_win)          # (M, FFT_N), DC-removed
                            fft_all = _np_fft.rfft(w_all, axis=1)  # (M, bins)
                            avg_pwr = np.mean(np.abs(fft_all) ** 2, axis=0)
                            demon_mag  = np.sqrt(avg_pwr) + 1e-12
                            demon_f_lo = 0.0
                            demon_f_hi = self._demon_ds_sr / 2.0

                    if self._norm_mode == "robust":
                        # DEMON has finer bins and smoother continuum;
                        # use a wider relative window + lower percentile.
                        demon_norm = robust_floor(demon_mag, window=51, percentile=22.0)
                    elif self._norm_mode == "oscfar":
                        # Slightly fewer training cells – DEMON FFT is smaller
                        demon_norm = os_cfar_floor(demon_mag, train=28, guard=2, rank=0.70)
                    elif self._norm_mode == "tpsw":
                        # Retuned: larger window, gentler alpha
                        demon_norm = tpsw(demon_mag, window=31, alpha=0.22)
                    else:
                        demon_norm = None
                    if demon_norm is not None:
                        if self._use_eigen:
                            demon_norm = np.maximum(
                                self._demon_eigen.denoise(demon_norm), 1e-9
                            )
                        np.log10(demon_norm, out=demon_norm)
                        demon_norm *= 20.0
                        demon_db = demon_norm
                    else:
                        np.log10(demon_mag, out=demon_mag)
                        demon_mag *= 20.0
                        demon_db = demon_mag - float(np.median(demon_mag))

                    if self._demon_smooth is None \
                            or len(self._demon_smooth) != len(demon_db):
                        self._demon_smooth = demon_db.copy()
                    else:
                        # In-place EMA: 3 fewer temporaries per DEMON frame
                        demon_db *= 0.55
                        self._demon_smooth *= 0.45
                        self._demon_smooth += demon_db
                    demon_pkg = (self._demon_smooth, demon_f_lo, demon_f_hi)

            # Emit tuples (spec, f_lo, f_hi) so the Waterfall can set its
            # image rect correctly for both full-range and zoomed spectra.
            self.spectrum_ready.emit(
                (lofar_db, lofar_f_lo, lofar_f_hi),
                demon_pkg,
            )

            # ── Audio out — bandpass-filtered signal at AUDIO_SR ────────────
            # Filters each source channel independently to preserve stereo;
            # the mono downmix above is for the LOFAR/DEMON analysis path only.
            if self._filter_stale:
                self._build_filter()
            n_ch = raw.shape[1] if raw.ndim > 1 else 1
            if self._sos is not None:
                if self._zi_list is None or len(self._zi_list) != n_ch:
                    self._zi_list = [
                        sp.sosfilt_zi(self._sos) * self._last_sample
                        for _ in range(n_ch)
                    ]
                if raw.ndim > 1:
                    ch_out = []
                    for c in range(n_ch):
                        filt_ch, self._zi_list[c] = sp.sosfilt(
                            self._sos, raw[:, c].astype(np.float64), zi=self._zi_list[c]
                        )
                        ch_out.append(filt_ch)
                    filtered = np.stack(ch_out, axis=1).astype(np.float32)
                else:
                    filtered, self._zi_list[0] = sp.sosfilt(
                        self._sos, raw.astype(np.float64), zi=self._zi_list[0]
                    )
                    filtered = filtered.astype(np.float32)
            else:
                filtered = raw.astype(np.float32)

            if self.aout is not None and self.aout.enabled:
                if len(filtered):
                    self.aout.push(filtered)

    def _build_filter(self) -> None:
        self._filter_stale = False
        nyq = self.sr / 2.0
        lo_n = np.clip(self._lo / nyq, 0.001, 0.996)
        hi_n = np.clip(self._hi / nyq, lo_n + 0.004, 0.999)
        try:
            self._sos = sp.butter(4, [lo_n, hi_n], btype="band", output="sos")
            self._zi_list = None   # reset on filter rebuild
        except ValueError:
            self._sos = None
            self._zi_list = None

    def _build_demon_filters(self) -> None:
        self._demon_stale = False
        nyq = self.sr / 2.0
        lo = np.clip(self._demon_bp_lo / nyq, 0.001, 0.9)
        hi = np.clip(min(self._demon_bp_hi, nyq * 0.95) / nyq, lo + 0.01, 0.99)
        try:
            self._demon_sos = sp.butter(4, [lo, hi], btype="band", output="sos")
            self._demon_zi = None
            # Target envelope rate so DEMON_FFT_N spans ~0..DEMON_DISP_HI with fine bins
            # bin width ≈ DEMON_DS_SR / DEMON_FFT_N = 2000/4096 ≈ 0.49 Hz
            target_sr = DEMON_DS_SR
            self._demon_ds = max(1, int(round(self.sr / target_sr)))
            self._demon_ds_sr = self.sr / self._demon_ds
            # Anti-alias LP before decimation (cutoff slightly above DEMON_DISP_HI)
            lp_hz = min(DEMON_DISP_HI * 1.2, self._demon_ds_sr * 0.45)
            lp = lp_hz / nyq
            self._demon_env_lp = sp.butter(4, lp, btype="low", output="sos")
            self._demon_env_zi = None
            self._demon_ds_phase = 0
        except ValueError:
            self._demon_sos = None
            self._demon_env_lp = None
            self._demon_ds = 1


# ═══════════════════════════════════════════════════════════════════════════════
#  WATERFALL
# ═══════════════════════════════════════════════════════════════════════════════
class Waterfall(pg.PlotWidget):
    band_nudge   = pyqtSignal(float, float)  # retained for API compat
    zoom_changed = pyqtSignal(float, float)  # emitted on every zoom/pan/reset

    def __init__(self, title: str = "", sr: int = SR_DEF,
                 fft_n: int = FFT_N, history: int = HISTORY,
                 f_lo: float = DISP_LO, f_hi: float = DISP_HI,
                 cmap_fn=cmap_phosphor, show_freq_axis: bool = True) -> None:
        super().__init__()
        self._sr = sr
        self._fft_n = fft_n
        self._history = history                    # rows displayed
        self._scroll_history = history * 10        # rows stored (10× deep scrollback)
        self._f_lo = f_lo
        self._f_hi = f_hi
        self._f_lo_orig = f_lo      # full-range limits — used for zoom clamping
        self._f_hi_orig = f_hi      # and double-click reset
        self._nbins = fft_n // 2 + 1
        self._freqs = _np_fft.rfftfreq(fft_n, 1.0 / sr)
        self._buf    = np.zeros((self._scroll_history, self._nbins), dtype=np.float32)
        self._render = np.zeros((self._history, self._nbins), dtype=np.float32)
        self._head        = 0                      # circular write index (newest row)
        self._rows_written = 0                     # actual rows pushed so far
        self._view_offset = 0                      # 0 = live; >0 = rows back in time
        self._rect   = (0.0, float(self._freqs[-1]))  # (f_lo, f_hi) of last push
        self._drag_x: float | None = None         # left-button drag origin (pixels)

        self._auto_levels = False
        self._fixed_levels = (-10.0, 25.0)
        self._level_counter = 0          # recompute percentiles every N frames


        self._img = pg.ImageItem()
        self._img.setAutoDownsample(True)   # skip rescaling sub-pixel columns
        self.addItem(self._img)
        self._img.setImage(self._render.T, autoLevels=False)
        self._img.setRect(pg.QtCore.QRectF(0.0, 0.0, float(self._freqs[-1]), float(history)))
        self._img.setLevels(self._fixed_levels)
        self._img.setColorMap(cmap_fn())

        self.setBackground(BG)
        self.showGrid(x=True, y=False, alpha=0.18)
        # Every window always shows its own x-axis so frequency can be read
        # after independent zoom/pan.  show_freq_axis kept for API compat.
        self.getAxis("bottom").setLabel("Frequency", units="Hz")
        self.getAxis("left").setTicks([])
        self.getAxis("left").setLabel("")
        self.setXRange(f_lo, f_hi, padding=0)
        self.setYRange(0.0, float(history), padding=0)
        vb = self.getViewBox()
        vb.invertY(True)
        vb.setMouseEnabled(x=False, y=False)
        vb.setLimits(xMin=f_lo, xMax=f_hi)

        self._title_item = pg.TextItem(title, color="#00cc66", anchor=(0, 0))
        self._title_item.setPos(f_lo + (f_hi - f_lo) * 0.01, 8)
        self.addItem(self._title_item)

        self.viewport().installEventFilter(self)

    def eventFilter(self, src, ev):
        if src is self.viewport():
            t = ev.type()

            if t == ev.Type.Wheel:
                px, py = ev.pixelDelta().x(), ev.pixelDelta().y()
                if px != 0 or py != 0:
                    h, v = px / 8.0, py / 8.0
                else:
                    h = ev.angleDelta().x() / 120.0
                    v = ev.angleDelta().y() / 120.0

                span = self._f_hi - self._f_lo
                cx   = (self._f_lo + self._f_hi) * 0.5

                if abs(h) > abs(v):
                    # Horizontal scroll → pan  (right = lower frequencies)
                    cx -= h * span * 0.12
                    bw  = span
                else:
                    # Vertical scroll → zoom to cursor position.
                    # Keep the frequency under the pointer fixed so it
                    # doesn't drift away as you zoom in.
                    bw = max(10.0, span * (1.08 ** v))
                    t  = float(np.clip(
                        ev.position().x() / max(1, self.viewport().width()),
                        0.0, 1.0))
                    f_cursor = self._f_lo + t * span
                    cx = f_cursor + (0.5 - t) * bw

                half = bw * 0.5
                lo_lim, hi_lim = self._f_lo_orig, self._f_hi_orig
                cx = float(np.clip(cx, lo_lim + half, hi_lim - half))
                self._f_lo = max(lo_lim, cx - half)
                self._f_hi = min(hi_lim, cx + half)
                self.setXRange(self._f_lo, self._f_hi, padding=0)
                self.zoom_changed.emit(self._f_lo, self._f_hi)
                return True

            elif t == ev.Type.MouseButtonDblClick:
                # Double-click → reset this window to full frequency range
                self._f_lo = self._f_lo_orig
                self._f_hi = self._f_hi_orig
                self.setXRange(self._f_lo, self._f_hi, padding=0)
                self.zoom_changed.emit(self._f_lo, self._f_hi)
                return True

            elif t == ev.Type.MouseButtonPress:
                if ev.button() == Qt.MouseButton.LeftButton:
                    self._drag_x = ev.position().x()
                    self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True

            elif t == ev.Type.MouseMove:
                if (ev.buttons() & Qt.MouseButton.LeftButton
                        and self._drag_x is not None):
                    dx = ev.position().x() - self._drag_x
                    self._drag_x = ev.position().x()
                    span = self._f_hi - self._f_lo
                    # pixels → frequency: dragging right shifts view left
                    freq_delta = -dx / max(1, self.viewport().width()) * span
                    lo_lim, hi_lim = self._f_lo_orig, self._f_hi_orig
                    new_lo = float(np.clip(self._f_lo + freq_delta,
                                          lo_lim, hi_lim - span))
                    self._f_lo = new_lo
                    self._f_hi = new_lo + span
                    self.setXRange(self._f_lo, self._f_hi, padding=0)
                    self.zoom_changed.emit(self._f_lo, self._f_hi)
                    return True

            elif t == ev.Type.MouseButtonRelease:
                if ev.button() == Qt.MouseButton.LeftButton:
                    self._drag_x = None
                    self.viewport().setCursor(Qt.CursorShape.CrossCursor)
                    return True

        return super().eventFilter(src, ev)

    def push(self, spec_db: np.ndarray, f_lo: float = 0.0, f_hi: float = None) -> None:
        """Push one spectrum row.  f_lo/f_hi describe the frequency extent of
        spec_db so the image rect is kept in sync with both full-range (standard
        FFT) and zoomed (ZoomFFT) spectra."""
        if f_hi is None:
            f_hi = float(self._freqs[-1])
        # Circular write into the full scroll buffer — always, even when paused.
        self._head = (self._head - 1 + self._scroll_history) % self._scroll_history
        self._rows_written = min(self._rows_written + 1, self._scroll_history)
        n = min(len(spec_db), self._nbins)
        self._buf[self._head, :n] = spec_db[:n]
        if n < self._nbins:
            self._buf[self._head, n:] = 0.0
        self._rect = (f_lo, f_hi)
        if self.height() < 8 or self._view_offset != 0:
            return   # paused — buffer updated but display frozen
        self._render_at(0)
        self._img.setRect(pg.QtCore.QRectF(f_lo, 0.0, f_hi - f_lo, float(self._history)))
        self._img.setImage(self._render.T, autoLevels=False)

        if self._auto_levels:
            self._level_counter += 1
            if self._level_counter >= 6:
                self._level_counter = 0
                flat  = self._render.ravel()
                valid = flat[flat > -25.0]
                if len(valid) < 50:
                    return
                floor = float(np.percentile(valid, 30.0))
                peak  = float(np.percentile(valid, 99.2))
                span  = max(12.0, min(peak - floor, 28.0))
                self._img.setLevels((floor - 1.5, floor + span))
        else:
            self._img.setLevels(self._fixed_levels)

    def _render_at(self, offset: int) -> None:
        """Fill self._render with self._history rows starting at `offset` rows
        back from the newest entry.  Two copyto calls — no Python allocation."""
        start = (self._head + offset) % self._scroll_history
        tail  = self._scroll_history - start
        if tail >= self._history:
            np.copyto(self._render, self._buf[start:start + self._history])
        else:
            np.copyto(self._render[:tail],  self._buf[start:])
            np.copyto(self._render[tail:],  self._buf[:self._history - tail])

    @property
    def scroll_max(self) -> int:
        """Scrollback range based on rows actually written, not total buffer size."""
        return max(0, min(self._rows_written, self._scroll_history) - self._history)

    def scroll_to(self, offset: int) -> None:
        """Show history at `offset` rows back from newest.  0 resumes live."""
        self._view_offset = max(0, min(offset, self.scroll_max))
        if self.height() < 8:
            return
        self._render_at(self._view_offset)
        f_lo, f_hi = self._rect
        self._img.setRect(pg.QtCore.QRectF(f_lo, 0.0, f_hi - f_lo, float(self._history)))
        self._img.setImage(self._render.T, autoLevels=False)
        if not self._auto_levels:
            self._img.setLevels(self._fixed_levels)

    def set_auto_levels(self, on: bool) -> None:
        self._auto_levels = bool(on)
        self._level_counter = 0
        if not on:
            self._img.setLevels(self._fixed_levels)

    def set_freq_range(self, lo: float, hi: float, make_full: bool = False) -> None:
        """Zoom (or redefine full range) and notify DSP via zoom_changed."""
        lo, hi = float(lo), float(hi)
        if make_full:
            self._f_lo_orig = lo
            self._f_hi_orig = hi
            vb = self.getViewBox()
            vb.setLimits(xMin=lo, xMax=hi)
        self._f_lo = max(self._f_lo_orig, lo)
        self._f_hi = min(self._f_hi_orig, hi)
        self.setXRange(self._f_lo, self._f_hi, padding=0)
        self.zoom_changed.emit(self._f_lo, self._f_hi)

    def set_cmap(self, cmap: pg.ColorMap) -> None:
        self._img.setColorMap(cmap)

    def set_sr(self, sr: int) -> None:
        if sr == self._sr:
            return
        self._sr = sr
        self._freqs = _np_fft.rfftfreq(self._fft_n, 1.0 / sr)
        self._nbins = len(self._freqs)
        self._buf    = np.zeros((self._scroll_history, self._nbins), dtype=np.float32)
        self._render = np.zeros((self._history, self._nbins), dtype=np.float32)
        self._head        = 0
        self._rows_written = 0
        self._view_offset = 0
        self._rect   = (0.0, float(self._freqs[-1]))
        self._img.setImage(self._render.T, autoLevels=False)
        self._img.setRect(pg.QtCore.QRectF(0.0, 0.0, float(self._freqs[-1]), float(self._history)))
        if not self._auto_levels:
            self._img.setLevels(self._fixed_levels)
        # Reset zoom to full range when SR changes
        self._f_lo = self._f_lo_orig
        self._f_hi = self._f_hi_orig
        self.setXRange(self._f_lo, self._f_hi, padding=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  BAND RETICULE
# ═══════════════════════════════════════════════════════════════════════════════
class BandReticule(pg.PlotWidget):
    band_changed = pyqtSignal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setBackground(BG)
        self.setFixedHeight(52)
        self.showGrid(x=True, y=False, alpha=0.25)
        self.getAxis("bottom").setLabel("Bandpass  ·  drag / scroll  (full 0 – 24 kHz)", units="Hz")
        self.getAxis("left").setTicks([])
        self.getAxis("left").setStyle(showValues=False)
        self.setXRange(BP_LO, BP_HI, padding=0)
        self.setYRange(0, 1, padding=0)
        vb = self.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.setLimits(xMin=BP_LO, xMax=BP_HI)

        self._rgn = pg.LinearRegionItem(
            values=[300.0, 3000.0], orientation="vertical",
            brush=pg.mkBrush(0, 180, 80, 70),
            pen=pg.mkPen("#00ff88", width=2.0),
            movable=True, swapMode="push",
        )
        self.addItem(self._rgn)
        self._rgn.sigRegionChanged.connect(self._on_rgn)
        self._ctr = pg.InfiniteLine(
            pos=1650.0, angle=90,
            pen=pg.mkPen("#00ff88", width=1.0, style=Qt.PenStyle.DotLine),
        )
        self.addItem(self._ctr)
        self.viewport().installEventFilter(self)

    def eventFilter(self, src, ev):
        if src is self.viewport() and ev.type() == ev.Type.Wheel:
            lo, hi = self._rgn.getRegion()
            cx, bw = (lo + hi) * 0.5, hi - lo
            px, py = ev.pixelDelta().x(), ev.pixelDelta().y()
            if px != 0 or py != 0:
                h, v = px / 8.0, py / 8.0
            else:
                h = ev.angleDelta().x() / 120.0
                v = ev.angleDelta().y() / 120.0
            if abs(h) > abs(v):
                cx -= h * (BP_HI - BP_LO) * 0.008   # horizontal → pan filter centre
            else:
                # Vertical scroll → resize, anchored on the frequency under
                # the mouse pointer so that point doesn't drift as the band
                # grows/shrinks (same convention as the waterfall zoom).
                new_bw = max(20.0, bw * (1.08 ** v))
                t = float(np.clip(
                    ev.position().x() / max(1, self.viewport().width()),
                    0.0, 1.0))
                f_cursor = BP_LO + t * (BP_HI - BP_LO)
                frac = (f_cursor - lo) / bw if bw > 0 else 0.5
                cx = f_cursor + (0.5 - frac) * new_bw
                bw = new_bw
            half = bw * 0.5
            cx = float(np.clip(cx, BP_LO + half, BP_HI - half))
            self.set_region(max(BP_LO, cx - half), min(BP_HI, cx + half))
            return True
        return super().eventFilter(src, ev)

    def _on_rgn(self) -> None:
        lo, hi = self._rgn.getRegion()
        self._ctr.setPos((lo + hi) * 0.5)
        self.band_changed.emit(float(lo), float(hi))

    def set_region(self, lo: float, hi: float) -> None:
        self._rgn.blockSignals(True)
        self._rgn.setRegion([lo, hi])
        self._rgn.blockSignals(False)
        self._ctr.setPos((lo + hi) * 0.5)
        self.band_changed.emit(float(lo), float(hi))

    def nudge(self, d_cx: float, bw_scale: float) -> None:
        lo, hi = self._rgn.getRegion()
        cx, bw = (lo + hi) * 0.5, hi - lo
        cx += d_cx
        bw = max(20.0, bw * bw_scale)
        half = bw * 0.5
        cx = float(np.clip(cx, BP_LO + half, BP_HI - half))
        self.set_region(max(BP_LO, cx - half), min(BP_HI, cx + half))


# ═══════════════════════════════════════════════════════════════════════════════
#  STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════
QSS = """
QMainWindow, QWidget {
    background: #060d06; color: #00cc66;
    font-family: Menlo, Monaco, 'Courier New'; font-size: 11px;
}
QGroupBox {
    border: 1px solid #004d22; border-radius: 3px; margin-top: 8px;
    padding-top: 5px; font-weight: bold; font-size: 10px;
    letter-spacing: 1px; color: #008833;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; }
QPushButton {
    background: #091a09; color: #00cc66; border: 1px solid #004d22;
    border-radius: 2px; padding: 4px 14px; font-weight: bold;
}
QPushButton:hover   { background: #102210; border-color: #009944; }
QPushButton:pressed { background: #040d04; }
QPushButton:checked { background: #002e11; border-color: #00ff88; color: #00ff88; }
QPushButton:disabled{ color: #264426; border-color: #172d17; }
QComboBox, QDoubleSpinBox {
    background: #091a09; color: #00cc66; border: 1px solid #004d22;
    border-radius: 2px; padding: 3px 6px; min-width: 90px;
}
QComboBox QAbstractItemView {
    background: #091a09; color: #00cc66; selection-background-color: #002211;
}
QCheckBox { color: #00cc66; spacing: 6px; }
QCheckBox::indicator {
    width: 12px; height: 12px; border: 1px solid #004d22; background: #091a09;
}
QCheckBox::indicator:checked { background: #00aa44; border-color: #00ff88; }
QLabel { color: #007a33; }
QStatusBar { background: #020702; color: #005e22; font-size: 10px; }
QSplitter::handle { background: #003311; height: 3px; }
QScrollBar:vertical {
    background: #050c05; width: 14px;
    border: 1px solid #003311; border-radius: 2px;
}
QScrollBar::handle:vertical {
    background: rgba(0, 160, 70, 140);
    border: 1px solid #00ff88; border-radius: 2px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover  { background: rgba(0, 200, 90, 190); }
QScrollBar::handle:vertical:pressed { background: rgba(0, 255, 136, 200); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
"""


def _fmt_khz(hz: float) -> str:
    """Compact sample-rate label for the SOURCE INFO strip — 48000 → "48kHz",
    44100 → "44.1kHz" — instead of the wider "48,000 Hz" form."""
    khz = hz / 1000.0
    s = f"{khz:.1f}".rstrip("0").rstrip(".")
    return f"{s}kHz"


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class SonarStation(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("◈  SONAR STATION  ·  LOFAR / DEMON  ◈")
        self.setMinimumSize(1280, 720)
        self.resize(1400, 900)
        self.setStyleSheet(QSS)

        self._sr = AUDIO_SR
        self._wav = None
        self._playing = False
        self._in_st = None
        self._feed_thr = None

        self._dsp = DSPWorker()
        self._dthr = QThread()
        self._dsp.moveToThread(self._dthr)
        self._dthr.started.connect(self._dsp.run)
        self._dsp.spectrum_ready.connect(self._recv_spec)
        self._dthr.start()

        self._ao = AudioOut()
        self._dsp.aout = self._ao

        self._pending_lofar: np.ndarray | None = None
        self._pending_demon: np.ndarray | None = None

        self._build_ui()

        self._timer = QTimer()
        self._timer.setInterval(10)   # snappier waterfall refresh
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        vb = QVBoxLayout(root)
        vb.setContentsMargins(8, 6, 8, 4)
        vb.setSpacing(2)
        vb.addWidget(self._make_toolbar())

        self._split = QSplitter(Qt.Orientation.Vertical)
        self._split.setChildrenCollapsible(False)

        # LOFAR waterfall initialised at lofar_sr so _freqs and the initial
        # image rect are correct from the first frame.
        self._wf_lofar = Waterfall(
            title="LOFAR  ·  machinery tonals",
            sr=LOFAR_SR, fft_n=FFT_N, history=HISTORY,
            f_lo=DISP_LO, f_hi=DISP_HI, cmap_fn=cmap_ice,
            show_freq_axis=True,
        )
        # Scroll on the LOFAR waterfall now zooms its own freq axis.
        # Bandpass reticule (bottom strip) remains the control for the filter band.
        self._split.addWidget(self._wf_lofar)

        self._wf_demon = Waterfall(
            title="DEMON  ·  shaft / blade rate",
            sr=int(DEMON_DS_SR), fft_n=DEMON_FFT_N, history=DEMON_HISTORY,
            f_lo=0.0, f_hi=DEMON_DISP_HI, cmap_fn=cmap_ice,
            show_freq_axis=True,
        )  # sr=500 matches _demon_ds_sr
        self._split.addWidget(self._wf_demon)
        self._split.setSizes([500, 280])

        # Wire zoom signals → DSP ZoomFFT setters so resolution adapts live
        self._wf_lofar.zoom_changed.connect(self._dsp.set_lofar_zoom)
        self._wf_demon.zoom_changed.connect(self._dsp.set_demon_zoom)

        # History scrollbar — sits to the right of both waterfalls.
        # Value 0 = live (newest); dragging down = further back in time.
        self._scroll_bar = QScrollBar(Qt.Orientation.Vertical)
        self._scroll_bar.setMinimum(0)
        self._scroll_bar.setMaximum(self._wf_lofar.scroll_max)
        self._scroll_bar.setValue(0)
        self._scroll_bar.setSingleStep(1)
        self._scroll_bar.setPageStep(HISTORY)
        self._scroll_bar.setFixedWidth(14)
        self._scroll_bar.setToolTip("Scroll history  ·  top = live")
        self._scroll_bar.valueChanged.connect(self._on_history_scroll)

        wf_row = QWidget()
        wf_hl  = QHBoxLayout(wf_row)
        wf_hl.setContentsMargins(0, 0, 0, 0)
        wf_hl.setSpacing(2)
        wf_hl.addWidget(self._split)
        wf_hl.addWidget(self._scroll_bar)
        vb.addWidget(wf_row)

        self._reticule = BandReticule()
        self._reticule.band_changed.connect(self._on_band)
        vb.addWidget(self._reticule)

        self._sb = QStatusBar()
        self.setStatusBar(self._sb)
        self._status("SYSTEM READY  —  load .wav or enable mic  ·  scroll waterfall = zoom  ·  dbl-click = reset")

        self._reticule.set_region(BP_LO, BP_HI)
        self._cmap_box.setCurrentText("Ice")
        self._norm_box.setCurrentIndex(3)
        self._chk_ale.setChecked(True)
        self._chk_auto.setChecked(True)
        self._chk_sub.setChecked(True)
        self._chk_eigen.setChecked(True)

    def _make_toolbar(self) -> QWidget:
        bar = QWidget()
        hb = QHBoxLayout(bar)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(8)

        g = QGroupBox("SOURCE")
        gl = QHBoxLayout(g)
        self._btn_wav = QPushButton("LOAD WAV")
        self._btn_wav.clicked.connect(self._load_wav)
        self._btn_play = QPushButton("▶  PLAY")
        self._btn_play.clicked.connect(self._play_stop_toggle)
        self._btn_mic = QPushButton("MIC")
        self._btn_mic.setCheckable(True)
        self._btn_mic.toggled.connect(self._on_mic_toggled)
        self._devbox = QComboBox()
        self._refresh_devs()
        gl.addWidget(self._btn_wav)
        gl.addWidget(self._btn_play)
        gl.addWidget(self._btn_mic)
        gl.addWidget(self._devbox)
        hb.addWidget(g)

        g3 = QGroupBox("SOURCE INFO")
        g3l = QHBoxLayout(g3)
        self._l_src = QLabel("—")
        self._l_src.setStyleSheet("color:#00ff88; font-size:12px;")
        g3l.addWidget(self._l_src)
        hb.addWidget(g3)

        g4 = QGroupBox("PROCESS")
        g4l = QHBoxLayout(g4)
        self._norm_box = QComboBox()
        self._norm_box.addItem("NORM: Off", "off")
        self._norm_box.addItem("NORM: TPSW", "tpsw")
        self._norm_box.addItem("NORM: Robust", "robust")
        self._norm_box.addItem("NORM: OS-CFAR", "oscfar")
        self._norm_box.setToolTip(
            "Spectrum normalization\n"
            "• Off      – median-subtracted raw dB\n"
            "• TPSW     – classic two-pass split-window (retuned)\n"
            "• Robust   – order-statistic / percentile floor (fast)\n"
            "• OS-CFAR  – full ordered-statistic CFAR with guard cells\n"
            "             (best isolation of strong tonals / blade rates)"
        )
        self._norm_box.currentIndexChanged.connect(self._on_norm)
        self._chk_ale = QCheckBox("ALE")
        self._chk_ale.toggled.connect(self._on_ale)
        self._chk_auto = QCheckBox("AUTO LVL")
        self._chk_auto.setToolTip(
            "Adaptive levels relative to estimated noise floor.\n"
            "Keeps tonals in the yellow/red range without washing out\n"
            "the display when normalization or ALE is active."
        )
        self._chk_auto.toggled.connect(self._on_auto_levels)
        self._chk_sub = QCheckBox("SUB")
        self._chk_sub.setToolTip(
            "SUB — DEMON multi-sub-band coherent averaging\n"
            "Splits the bandpass (reticule) region into 8 independent\n"
            "sub-bands, demodulates each one separately, and averages their\n"
            "power spectra before the DEMON FFT (√ of the mean power).\n"
            "Propeller / shaft modulation is coherent across sub-bands\n"
            "while noise is not, giving a √8 ≈ 2.8× amplitude SNR gain."
        )
        self._chk_sub.toggled.connect(self._on_demon_sub)
        self._chk_eigen = QCheckBox("EIGEN")
        self._chk_eigen.setToolTip(
            "Cross-frame PCA/SVD subspace denoiser (LOFAR + DEMON)\n"
            "Keeps a short rolling history of floor-normalized spectra and\n"
            "rebuilds each new one from only its top few eigen-spectra.\n"
            "A persistent tonal sits in the same bin frame after frame, so\n"
            "it dominates the leading singular vectors; noise that's\n"
            "incoherent frame-to-frame is spread across the rest and is\n"
            "dropped. Sharpens strong, stable lines — a very weak or\n"
            "short-lived tonal can get smoothed away with the noise, so\n"
            "toggle off if you need every faint transient preserved.\n"
            "No effect when NORM is Off."
        )
        self._chk_eigen.toggled.connect(self._on_eigen)

        g4l.addWidget(self._norm_box)
        g4l.addWidget(self._chk_ale)
        g4l.addWidget(self._chk_auto)
        g4l.addWidget(self._chk_sub)
        g4l.addWidget(self._chk_eigen)
        hb.addWidget(g4)

        g5 = QGroupBox("AUDIO OUT")
        g5l = QHBoxLayout(g5)
        self._btn_out = QPushButton("🔊 ON")
        self._btn_out.setCheckable(True)
        self._btn_out.setChecked(True)
        self._btn_out.clicked.connect(self._toggle_out)
        g5l.addWidget(self._btn_out)
        hb.addWidget(g5)

        g6 = QGroupBox("DISPLAY")
        g6l = QHBoxLayout(g6)
        self._cmap_box = QComboBox()
        for name in CMAPS:
            self._cmap_box.addItem(name)
        self._cmap_box.setToolTip(
            "Colour map (tactical first)\n"
            "• Green Phosphor / Night Vision – classic detection\n"
            "• Amber / Hot / Crimson         – warm high-contrast\n"
            "• Ice / Bone / Copper / Gray    – cool / neutral\n"
            "• Jet (analysis)               – rainbow"
        )
        self._cmap_box.currentTextChanged.connect(self._on_cmap)
        g6l.addWidget(self._cmap_box)
        hb.addWidget(g6)

        g7 = QGroupBox("PRESET")
        g7l = QHBoxLayout(g7)
        self._preset_box = QComboBox()
        self._preset_box.addItem("Drone 0–500 Hz", "drone500")
        self._preset_box.addItem("Drone 0–1 kHz", "drone1000")
        self._preset_box.addItem("Ship 0–200 Hz", "ship200")
        self._preset_box.addItem("LOFAR 0–1 kHz", "lofar1k")
        self._preset_box.addItem("LOFAR 0–2 kHz", "lofar2k")
        self._preset_box.addItem("LOFAR 0–4 kHz", "lofar4k")
        self._preset_box.addItem("LOFAR 0–8 kHz", "lofar8k")
        self._preset_box.addItem("Full (both)", "full")
        self._preset_box.setCurrentIndex(0)
        self._preset_box.setToolTip(
            "Quick frequency-scale presets\n"
            "• Drone 0–500 Hz / 0–1 kHz – blade-rate region for small UAVs\n"
            "• Ship 0–200 Hz            – classic propeller DEMON\n"
            "• LOFAR 0–1/2/4/8 kHz      – machinery tonal bands\n"
            "• Full                     – reset both displays to maximum"
        )
        self._preset_box.currentIndexChanged.connect(self._on_preset)
        g7l.addWidget(self._preset_box)
        hb.addWidget(g7)

        hb.addStretch()

        btn_snap = QPushButton("📷 SNAP")
        btn_snap.setToolTip("Save screenshot to current directory")
        btn_snap.clicked.connect(self._screenshot)
        hb.addWidget(btn_snap)

        return bar

    def _recv_spec(self, lofar_pkg, demon_pkg) -> None:
        self._pending_lofar = lofar_pkg   # (array, f_lo, f_hi)
        self._pending_demon = demon_pkg   # (array, f_lo, f_hi) | None

    def _tick(self) -> None:
        # Detect a dead input stream — finished_callback is unreliable on
        # USB disconnect with AUHAL; polling .active is the safe fallback.
        if self._playing and self._in_st is not None:
            try:
                alive = self._in_st.active
            except Exception:
                alive = False
            if not alive:
                self._on_device_lost()
                return

        if self._pending_lofar is not None:
            arr, f_lo, f_hi = self._pending_lofar
            self._wf_lofar.push(arr, f_lo, f_hi)
            self._pending_lofar = None
            new_max = self._wf_lofar.scroll_max
            if new_max != self._scroll_bar.maximum():
                self._scroll_bar.setMaximum(new_max)
        if self._pending_demon is not None:
            arr, f_lo, f_hi = self._pending_demon
            self._wf_demon.push(arr, f_lo, f_hi)
            self._pending_demon = None

    def _screenshot(self) -> None:
        path = time.strftime("sonar_station_%Y-%m-%d_%H-%M-%S.png")
        # grab() misses OpenGL framebuffers — grabWindow() captures the
        # composited OS-level pixels, including the waterfall displays.
        # grabWindow() must run on the GUI thread; pixmap.toImage() is fast.
        screen = self.screen() or QApplication.primaryScreen()
        pixmap = screen.grabWindow(int(self.winId()))
        # Convert to QImage here (GUI thread, fast), then PNG-compress and
        # write to disk in a daemon thread so the main-thread event loop —
        # and therefore the _tick() timer, waterfall updates, and DSP signal
        # delivery — are never stalled by the slow compression step.
        image = pixmap.toImage()
        def _save(img: object, p: str) -> None:
            img.save(p)
        threading.Thread(target=_save, args=(image, path), daemon=True).start()
        self._status(f"SCREENSHOT  →  {path}")

    def _on_history_scroll(self, value: int) -> None:
        self._wf_lofar.scroll_to(value)
        self._wf_demon.scroll_to(value)
        if value == 0:
            self._status("LIVE")
        else:
            self._status(f"HISTORY  —  {value} rows back  ·  scroll to top to resume live")

    def _on_band(self, lo: float, hi: float) -> None:
        self._dsp.set_band(lo, hi)

    def _on_norm(self, _idx: int = 0) -> None:
        mode = self._norm_box.currentData()
        self._dsp.set_norm(mode)
        labels = {
            "off":    "OFF — median-subtracted raw dB",
            "tpsw":   "TPSW — classic two-pass (retuned)",
            "robust": "ROBUST — order-statistic floor",
            "oscfar": "OS-CFAR — ordered-statistic CFAR + guards",
        }
        self._status(f"NORM {labels.get(mode, mode)}")

    def _on_cmap(self, name: str) -> None:
        fn = CMAPS.get(name)
        if fn is None:
            return
        cmap = fn()
        self._wf_lofar.set_cmap(cmap)
        self._wf_demon.set_cmap(cmap)
        self._status(f"DISPLAY  {name}")

    def _on_preset(self, _idx: int = 0) -> None:
        key = self._preset_box.currentData()
        # LOFAR full span is always 0 … DISP_HI (8 kHz)
        # DEMON full span defaults to 0 … DEMON_DISP_HI (500 Hz)
        if key == "drone500":
            self._wf_lofar.set_freq_range(0.0, DISP_HI)
            self._wf_demon.set_freq_range(0.0, 500.0, make_full=True)
            self._status("PRESET  Drone 0–500 Hz (DEMON) + LOFAR full")
        elif key == "drone1000":
            self._wf_lofar.set_freq_range(0.0, DISP_HI)
            self._wf_demon.set_freq_range(0.0, 1000.0, make_full=True)
            self._status("PRESET  Drone 0–1 kHz (DEMON) + LOFAR full")
        elif key == "ship200":
            self._wf_lofar.set_freq_range(0.0, DISP_HI)
            self._wf_demon.set_freq_range(0.0, 200.0, make_full=True)
            self._status("PRESET  Ship 0–200 Hz (classic DEMON)")
        elif key == "lofar1k":
            self._wf_lofar.set_freq_range(0.0, 1000.0)
            self._status("PRESET  LOFAR 0–1 kHz")
        elif key == "lofar2k":
            self._wf_lofar.set_freq_range(0.0, 2000.0)
            self._status("PRESET  LOFAR 0–2 kHz")
        elif key == "lofar4k":
            self._wf_lofar.set_freq_range(0.0, 4000.0)
            self._status("PRESET  LOFAR 0–4 kHz")
        elif key == "lofar8k":
            self._wf_lofar.set_freq_range(0.0, DISP_HI)
            self._status("PRESET  LOFAR 0–8 kHz")
        elif key == "full":
            self._wf_lofar.set_freq_range(0.0, DISP_HI, make_full=True)
            self._wf_demon.set_freq_range(0.0, 1000.0, make_full=True)
            self._status("PRESET  Full range both displays")

    def _on_ale(self, on: bool) -> None:
        self._dsp.set_ale(on)
        self._status(f"ALE {'ON — adaptive line enhance' if on else 'OFF'}")

    def _on_auto_levels(self, on: bool) -> None:
        self._wf_lofar.set_auto_levels(on)
        self._wf_demon.set_auto_levels(on)
        self._status(
            f"AUTO LVL {'ON — noise-floor relative levels' if on else 'OFF — fixed levels (-10 … 25 dB)'}"
        )

    def _on_demon_sub(self, on: bool) -> None:
        self._dsp.set_demon_sub(on)
        self._status(
            "DEMON SUB  ON — broadband noise suppression active"
            if on else
            "DEMON SUB  OFF"
        )

    def _on_eigen(self, on: bool) -> None:
        self._dsp.set_eigen(on)
        self._status(
            "EIGEN  ON — cross-frame PCA/SVD subspace denoising (LOFAR + DEMON)"
            if on else
            "EIGEN  OFF"
        )

    def _toggle_out(self, on: bool) -> None:
        self._ao.enabled = on
        self._btn_out.setText("🔊 ON" if on else "🔇 OFF")

    def _load_wav(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open WAV File", "", "Audio (*.wav *.WAV)"
        )
        if not path:
            return
        try:
            sr, raw = wavfile.read(path)
            if raw.dtype == np.int16:
                data = raw.astype(np.float32) / 32768.0
            elif raw.dtype == np.int32:
                data = raw.astype(np.float32) / 2147483648.0
            elif raw.dtype == np.uint8:
                data = (raw.astype(np.float32) - 128.0) / 128.0
            else:
                data = raw.astype(np.float32)
            self._wav = data
            self._sr = sr
            self._dsp.set_sr(sr)
            # Waterfall tracks the post-decimate LOFAR rate, not the raw WAV rate
            self._wf_lofar.set_sr(int(self._dsp.lofar_sr))
            # DEMON axis stays referenced to decimated envelope rate (~500 Hz)
            dur = len(data) / sr
            n_ch = data.shape[1] if data.ndim > 1 else 1
            chs = f"{n_ch}ch"
            self._l_src.setText(
                f"WAV  ·  {chs}  ·  {_fmt_khz(sr)}  ·  {dur:.1f} s"
            )
            self._status(
                f"LOADED  {Path(path).name}  ·  {sr} Hz  ·  {dur:.1f} s  ·  {chs}"
                f"  ·  LOFAR analysis at {int(self._dsp.lofar_sr)} Hz"
            )
        except Exception as exc:
            self._status(f"WAV error: {exc}")

    def _play_stop_toggle(self) -> None:
        if self._playing:
            self._stop()
        else:
            self._play()

    def _play(self) -> None:
        if self._playing:
            return
        if self._wav is not None:
            self._playing = True
            self._btn_play.setText("■  STOP")
            self._wav_start()
        else:
            self._status("No source — load a .wav")

    def _on_mic_toggled(self, checked: bool) -> None:
        if checked:
            # Stop any WAV playback first without unchecking the MIC button
            if self._playing:
                self._playing = False
                if self._in_st is not None:
                    try:
                        self._in_st.stop()
                        self._in_st.close()
                    except Exception:
                        pass
                    self._in_st = None
                self._ao.stop()
            self._playing = True
            self._mic_start()
        else:
            self._stop()

    def _stop(self) -> None:
        self._playing = False
        if self._in_st is not None:
            try:
                self._in_st.stop()
                self._in_st.close()
            except Exception:
                pass
            self._in_st = None
        self._ao.stop()
        self._btn_mic.blockSignals(True)
        self._btn_mic.setChecked(False)
        self._btn_mic.blockSignals(False)
        self._btn_play.setText("▶  PLAY")
        self._scroll_bar.blockSignals(True)
        self._scroll_bar.setValue(0)
        self._scroll_bar.blockSignals(False)
        self._l_src.setText("—")
        self._status("STOPPED")

    def _mic_start(self) -> None:
        # Always capture mic at AUDIO_SR regardless of any previously loaded WAV.
        if self._dsp.sr != AUDIO_SR:
            self._dsp.set_sr(AUDIO_SR)
            self._wf_lofar.set_sr(LOFAR_SR)
        self._sr = AUDIO_SR

        dev = self._devbox.currentData()
        q = self._dsp.audio_q

        # Query the device name and channel count from the OS driver.
        # max_input_channels is what the driver exposes — it may include
        # loopback or monitor channels beyond the physical inputs, but we
        # open all of them and let the DSP worker downmix to mono for analysis.
        try:
            dev_info = sd.query_devices(dev)
            dev_name = dev_info["name"]
            n_ch = max(1, int(dev_info["max_input_channels"]))
        except Exception:
            dev_name = f"device {dev}"
            n_ch = 1

        try:
            self._ao.start(self._sr, n_ch)
        except Exception as exc:
            self._status(f"Audio out error: {exc}  —  device disconnected?")
            self._playing = False
            self._refresh_devs()
            return

        self._l_src.setText(
            f"MIC  ·  {dev_name}  ·  {_fmt_khz(self._sr)}"
            f"  ·  LOFAR {_fmt_khz(self._dsp.lofar_sr)}"
        )

        def _cb(indata, frames, _t, _st):
            if not self._playing:
                raise sd.CallbackStop()
            try:
                q.put_nowait(indata.copy())
            except queue.Full:
                pass

        def _on_finished():
            # Called by PortAudio when the stream stops — expected on _stop(),
            # but also fires on device disconnection.  Schedule cleanup on the
            # GUI thread so Qt objects are touched safely.
            if self._playing:
                QTimer.singleShot(0, self._on_device_lost)

        try:
            self._in_st = sd.InputStream(
                device=dev, samplerate=self._sr, channels=n_ch, dtype="float32",
                blocksize=CHUNK, callback=_cb, finished_callback=_on_finished,
                latency="low",
            )
            self._in_st.start()
            self._status(
                f"MIC LIVE  ·  {dev_name}  ·  {self._sr:,} Hz  ·  {n_ch}ch"
                f"  ·  analysis: mono downmix  ·  LOFAR {int(self._dsp.lofar_sr):,} Hz"
            )
        except Exception as exc:
            self._status(f"Mic error: {exc}  —  device disconnected?")
            self._playing = False
            self._ao.stop()
            self._refresh_devs()

    def _on_device_lost(self) -> None:
        """Called on the GUI thread when the input stream stops unexpectedly."""
        self._stop()
        # PortAudio holds a stale device ID after a USB disconnect — terminate
        # and reinitialise so it re-enumerates Core Audio devices before the
        # next stream open attempt.  Must happen after _stop() closes all streams.
        try:
            sd._terminate()
            time.sleep(0.3)   # let Core Audio finish invalidating the old session
            sd._initialize()
        except Exception:
            pass
        self._refresh_devs()
        self._status("DEVICE DISCONNECTED  —  restart the application to recover")

    def _wav_start(self) -> None:
        data, sr, q = self._wav, self._sr, self._dsp.audio_q
        n_wav_ch = data.shape[1] if data.ndim > 1 else 1
        self._ao.start(sr, n_wav_ch)
        chs = f"{n_wav_ch}ch"
        self._l_src.setText(f"WAV  ·  {chs}  ·  {_fmt_khz(sr)}  ·  {len(data)/sr:.1f} s")
        dt = CHUNK / sr

        def _feed():
            pos = 0
            next_t = time.monotonic()
            while self._playing:
                end = min(pos + CHUNK, len(data))
                piece = data[pos:end]
                # no downmix — raw stereo reaches the queue;
                # DSPWorker handles the mono downmix for analysis
                piece = piece.astype(np.float32)
                if len(piece) > 0:
                    try:
                        q.put(piece, timeout=0.25)
                    except queue.Full:
                        pass
                pos = end if end < len(data) else 0
                next_t += dt
                sleep = next_t - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.monotonic()

        self._feed_thr = threading.Thread(target=_feed, daemon=True)
        self._feed_thr.start()
        self._status("▶ WAV PLAYING (looping)")

    def _refresh_devs(self) -> None:
        self._devbox.clear()
        try:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    self._devbox.addItem(f"{i}: {d['name'][:38]}", i)
        except Exception:
            pass

    def _status(self, msg: str) -> None:
        self._sb.showMessage(f"  {msg}")

    def closeEvent(self, ev) -> None:
        self._stop()
        self._dsp.stop()
        self._dsp.audio_q.put(None)
        self._dthr.quit()
        self._dthr.wait(3000)
        ev.accept()


def main() -> None:
    try:
        pg.setConfigOptions(antialias=False, useOpenGL=True)
    except Exception:
        pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SonarStation()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
