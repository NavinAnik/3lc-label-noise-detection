# Recording a Demo GIF

This guide walks you through creating `assets/demo.gif` for the repository.

## What to Show

Capture the following flow:

1. **Launch** — Start the Streamlit app and show the main interface
2. **Configure** — Adjust sidebar controls (noise level, confidence threshold, epochs)
3. **Run pipeline** — Click "Run pipeline" and show the loading spinner
4. **Metrics** — Scroll to the Metrics Dashboard showing before/after comparison with deltas
5. **Visualizations** — Browse tabs: Confusion matrix, Confidence distribution, Training curves
6. **Data inspection** — Show incorrect labels and corrected labels sample grids

Aim for 30–60 seconds total.

---

## Tools

### macOS

**Option A: Kap (recommended)**
- Free, open-source: [getkap.co](https://getkap.co/)
- Records region or window, exports directly to GIF
- Install: `brew install kap`

**Option B: Built-in Screen Recording + ffmpeg**
1. `Cmd + Shift + 5` to start screen recording (region or window)
2. Save as `.mov`
3. Convert to GIF with ffmpeg (see below)

**Option C: LICEcap**
- Lightweight: [cockos.com/licecap](https://www.cockos.com/licecap/)
- Records and saves as GIF directly

### Windows

- **ScreenToGif** — [screentogif.com](https://www.screentogif.com/) — Record and export as GIF
- **ShareX** — Includes GIF recording and conversion

### Linux

- **Peek** — `sudo apt install peek` (or equivalent)
- **SimpleScreenRecorder** + ffmpeg for conversion

---

## Converting Video to GIF (ffmpeg)

If you recorded a `.mov` or `.mp4`:

```bash
# Basic conversion (may produce large file)
ffmpeg -i recording.mov -vf "fps=10,scale=960:-1:flags=lanczos" assets/demo.gif

# Optimized (smaller file, uses palette)
ffmpeg -i recording.mov -vf "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" assets/demo.gif
```

**Tips:**
- Use `fps=10` or `fps=8` for smaller GIFs
- `scale=960:-1` keeps width 960px, height auto
- Trim the video first if needed: `-ss 00:00:05 -t 00:00:45` (start at 5s, duration 45s)

---

## Screenshot for README

For `assets/ui_screenshot.png`:

1. Run the Streamlit app
2. Click "Run pipeline" and wait for completion
3. Take a full-window or cropped screenshot:
   - macOS: `Cmd + Shift + 4` (region) or `Cmd + Shift + 5` (options)
   - Save as `assets/ui_screenshot.png`

---

## Checklist

- [ ] Demo GIF shows: run pipeline, metrics update, visualizations
- [ ] GIF saved as `assets/demo.gif`
- [ ] Screenshot saved as `assets/ui_screenshot.png`
- [ ] File sizes reasonable (GIF < 5MB if possible)
