# AI Squash Coach

Streamlit app that analyzes squash footage with Gemini Vision.

## Run
```
pip install -r requirements.txt
export GOOGLE_API_KEY=...   # or put it in .streamlit/secrets.toml, or enter it in the sidebar
streamlit run app.py
```

## Benchmarks
Pro reference breakdowns live in `benchmarks.json` (video file, focus, key points). Add an entry
there and the Pro tab and the comparison selector pick it up automatically. Analysis output is
structured JSON (score, strengths, issues vs. benchmark, drills) rendered as a report.

## History
Every successful analysis is saved to `history.json` (override with `SQUASH_HISTORY_FILE`) and shown in
the **History & Progress** tab with a score trend. Hosted filesystems (e.g. Streamlit Cloud) can reset on
restart, so use the tab's download/restore buttons to back up.

## Skeleton metrics
`pose_metrics.py` runs MediaPipe Pose locally (bundled model in `models/`) and measures knee bend,
torso lean, stance width, racket-hand speed and distance covered (normalised by torso length). The numbers
are 2D and camera-angle dependent, and are passed to Gemini as supporting evidence. When two players overlap
the tracker can merge them - the app shows a checkable skeleton frame and a low-confidence warning. Best
results: one player clearly visible, fixed camera.

## Court calibration & recovery to the T
Tick *Calibrate the court* and click 4+ known floor points (e.g. the four floor corners) on a video frame.
`court.py` builds a pixel->court homography (court 6.4 x 9.75 m, T at 3.2 m / 5.49 m), maps the player's
feet onto the court and measures each trip away from the T: time from the farthest point (beyond 2 m)
until back within 1 m of the T. Pauses longer than 5 s are ignored. Needs a fixed camera; accuracy depends
on how precisely the points are clicked and on the pose tracker following the right player.
