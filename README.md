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
