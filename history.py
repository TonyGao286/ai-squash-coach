"""Persistent analysis history stored as a JSON list (newest last)."""
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone

HISTORY_FILE = os.environ.get(
    "SQUASH_HISTORY_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.json"),
)


def load_history(path=HISTORY_FILE):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    return [r for r in data if isinstance(r, dict) and isinstance(r.get("report"), dict)] \
        if isinstance(data, list) else []


def _write(records, path):
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def add_record(report, filename="", note="", benchmarks=(), path=HISTORY_FILE):
    record = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "filename": filename,
        "note": note,
        "benchmarks": list(benchmarks),
        "report": report,
    }
    records = load_history(path)
    records.append(record)
    _write(records, path)
    return record


def delete_record(record_id, path=HISTORY_FILE):
    records = [r for r in load_history(path) if r.get("id") != record_id]
    _write(records, path)


def merge_records(incoming, path=HISTORY_FILE):
    """Merge imported records (by id). Returns the number of new records added."""
    if not isinstance(incoming, list):
        return 0
    records = load_history(path)
    known = {r.get("id") for r in records}
    new = [
        r for r in incoming
        if isinstance(r, dict) and isinstance(r.get("report"), dict)
        and r.get("id") and r["id"] not in known and r.get("timestamp")
    ]
    if new:
        records.extend(new)
        records.sort(key=lambda r: r["timestamp"])
        _write(records, path)
    return len(new)
