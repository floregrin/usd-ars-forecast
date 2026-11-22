"""USD/ARS minute worker. Install numpy pandas yfinance xgboost. Run with --help.
SQLite timestamps are UTC Unix seconds. Backtesting view contains matured live issues.
Run on a persistent disk. Use one worker process per database.
"""
import argparse
import io
import zipfile
import logging
import sqlite3
import time
from contextlib import closing

import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from fx_model import (DAY, TZ, WINDOW, HORIZONS, fit_model, forecast, load_config, make_pairs)

DDL = """
CREATE TABLE IF NOT EXISTS quotes (
    bar_end INTEGER, resolution INTEGER, close REAL NOT NULL, first_seen INTEGER NOT NULL,
    PRIMARY KEY(bar_end,resolution));
CREATE VIEW IF NOT EXISTS prices AS
    SELECT q.* FROM quotes q WHERE resolution=(
        SELECT MIN(resolution) FROM quotes x WHERE x.bar_end=q.bar_end);
CREATE TABLE IF NOT EXISTS models (version TEXT PRIMARY KEY, trained_at INTEGER, artifact BLOB);
CREATE TABLE IF NOT EXISTS chunks (name TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS forecasts (
    issued_at INTEGER, horizon TEXT, model_version TEXT, issue_minute INTEGER,
    base_quote_at INTEGER, base_price REAL, target_at INTEGER, predicted_price REAL,
    blend_price REAL, actual_at INTEGER, actual_price REAL, status TEXT DEFAULT 'pending',
    PRIMARY KEY(issued_at,horizon,model_version), UNIQUE(issue_minute,horizon,model_version));
CREATE INDEX IF NOT EXISTS pending ON forecasts(status,target_at);
CREATE VIEW IF NOT EXISTS backtesting AS SELECT *,
    abs(actual_price-predicted_price) AS xgb_error, abs(actual_price-base_price) AS baseline_error,
    abs(actual_price-blend_price) AS blend_error FROM forecasts WHERE status='scored';
"""


def now():
    return pd.Timestamp.now(tz="UTC").floor("s")


def fetch(db, start, stop, interval):
    daily = interval == "1d"
    a, b = ((start.tz_convert(TZ).date(), stop.tz_convert(TZ).date()) if daily
            else (start.to_pydatetime(), stop.to_pydatetime()))
    if a >= b:
        return 0
    frame = yf.Ticker("ARS=X").history(start=a, end=b, interval=interval,
                                       auto_adjust=False, raise_errors=True, timeout=15)
    if frame.empty:
        return 0
    close = frame["Close"].dropna()
    seconds = {"1d": 86400, "1h": 3600, "1m": 60}[interval]
    if daily:
        ends = (pd.DatetimeIndex(close.index.date) + DAY).tz_localize(TZ).tz_convert("UTC")
    else:
        ends = close.index.tz_convert("UTC") + pd.Timedelta(seconds=seconds)
    seen = int(now().timestamp())
    rows = [(int(t.timestamp()), seconds, float(v), seen) for t, v in zip(ends, close)
            if start < t <= stop and np.isfinite(v) and v > 0]
    with db:
        db.executemany("INSERT OR IGNORE INTO quotes VALUES (?,?,?,?)", rows)
    return len(rows)


def bootstrap(db, stop):
    # Fixed start dates allow interrupted startup to resume from completed chunks.
    end = stop.floor("h")
    cut = end - pd.Timedelta(days=729)
    for interval, start, finish, step in [
        ("1d", end - pd.DateOffset(years=4), cut, 800),
        ("1h", cut, end, 180), ("1m", end - 28*DAY, end, 7)]:
        while start < finish:
            until = min(start + step*DAY, finish)
            key = f"{interval}/{start}/{until}"
            if not db.execute("SELECT 1 FROM chunks WHERE name=?", (key,)).fetchone():
                count = fetch(db, start, until, interval)
                if not count:
                    raise RuntimeError(f"Empty historical chunk {key}. Retry startup.")
                with db:
                    db.execute("INSERT OR IGNORE INTO chunks VALUES (?)", (key,))
            start = until
    with db:
        db.execute("INSERT OR IGNORE INTO chunks VALUES ('initialized')")


def quote_history(db, stamp):
    # Rank resolutions after filtering for receipt time so a later finer bar
    # cannot mask a coarser quote that was already available.
    rows = db.execute("""
        SELECT bar_end,close FROM (
            SELECT bar_end,close,ROW_NUMBER() OVER (
                PARTITION BY bar_end ORDER BY resolution) AS rank
            FROM quotes WHERE bar_end BETWEEN ? AND ? AND first_seen<=?
        ) WHERE rank=1 ORDER BY bar_end
    """, (int((stamp-pd.DateOffset(years=4)).timestamp()), int(stamp.timestamp()),
          int(stamp.timestamp()))).fetchall()
    return pd.Series([v for _,v in rows], index=pd.to_datetime([t for t,_ in rows],unit="s",utc=True),
                     dtype=float)


def daily_model(db, stamp):
    specs, key = load_config()
    version = f"elapsed-v2/{key}/{stamp.tz_convert(TZ).date()}"
    row = db.execute("SELECT artifact FROM models WHERE version=?", (version,)).fetchone()
    if row is None:
        training = make_pairs(quote_history(db, stamp))
        if training.h.value_counts().reindex(HORIZONS, fill_value=0).min() < 30:
            raise RuntimeError("Insufficient matured examples for at least one horizon")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            for h, spec in specs.items():
                model = fit_model(training, spec["params"])
                archive.writestr(h, bytes(model.get_booster().save_raw(raw_format="ubj")))
        with db:
            db.execute("INSERT OR IGNORE INTO models VALUES (?,?,?)",
                       (version, int(stamp.timestamp()), buf.getvalue()))
        row = db.execute("SELECT artifact FROM models WHERE version=?", (version,)).fetchone()
    models = {}
    with zipfile.ZipFile(io.BytesIO(row[0])) as archive:
        for h in specs:
            model = XGBRegressor()
            model.load_model(bytearray(archive.read(h)))
            models[h] = model, specs[h]["xgb_weight"]
    return version, models


def issue(db, models, version, stamp):
    issued = int(stamp.timestamp())
    quote_at, base = db.execute("SELECT bar_end,close FROM quotes WHERE bar_end<=? "
                                "AND first_seen<=? ORDER BY bar_end DESC,resolution LIMIT 1",
                                (issued, issued)).fetchone()
    rows = []
    for h, offset in HORIZONS.items():
        target = stamp.tz_convert(TZ) + offset
        model, weight = models[h]
        price, blend = forecast(model, base, pd.Timestamp(quote_at,unit="s",tz="UTC"),target,weight)
        rows.append((issued,h,version,issued//60,quote_at,base,int(target.timestamp()),price,blend))
    db.executemany("INSERT OR IGNORE INTO forecasts "
                   "(issued_at,horizon,model_version,issue_minute,base_quote_at,base_price,"
                   "target_at,predicted_price,blend_price) VALUES (?,?,?,?,?,?,?,?,?)", rows)


def score(db, stamp):
    cutoff = int(stamp.timestamp())
    delay = int(WINDOW.total_seconds())
    # Reopen older unmatched forecasts once when expanding the matching window.
    marker = f"score_window_{delay}"
    if db.execute("INSERT OR IGNORE INTO chunks(name) VALUES (?)", (marker,)).rowcount:
        db.execute("UPDATE forecasts SET status='pending' WHERE status='unmatched'")
    # A delayed bar can improve an earlier match or resolve an unmatched target.
    # Revisit affected targets when previously unseen observations enter the archive.
    last = db.execute("SELECT seen FROM score_cursor").fetchone()
    since = last[0] if last else 0
    changed = db.execute("SELECT DISTINCT bar_end FROM quotes WHERE first_seen BETWEEN ? AND ?",
                         (since, cutoff)).fetchall()
    for (end,) in changed:
        db.execute("UPDATE forecasts SET status='pending' WHERE target_at BETWEEN ? AND ? "
                   "AND (actual_at IS NULL OR actual_at>=?)", (end-delay, end, end))
    due = db.execute("SELECT DISTINCT target_at FROM forecasts WHERE status='pending' "
                     "AND target_at<=?", (cutoff,)).fetchall()
    for (target,) in due:
        actual = db.execute("SELECT bar_end,close FROM prices WHERE bar_end BETWEEN ? AND ? "
                            "ORDER BY bar_end LIMIT 1", (target, min(cutoff,target+delay))).fetchone()
        if actual:
            db.execute("UPDATE forecasts SET actual_at=?,actual_price=?,status='scored' "
                       "WHERE target_at=?", (*actual, target))
        elif cutoff >= target+delay:
            db.execute("UPDATE forecasts SET status='unmatched' WHERE target_at=?", (target,))
    db.execute("DELETE FROM score_cursor")
    db.execute("INSERT INTO score_cursor VALUES (?)", (cutoff,))


def cycle(db):
    stamp = now()
    if not db.execute("SELECT 1 FROM chunks WHERE name='initialized'").fetchone():
        bootstrap(db, stamp)
    # Overlap polls to collect delayed bars. Empty weekends can still use the last close.
    for interval, lookback in [("1h", DAY), ("1m", DAY)]:
        try:
            fetch(db, stamp.floor("h")-lookback, stamp, interval)
        except Exception:
            logging.exception("Yahoo %s poll failed; using archived quotes", interval)
    version, model = daily_model(db, now())
    stamp = now()
    with db:
        issue(db, model, version, stamp)
        score(db, stamp)
    result = pd.read_sql_query("SELECT horizon,base_quote_at,base_price,target_at,predicted_price,"
        "blend_price FROM forecasts WHERE issue_minute=? AND model_version=? ORDER BY target_at",
        db, params=(int(stamp.timestamp())//60, version))
    for col in ["base_quote_at", "target_at"]:
        result[col] = pd.to_datetime(result[col], unit="s", utc=True).dt.tz_convert(TZ)
    print(stamp, version, result.to_string(index=False, float_format="{:.2f}".format), sep="\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="usd_ars.sqlite")
    parser.add_argument("--loop", action="store_true", help="Repeat at minute boundaries")
    args = parser.parse_args()
    load_config()
    with closing(sqlite3.connect(args.db, timeout=30)) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(DDL + """
            CREATE TABLE IF NOT EXISTS score_cursor(seen INTEGER);
            CREATE INDEX IF NOT EXISTS target_idx ON forecasts(target_at);
            CREATE INDEX IF NOT EXISTS seen_idx ON quotes(first_seen);
        """)
        while True:
            try:
                cycle(db)
            except Exception:
                if not args.loop:
                    raise
                logging.exception("Cycle failed")
            if not args.loop:
                break
            time.sleep(60 - time.time() % 60)


if __name__ == "__main__":
    main()
