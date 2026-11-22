"""Run beside usd_ars.sqlite with streamlit run usd_ars_dashboard.py."""
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

TZ = "America/Argentina/Buenos_Aires"
DB = Path(os.getenv("USD_ARS_DB", "usd_ars.sqlite")).resolve()
HORIZONS = ["1D", "7D", "14D", "1M", "3M", "6M", "1Y"]
MODELS = {"base_price": ("No change", "#596981"), "predicted_price": ("XGBoost", "#bd580e"),
          "blend_price": ("Blend", "#805a9f")}


def local(seconds):
    return pd.to_datetime(seconds, unit="s", utc=True).tz_convert(TZ)


def draw(history, row):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.index, history, color="#10745e", label="Daily close")
    for field, (label, color) in MODELS.items():
        ax.plot([local(row.base_quote_at), local(row.target_at)], [row.base_price, row[field]],
                "o--", color=color, markersize=4, label=label)
    finish(fig, ax)


def finish(fig, ax):
    ax.set_ylabel("ARS per USD")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


SNAPSHOT = Path(__file__).with_name("snapshot")


METRIC_HELP = {
    "selection_n": "Matched forecasts in the earlier period used to select model settings and blend weight.",
    "test_issued": "Historical forecasts reconstructed for the later test period.",
    "test_scored": "Test forecasts with a quote at or within six hours after the target. Each contributes one point per plotted line.",
    "xgb_weight": "Fraction of the XGBoost price move retained by the tuned blend. The remainder goes to no change.",
    "baseline_MAE": "Mean absolute error when carrying the starting price forward.",
    "XGB_MAE": "Mean absolute error of the full XGBoost forecast.",
    "half_MAE": "Mean absolute error of the midpoint between no change and XGBoost.",
    "tuned_MAE": "Mean absolute error of the blend selected for this horizon.",
}


def historical_panel():
    horizon = st.select_slider("Historical backtest horizon", HORIZONS, key="backtest_horizon")
    summary = pd.read_csv(SNAPSHOT/"metrics.csv").set_index("horizon")
    columns = {name: st.column_config.NumberColumn(help=meaning, format="%.2f" if name.endswith("MAE") or name == "xgb_weight" else "%d")
               for name, meaning in METRIC_HELP.items()}
    st.dataframe(summary.loc[[horizon]], column_config=columns)
    st.caption("Errors are in ARS per USD. Lower is better. The chart shows the scored test forecasts.")
    with st.expander("Compare all seven horizons"):
        st.dataframe(summary, column_config=columns)
    with st.expander("What do the columns mean?"):
        st.markdown("**horizon** — Time from forecast issue to target.\n\n" +
                    "\n\n".join(f"**{name}** — {meaning}" for name, meaning in METRIC_HELP.items()))
    g = pd.read_csv(SNAPSHOT/"holdout.csv")
    g = g.loc[g.horizon.eq(horizon)].dropna(subset=["actual"])
    fig, ax = plt.subplots(figsize=(10, 4))
    target = pd.to_datetime(g.due, utc=True).dt.tz_convert(TZ)
    for field, label in [("actual","Matched close"),("predicted","XGBoost"),
                         ("base","No change"),("blend","Tuned blend")]:
        ax.plot(target,g[field],marker=".",label=label)
    ax.set_title(f"{horizon} backtest · {len(g)} scored forecasts")
    ax.set_xlabel("Historical target date")
    finish(fig, ax)


def show_snapshot():
    if not (SNAPSHOT/"forecast.csv").exists():
        st.info("Run the tuning notebook to generate the dashboard snapshot.")
        return
    forecast, monitoring = st.tabs(["Forecast", "Historical backtest"])
    with forecast:
        h = st.select_slider("Prediction horizon", HORIZONS)
        batch = pd.read_csv(SNAPSHOT/"forecast.csv").set_index("horizon")
        row = batch.loc[h].copy()
        for k in ("issued_at","base_quote_at","target_at"):
            row[k] = int(pd.Timestamp(row[k]).timestamp())
        st.caption(f"Historical snapshot as of {local(row.issued_at):%d %b %Y %H:%M} ART · "
                   f"Target {local(row.target_at):%d %b %Y %H:%M} ART · "
                   f"XGBoost weight {row.xgb_weight:.0%}")
        for col, (field, (label, _)) in zip(st.columns(3), MODELS.items()):
            col.metric(label,f"{row[field]:,.2f}")
        hist = pd.read_csv(SNAPSHOT/"history.csv")
        hist.index = pd.to_datetime(hist.time,utc=True).dt.tz_convert(TZ)
        draw(hist.close,row)
    with monitoring:
        historical_panel()


@st.fragment(run_every="60s")
def render():
    if not DB.is_file():
        show_snapshot()
        return
    with closing(sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        batch = pd.read_sql_query("SELECT * FROM forecasts WHERE issued_at="
                                  "(SELECT MAX(issued_at) FROM forecasts) ORDER BY target_at", db)
        if batch.empty:
            st.info("Waiting for the first forecast.")
            return
        horizon = st.select_slider("Prediction horizon", HORIZONS, value="1D")
        row = batch.set_index("horizon").loc[horizon]
        issued = int(row.issued_at)
        forecast, monitor, historical = st.tabs(["Forecast", "Live backtesting", "Historical backtest"])
        with historical:
            if (SNAPSHOT/"metrics.csv").exists():
                historical_panel()
        with forecast:
            st.caption(f"Issued {local(issued):%d %b %Y %H:%M:%S} ART · "
                       f"Base quote {local(row.base_quote_at):%d %b %Y %H:%M} ART · "
                       f"Target {local(row.target_at):%d %b %Y %H:%M:%S} ART")
            for col, (field, (label, _)) in zip(st.columns(3), MODELS.items()):
                col.metric(label, f"{row[field]:,.2f}")
            cutoff = int((local(issued)-pd.DateOffset(years=1)).timestamp())
            history = pd.read_sql_query("SELECT bar_end,close FROM prices WHERE "
                                        "bar_end BETWEEN ? AND ? AND first_seen<=? ORDER BY bar_end",
                                        db, params=(cutoff, int(row.base_quote_at), issued))
            history.index = local(history.bar_end.to_numpy())
            history = history.close.groupby(history.index.normalize()).tail(1)
            draw(history, row)
        with monitor:
            # Group daily fits that share the same deployed configuration.
            family = row.model_version.rsplit("/", 1)[0]+"/%"
            st.caption(f"Forecast family: {family[:-2]}")
            days = st.selectbox("Target-date window (days)", [7, 30, 90, 365], index=1)
            now = int(pd.Timestamp.now(tz="UTC").timestamp())
            start = now - days*86400
            counts = pd.read_sql_query(
                "SELECT horizon, COUNT(*) AS due, SUM(status='scored') AS scored, "
                "SUM(status='pending') AS pending, SUM(status='unmatched') AS unmatched, "
                "COUNT(DISTINCT actual_at) AS distinct_closes, "
                "AVG(ABS(actual_price-base_price)) AS no_change_MAE, "
                "AVG(ABS(actual_price-predicted_price)) AS XGB_MAE, "
                "AVG(ABS(actual_price-blend_price)) AS blend_MAE FROM forecasts "
                "WHERE target_at BETWEEN ? AND ? AND model_version LIKE ? GROUP BY horizon", db, params=(start, now, family)
            ).set_index("horizon").reindex(HORIZONS)
            for name in ["due","scored","pending","unmatched","distinct_closes"]:
                counts[name] = counts[name].fillna(0).astype(int)
            st.dataframe(counts, column_config={c: st.column_config.NumberColumn(format="%.2f")
                                                for c in counts if c.endswith("MAE")})
            rows = pd.read_sql_query("SELECT * FROM backtesting WHERE horizon=? AND target_at "
                                     "BETWEEN ? AND ? AND model_version LIKE ? ORDER BY target_at", db, params=(horizon,start,now,family))
            if rows.empty:
                st.info(f"No scored {horizon} forecasts in this window yet.")
                return
            fig, ax = plt.subplots(figsize=(10, 4))
            targets = local(rows.target_at.to_numpy())
            ax.plot(targets, rows.actual_price, color="#10745e", label="Matched close")
            for field, (label, color) in MODELS.items():
                ax.plot(targets, rows[field], "--", color=color, label=label)
            ax.set_xlabel("Forecast target time (Buenos Aires)")
            finish(fig, ax)


st.set_page_config(page_title="USD/ARS forecasts", layout="wide")
st.title("USD/ARS forecasts")
render()
