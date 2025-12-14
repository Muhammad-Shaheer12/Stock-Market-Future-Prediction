import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


def _secret(key: str) -> str | None:
    try:
        # st.secrets raises if no secrets.toml exists
        val = st.secrets.get(key)  # type: ignore[attr-defined]
        return str(val) if val is not None else None
    except Exception:
        return None


def _api_base_url() -> str:
    url = _secret("API_URL") or os.getenv("API_URL", "http://localhost:8001")
    return str(url).rstrip("/")


API_URL = _api_base_url()
TIMEOUT_S = 25


def api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    r = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "—"


st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("Multi-Horizon Stock Predictor")
st.caption("Streamlit frontend (Cloud) + FastAPI backend (Render)")

with st.sidebar:
    st.subheader("Settings")
    st.write("**API URL**")
    st.code(API_URL, language="text")
    symbol = st.text_input("Symbol", value=st.session_state.get("symbol", "AAPL"))
    st.session_state["symbol"] = symbol
    st.divider()
    st.write("If this app errors, check your backend is reachable at `/health`.")


@st.cache_data(ttl=60)
def cached_prices(sym: str) -> dict[str, Any]:
    return api_get("/prices", params={"symbol": sym, "limit": 120})


@st.cache_data(ttl=30)
def cached_details(sym: str) -> dict[str, Any]:
    return api_post("/details", {"symbol": sym})


def show_error(e: Exception):
    st.error(str(e))
    st.info("Tip: confirm `API_URL` is correct and public.")


tab_pred, tab_details = st.tabs(["Predictions", "Further details"])

with tab_pred:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Recent prices")
        try:
            pr = cached_prices(symbol.upper())
            dfp = pd.DataFrame(
                {"date": pr.get("labels", []), "adj_close": pr.get("values", [])}
            )
            if not dfp.empty:
                dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")
                dfp = dfp.dropna().set_index("date")
                st.line_chart(dfp["adj_close"], height=260)
            else:
                st.warning("No price data returned.")
        except Exception as e:
            show_error(e)

    with col_b:
        st.subheader("Predictions")
        if st.button("Get predictions", type="primary"):
            try:
                pred = api_post("/predict", {"symbol": symbol.upper()})
                st.session_state["pred"] = pred
            except Exception as e:
                show_error(e)

        pred = st.session_state.get("pred")
        if pred and isinstance(pred, dict):
            preds = pred.get("predicted_log_returns", {})
            if isinstance(preds, dict) and preds:
                rows = []
                for h, obj in preds.items():
                    if isinstance(obj, dict):
                        rows.append(
                            {
                                "horizon_days": int(h),
                                "predicted_log_return": float(
                                    obj.get("predicted_log_return")
                                ),
                                "predicted_%": float(obj.get("predicted_log_return"))
                                * 100,
                                "winner_model": obj.get("winner_model"),
                            }
                        )
                df = pd.DataFrame(rows).sort_values("horizon_days")
                if not df.empty:
                    c1, c2, c3, c4 = st.columns(4)
                    for i, h in enumerate([1, 3, 7, 30]):
                        row = df[df["horizon_days"] == h]
                        if not row.empty:
                            val = float(row.iloc[0]["predicted_log_return"])
                            win = str(row.iloc[0]["winner_model"] or "")
                            [c1, c2, c3, c4][i].metric(
                                f"H{h}",
                                fmt_pct(val),
                                help=f"winner: {win}" if win else None,
                            )
                    st.dataframe(
                        df[
                            [
                                "horizon_days",
                                "predicted_log_return",
                                "predicted_%",
                                "winner_model",
                            ]
                        ],
                        use_container_width=True,
                    )
            else:
                st.warning("No predictions returned.")

with tab_details:
    st.subheader("Beta, correlation, volatility, drawdown, uncertainty")

    if st.button("Load further details"):
        try:
            details = cached_details(symbol.upper())
            st.session_state["details"] = details
        except Exception as e:
            show_error(e)

    details = st.session_state.get("details")
    if details and isinstance(details, dict):
        beta = details.get("beta_vs_spy")
        dd = details.get("drawdown", {})
        mdd = dd.get("max_drawdown")
        level = None
        intervals = details.get("prediction_intervals") or {}
        if isinstance(intervals, dict) and intervals:
            first = next(iter(intervals.values()))
            if isinstance(first, dict):
                level = first.get("level")

        k1, k2, k3 = st.columns(3)
        k1.metric("Beta vs SPY", f"{float(beta):.2f}" if beta is not None else "—")
        k2.metric("Max drawdown", fmt_pct(mdd))
        k3.metric(
            "Interval level",
            f"{int(float(level) * 100)}%" if level is not None else "—",
        )

        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("#### Correlation heatmap")
            corr = details.get("correlation", {})
            symbols = corr.get("symbols", [])
            matrix = corr.get("matrix", [])
            if symbols and matrix:
                dfc = pd.DataFrame(matrix, index=symbols, columns=symbols)
                st.dataframe(
                    dfc.style.background_gradient(cmap="RdYlGn").format("{:.2f}"),
                    use_container_width=True,
                )
            else:
                st.info("Correlation not available.")

        with c2:
            st.markdown("#### Prediction intervals")
            if isinstance(intervals, dict) and intervals:
                rows = []
                for h, obj in intervals.items():
                    if isinstance(obj, dict):
                        rows.append(
                            {
                                "horizon_days": int(h),
                                "pred": float(obj.get("pred")),
                                "lo": float(obj.get("lo")),
                                "hi": float(obj.get("hi")),
                                "winner_model": obj.get("winner_model"),
                            }
                        )
                dfi = pd.DataFrame(rows).sort_values("horizon_days")
                st.dataframe(dfi, use_container_width=True)
                chart_df = dfi.set_index("horizon_days")[["lo", "pred", "hi"]]
                st.line_chart(chart_df, height=260)
            else:
                st.info("Intervals not available.")

        st.markdown("#### Volatility / ATR / Drawdown")
        v = details.get("volatility", {})
        atr = details.get("atr", {})

        df_vol = pd.DataFrame(
            {"date": v.get("dates", []), "vol_annualized": v.get("annualized", [])}
        )
        df_atr = pd.DataFrame(
            {"date": atr.get("dates", []), "atr": atr.get("values", [])}
        )
        df_dd = pd.DataFrame(
            {"date": dd.get("dates", []), "drawdown": dd.get("values", [])}
        )

        for dfx in (df_vol, df_atr, df_dd):
            if not dfx.empty:
                dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")

        c3, c4, c5 = st.columns(3)
        with c3:
            if not df_vol.empty:
                st.line_chart(
                    df_vol.dropna().set_index("date")["vol_annualized"], height=200
                )
            else:
                st.info("No volatility series.")
        with c4:
            if not df_atr.empty:
                st.line_chart(df_atr.dropna().set_index("date")["atr"], height=200)
            else:
                st.info("No ATR series.")
        with c5:
            if not df_dd.empty:
                st.line_chart(df_dd.dropna().set_index("date")["drawdown"], height=200)
            else:
                st.info("No drawdown series.")
