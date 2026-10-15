"""Mixue Demand Forecasting — Operations Management Chapter 3.

Methods adapted to the concepts in Chapter 3 (Forecasting):
- Naive forecast
- Simple moving average
- Weighted moving average
- Exponential smoothing
- Trend projection
- Weekday seasonal index / deseasonalization
- Forecast accuracy: MAD, MSE, MAPE

IMPORTANT: Built-in demand values are deterministic illustrative/simulated data,
not official Mixue sales data. Replace them with real case-study data when available.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Mixue Forecasting — Chapter 3",
    page_icon="🍦",
    layout="wide",
)

st.title("🍦 Mixue Ice Cream & Tea")
st.subheader("Demand Forecasting Application — Operations Management Chapter 3")
st.markdown(
    "Aplikasi ini menerapkan metode peramalan kuantitatif dari Chapter 3: "
    "naive, moving averages, exponential smoothing, trend projection, seasonal index, "
    "regresi kausal sederhana, dan pengukuran akurasi forecast. Hasilnya dapat "
    "mendukung keputusan persediaan serta penjadwalan operasional."
)
st.warning(
    "Data bawaan adalah **data simulasi untuk demonstrasi**, bukan data penjualan resmi Mixue. "
    "Edit kolom permintaan pada tabel agar sesuai dengan data studi kasus yang diberikan dosen."
)


# -----------------------------------------------------------------------------
# Diagram alur kerja forecasting — dibuat dengan Plotly agar tidak memerlukan
# dependency diagram tambahan seperti Graphviz.
# -----------------------------------------------------------------------------
st.header("Diagram Alur Analisis Forecasting — Chapter 3")
st.caption(
    "Alur berikut menunjukkan bagaimana data permintaan diolah menjadi forecast, "
    "dievaluasi akurasinya, lalu digunakan untuk membantu keputusan operasional."
)
flow = go.Figure()
flow_nodes = [
    {"x0": 0.03, "x1": 0.27, "y0": 0.66, "y1": 0.88, "title": "1. Data historis", "detail": "Permintaan harian + indikator promo", "color": "#E0F2FE"},
    {"x0": 0.38, "x1": 0.62, "y0": 0.66, "y1": 0.88, "title": "2. Persiapan data", "detail": "Validasi, urut tanggal, pola hari", "color": "#DCFCE7"},
    {"x0": 0.73, "x1": 0.97, "y0": 0.66, "y1": 0.88, "title": "3. Metode forecast", "detail": "Naive • MA • WMA • ES • Trend • Regresi", "color": "#FEF3C7"},
    {"x0": 0.73, "x1": 0.97, "y0": 0.17, "y1": 0.39, "title": "4. Evaluasi akurasi", "detail": "MAD • MSE • MAPE", "color": "#FCE7F3"},
    {"x0": 0.38, "x1": 0.62, "y0": 0.17, "y1": 0.39, "title": "5. Pilih metode", "detail": "Bandingkan kesalahan forecast", "color": "#EDE9FE"},
    {"x0": 0.03, "x1": 0.27, "y0": 0.17, "y1": 0.39, "title": "6. Keputusan operasional", "detail": "Stok • produksi • tenaga kerja", "color": "#E2E8F0"},
]
for node in flow_nodes:
    flow.add_shape(
        type="rect",
        x0=node["x0"], x1=node["x1"], y0=node["y0"], y1=node["y1"],
        xref="x", yref="y",
        line={"color": "#64748B", "width": 1.5},
        fillcolor=node["color"],
        layer="below",
    )
    flow.add_annotation(
        x=(node["x0"] + node["x1"]) / 2,
        y=(node["y0"] + node["y1"]) / 2,
        xref="x", yref="y",
        text=f"<b>{node['title']}</b><br><span style='font-size:11px'>{node['detail']}</span>",
        showarrow=False,
        align="center",
        font={"size": 13, "color": "#0F172A"},
    )
flow_arrows = [
    ((0.27, 0.77), (0.38, 0.77)),
    ((0.62, 0.77), (0.73, 0.77)),
    ((0.85, 0.66), (0.85, 0.40)),
    ((0.73, 0.28), (0.62, 0.28)),
    ((0.38, 0.28), (0.27, 0.28)),
]
for start, end in flow_arrows:
    flow.add_annotation(
        x=end[0], y=end[1], ax=start[0], ay=start[1],
        xref="x", yref="y", axref="x", ayref="y",
        text="", showarrow=True, arrowhead=3, arrowsize=1.15,
        arrowwidth=2, arrowcolor="#64748B",
    )
flow.update_layout(
    height=410,
    margin={"l": 10, "r": 10, "t": 20, "b": 10},
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis={"range": [0, 1], "visible": False, "fixedrange": True},
    yaxis={"range": [0, 1], "visible": False, "fixedrange": True},
    showlegend=False,
)
st.plotly_chart(flow, use_container_width=True, config={"displayModeBar": False})

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
DAY_NAMES = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
METHODS = [
    "Naive",
    "Simple Moving Average",
    "Weighted Moving Average",
    "Exponential Smoothing",
    "Trend Projection",
    "Causal Regression (Trend + Promo)",
]


def create_demo_data(periods: int = 56) -> pd.DataFrame:
    """Create repeatable demo observations; not real company data."""
    end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end_date, periods=periods, freq="D")
    weekday_base = {0: 3200, 1: 3300, 2: 3400, 3: 3500, 4: 4000, 5: 5200, 6: 4900}
    demand = []
    promotions = []
    # A few irregular promotion days for demonstrating the causal model.
    promo_positions = {8, 17, 26, 34, 43, 51}
    for i, date in enumerate(dates):
        promo = 1 if i in promo_positions else 0
        seasonal_base = weekday_base[int(date.dayofweek)]
        trend = 12 * i
        repeatable_variation = int(130 * math.sin(i * 1.17) + 70 * math.cos(i * 0.51))
        promo_lift = 320 if promo else 0
        demand.append(max(0, seasonal_base + trend + repeatable_variation + promo_lift))
        promotions.append(promo)
    return pd.DataFrame({
        "Tanggal": dates,
        "Permintaan Aktual (cup)": demand,
        "Promo (0/1)": promotions,
    })


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns, types, sort order and nonnegative demand."""
    required = ["Tanggal", "Permintaan Aktual (cup)"]
    result = df[required].copy()
    if "Promo (0/1)" in df.columns:
        result["Promo (0/1)"] = pd.to_numeric(df["Promo (0/1)"], errors="coerce").fillna(0).clip(0, 1).round().astype(int)
    else:
        result["Promo (0/1)"] = 0
    result["Tanggal"] = pd.to_datetime(result["Tanggal"], errors="coerce")
    result["Permintaan Aktual (cup)"] = pd.to_numeric(
        result["Permintaan Aktual (cup)"], errors="coerce"
    )
    result = result.dropna(subset=["Tanggal", "Permintaan Aktual (cup)"])
    result["Permintaan Aktual (cup)"] = result["Permintaan Aktual (cup)"].clip(lower=0)
    result = result.sort_values("Tanggal").drop_duplicates(subset=["Tanggal"], keep="last")
    return result.reset_index(drop=True)


def get_weekday_indices(history: pd.DataFrame) -> Dict[int, float]:
    """Compute weekday seasonal indices = weekday average / overall average.

    Weekday index > 1 means demand for that weekday is above the average day;
    index < 1 means below average. This is an illustrative weekday application
    of the seasonal-index concept in Chapter 3.
    """
    values = history["Permintaan Aktual (cup)"].astype(float)
    overall_mean = float(values.mean()) if len(values) else 0.0
    if overall_mean <= 0:
        return {weekday: 1.0 for weekday in range(7)}

    work = history.copy()
    work["weekday"] = work["Tanggal"].dt.dayofweek
    weekday_means = work.groupby("weekday")["Permintaan Aktual (cup)"].mean()
    result: Dict[int, float] = {}
    for weekday in range(7):
        weekday_mean = float(weekday_means.get(weekday, overall_mean))
        index = weekday_mean / overall_mean if overall_mean > 0 else 1.0
        result[weekday] = index if np.isfinite(index) and index > 0 else 1.0
    return result


def forecast_methods(
    history: pd.DataFrame,
    target_date: pd.Timestamp,
    alpha: float,
    moving_window: int = 3,
    target_promo: int = 0,
) -> Tuple[Dict[str, float], Dict[int, float]]:
    """Forecast one target day using historical data only.

    Time-series methods are calculated on deseasonalized demand and then
    reseasonalized for the target weekday. The weights assign more importance
    to the latest observations. Outputs are clipped at zero.
    """
    hist = prepare_history(history)
    if hist.empty:
        return {method: 0.0 for method in METHODS}, {i: 1.0 for i in range(7)}

    alpha = float(np.clip(alpha, 0.01, 1.0))
    actual = hist["Permintaan Aktual (cup)"].astype(float).to_numpy()
    indices = get_weekday_indices(hist)
    weekday_ids = hist["Tanggal"].dt.dayofweek.to_numpy()
    seasonal = np.array([indices[int(day)] for day in weekday_ids], dtype=float)
    seasonal = np.where(seasonal > 0, seasonal, 1.0)
    deseasonalized = actual / seasonal

    n = len(deseasonalized)
    window = max(1, min(int(moving_window), n))
    target_index = indices.get(int(pd.Timestamp(target_date).dayofweek), 1.0)
    if not np.isfinite(target_index) or target_index <= 0:
        target_index = 1.0

    # 1) Naive: next period = last observed deseasonalized value.
    naive_base = float(deseasonalized[-1])

    # 2) Simple moving average of the most recent n observations.
    sma_base = float(np.mean(deseasonalized[-window:]))

    # 3) Weighted moving average. Recent periods receive larger weights.
    recent = deseasonalized[-window:]
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    wma_base = float(np.dot(recent, weights))

    # 4) Single exponential smoothing, initialized with the first observation.
    level = float(deseasonalized[0])
    for observation in deseasonalized[1:]:
        level = alpha * float(observation) + (1.0 - alpha) * level
    es_base = level

    # 5) Linear trend projection on deseasonalized observations.
    x = np.arange(n, dtype=float)
    if n >= 2:
        slope, intercept = np.polyfit(x, deseasonalized, 1)
        trend_base = float(slope * n + intercept)
    else:
        trend_base = float(deseasonalized[-1])

    # 6) Associative/causal model: demand explained by time trend and promotion.
    # Fit on deseasonalized demand to isolate the weekday pattern first.
    promotion = (
        pd.to_numeric(hist["Promo (0/1)"], errors="coerce").fillna(0).clip(0, 1).to_numpy(dtype=float)
        if "Promo (0/1)" in hist.columns else np.zeros(n, dtype=float)
    )
    target_promo = int(np.clip(target_promo, 0, 1))
    design = np.column_stack([np.ones(n, dtype=float), x, promotion])
    try:
        coefficients, *_ = np.linalg.lstsq(design, deseasonalized, rcond=None)
        causal_base = float(coefficients[0] + coefficients[1] * n + coefficients[2] * target_promo)
    except (ValueError, np.linalg.LinAlgError):
        causal_base = trend_base

    base = {
        "Naive": naive_base,
        "Simple Moving Average": sma_base,
        "Weighted Moving Average": wma_base,
        "Exponential Smoothing": es_base,
        "Trend Projection": trend_base,
        "Causal Regression (Trend + Promo)": causal_base,
    }
    forecasts = {
        method: max(0.0, float(value) * target_index)
        for method, value in base.items()
    }
    return forecasts, indices


def accuracy_metrics(actual: Iterable[float], forecast: Iterable[float]) -> Dict[str, float]:
    """Compute MAD, MSE and MAPE; MAPE ignores zero actuals."""
    actual_arr = np.asarray(list(actual), dtype=float)
    forecast_arr = np.asarray(list(forecast), dtype=float)
    valid = np.isfinite(actual_arr) & np.isfinite(forecast_arr)
    actual_arr = actual_arr[valid]
    forecast_arr = forecast_arr[valid]
    if len(actual_arr) == 0:
        return {"MAD": float("nan"), "MSE": float("nan"), "MAPE (%)": float("nan"), "n": 0}

    errors = actual_arr - forecast_arr
    abs_errors = np.abs(errors)
    positive = actual_arr != 0
    mape = (
        float(np.mean(abs_errors[positive] / np.abs(actual_arr[positive])) * 100.0)
        if positive.any()
        else float("nan")
    )
    return {
        "MAD": float(np.mean(abs_errors)),
        "MSE": float(np.mean(errors**2)),
        "MAPE (%)": mape,
        "n": int(len(actual_arr)),
    }


def rolling_backtest(history: pd.DataFrame, alpha: float, window: int = 3, holdout: int = 14) -> pd.DataFrame:
    """One-step-ahead rolling evaluation, with training data ending before target."""
    hist = prepare_history(history)
    if len(hist) < 5:
        return pd.DataFrame()

    start = max(window, len(hist) - max(1, holdout))
    records = []
    for target_pos in range(start, len(hist)):
        train = hist.iloc[:target_pos].copy()
        target_date = pd.Timestamp(hist.iloc[target_pos]["Tanggal"])
        actual_value = float(hist.iloc[target_pos]["Permintaan Aktual (cup)"])
        target_promo = int(hist.iloc[target_pos].get("Promo (0/1)", 0))
        predictions, _ = forecast_methods(
            train, target_date, alpha, moving_window=window, target_promo=target_promo
        )
        for method, value in predictions.items():
            records.append({
                "Tanggal": target_date,
                "Metode": method,
                "Aktual (cup)": actual_value,
                "Forecast (cup)": float(value),
                "Error": actual_value - float(value),
                "Absolute Error": abs(actual_value - float(value)),
            })
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Sidebar options
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Parameter Forecasting")
alpha = st.sidebar.slider(
    "Alpha (α) — Exponential Smoothing",
    min_value=0.01,
    max_value=1.00,
    value=0.30,
    step=0.01,
    help="Alpha yang lebih besar membuat forecast lebih cepat mengikuti permintaan aktual terbaru.",
)
moving_window = st.sidebar.slider(
    "Jumlah periode untuk Moving Average",
    min_value=2,
    max_value=7,
    value=3,
    step=1,
)
backtest_days = st.sidebar.slider(
    "Jumlah hari untuk evaluasi akurasi",
    min_value=7,
    max_value=28,
    value=14,
    step=1,
)
target_promo_label = st.sidebar.selectbox(
    "Apakah ada promosi pada hari forecast?",
    options=["Tidak ada promosi", "Ada promosi"],
    index=0,
)
target_promo = 1 if target_promo_label == "Ada promosi" else 0

# -----------------------------------------------------------------------------
# Historical data input
# -----------------------------------------------------------------------------
st.header("1. Data Permintaan Historis Mixue")
st.caption(
    "Data harian selama delapan minggu. Kolom Promo menggunakan 0 = tidak ada promosi dan 1 = ada promosi. "
    "Untuk tugas akademik, ganti angka simulasi dengan data studi kasus yang sebenarnya jika tersedia."
)

if "mixue_ch3_data" not in st.session_state:
    st.session_state["mixue_ch3_data"] = create_demo_data()

edited = st.data_editor(
    st.session_state["mixue_ch3_data"],
    key="mixue_ch3_historical_editor",
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    column_config={
        "Tanggal": st.column_config.DateColumn("Tanggal", disabled=True, format="DD/MM/YYYY"),
        "Permintaan Aktual (cup)": st.column_config.NumberColumn(
            "Permintaan Aktual (cup)", min_value=0, step=50, format="%d"
        ),
        "Promo (0/1)": st.column_config.NumberColumn(
            "Promo (0 = tidak, 1 = ya)", min_value=0, max_value=1, step=1, format="%d"
        ),
    },
)

history = prepare_history(edited)
if len(history) < 5:
    st.error("Masukkan sekurang-kurangnya 5 hari data valid agar metode dan evaluasi dapat dihitung.")
    st.stop()
if (history["Permintaan Aktual (cup)"] < 0).any():
    st.error("Permintaan tidak boleh bernilai negatif.")
    st.stop()

# Keep the edited values in session state so they persist across Streamlit reruns.
st.session_state["mixue_ch3_data"] = history.copy()

history_for_models = history.rename(columns={"Permintaan Aktual (cup)": "Demand"})
# Internal helper columns use stable names.
model_history = pd.DataFrame({
    "Tanggal": history["Tanggal"],
    "Permintaan Aktual (cup)": history["Permintaan Aktual (cup)"],
    "Promo (0/1)": history["Promo (0/1)"],
})
next_date = pd.Timestamp(history["Tanggal"].max()) + pd.Timedelta(days=1)

# -----------------------------------------------------------------------------
# Seasonal index
# -----------------------------------------------------------------------------
st.header("2. Seasonal Index — Indeks Pola Hari")
indices = get_weekday_indices(history)
seasonal_table = pd.DataFrame({
    "Hari": DAY_NAMES,
    "Rata-rata Aktual (cup)": [
        float(history.loc[history["Tanggal"].dt.dayofweek == i, "Permintaan Aktual (cup)"].mean())
        if (history["Tanggal"].dt.dayofweek == i).any() else float(history["Permintaan Aktual (cup)"].mean())
        for i in range(7)
    ],
    "Seasonal Index": [indices[i] for i in range(7)],
})
seasonal_table["Interpretasi"] = seasonal_table["Seasonal Index"].apply(
    lambda value: "Di atas rata-rata" if value > 1.02 else ("Di bawah rata-rata" if value < 0.98 else "Mendekati rata-rata")
)
st.dataframe(
    seasonal_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rata-rata Aktual (cup)": st.column_config.NumberColumn(format="%d"),
        "Seasonal Index": st.column_config.NumberColumn(format="%.3f"),
    },
)
st.caption(
    "Indeks hari dihitung sebagai rata-rata permintaan pada hari tersebut dibagi rata-rata seluruh hari. "
    "Indeks > 1 menunjukkan pola permintaan yang relatif lebih tinggi."
)

# -----------------------------------------------------------------------------
# Forecasts for next day
# -----------------------------------------------------------------------------
st.header("3. Forecast Permintaan Hari Berikutnya")
st.caption(f"Asumsi promosi untuk tanggal target: **{target_promo_label}**. Model regresi kausal menggunakan tren waktu dan indikator promosi historis.")
forecasts, indices_for_forecast = forecast_methods(
    model_history,
    next_date,
    alpha=alpha,
    moving_window=moving_window,
    target_promo=target_promo,
)
forecast_table = pd.DataFrame({
    "Metode": list(forecasts.keys()),
    "Tanggal Forecast": [next_date] * len(forecasts),
    "Forecast (cup)": [max(0, int(round(value))) for value in forecasts.values()],
})

best_model = None
if len(history) >= max(10, moving_window + 2):
    st.caption(
        f"Target forecast: {next_date.strftime('%d/%m/%Y')} ({DAY_NAMES[next_date.dayofweek]}). "
        "Semua metode memakai data historis hingga tanggal terakhir pada tabel."
    )

cols = st.columns(len(forecasts))
for col, (method, value) in zip(cols, forecasts.items()):
    with col:
        st.metric(method, f"{max(0, int(round(value))):,} cup")

fig_forecast = px.bar(
    forecast_table,
    x="Metode",
    y="Forecast (cup)",
    color="Metode",
    title=f"Perbandingan Forecast untuk {next_date.strftime('%d %b %Y')}",
    labels={"Metode": "Metode Forecasting", "Forecast (cup)": "Permintaan (cup)"},
)
fig_forecast.update_layout(showlegend=False, xaxis_tickangle=-20)
st.plotly_chart(fig_forecast, use_container_width=True)
st.dataframe(
    forecast_table,
    use_container_width=True,
    hide_index=True,
    column_config={"Forecast (cup)": st.column_config.NumberColumn(format="%d")},
)

# -----------------------------------------------------------------------------
# Forecast accuracy: rolling one-step-ahead backtest
# -----------------------------------------------------------------------------
st.header("4. Evaluasi Akurasi Forecast — Chapter 3")
st.markdown(
    "Evaluasi ini melakukan *rolling backtest*: setiap prediksi untuk satu hari dibuat hanya "
    "dengan data hari-hari sebelumnya. MAD dan MAPE lebih kecil biasanya menunjukkan forecast "
    "lebih akurat; MSE memberi penalti lebih besar pada kesalahan yang besar."
)
backtest = rolling_backtest(history, alpha=alpha, window=moving_window, holdout=backtest_days)

if backtest.empty:
    st.info("Data belum cukup untuk evaluasi. Tambahkan lebih banyak observasi historis.")
    accuracy_table = pd.DataFrame(columns=["Metode", "MAD", "MSE", "MAPE (%)", "Jumlah evaluasi"])
else:
    metric_records = []
    for method in METHODS:
        subset = backtest[backtest["Metode"] == method]
        metrics = accuracy_metrics(subset["Aktual (cup)"], subset["Forecast (cup)"])
        metric_records.append({
            "Metode": method,
            "MAD": metrics["MAD"],
            "MSE": metrics["MSE"],
            "MAPE (%)": metrics["MAPE (%)"],
            "Jumlah evaluasi": metrics["n"],
        })
    accuracy_table = pd.DataFrame(metric_records)
    if accuracy_table["MAPE (%)"].notna().any():
        best_row = accuracy_table.loc[accuracy_table["MAPE (%)"].idxmin()]
        best_model = str(best_row["Metode"])
        st.success(
            f"Berdasarkan MAPE terendah pada periode evaluasi yang dipilih, metode dengan hasil terbaik adalah "
            f"**{best_model}** (MAPE {best_row['MAPE (%)']:.2f}%). Ini hanya kesimpulan dari data yang sedang digunakan."
        )

    st.dataframe(
        accuracy_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "MAD": st.column_config.NumberColumn("MAD (cup)", format="%.2f"),
            "MSE": st.column_config.NumberColumn("MSE (cup²)", format="%.2f"),
            "MAPE (%)": st.column_config.NumberColumn("MAPE (%)", format="%.2f%%"),
            "Jumlah evaluasi": st.column_config.NumberColumn(format="%d"),
        },
    )
    accuracy_chart = accuracy_table.dropna(subset=["MAPE (%)"])
    if not accuracy_chart.empty:
        fig_accuracy = px.bar(
            accuracy_chart,
            x="Metode",
            y="MAPE (%)",
            color="Metode",
            title="Perbandingan MAPE — Lebih Rendah Umumnya Lebih Baik",
            labels={"MAPE (%)": "MAPE (%)", "Metode": "Metode Forecasting"},
        )
        fig_accuracy.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig_accuracy, use_container_width=True)

    with st.expander("Lihat detail kesalahan forecast pada setiap hari evaluasi"):
        st.dataframe(backtest, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Historical chart and practical interpretation
# -----------------------------------------------------------------------------
st.header("5. Pola Permintaan Historis")
history_chart = history.tail(28).copy()
history_chart["Hari"] = history_chart["Tanggal"].dt.strftime("%d/%m")
fig_history = px.line(
    history_chart,
    x="Tanggal",
    y="Permintaan Aktual (cup)",
    markers=True,
    title="Permintaan Aktual Mixue — 28 Hari Terakhir",
    labels={"Tanggal": "Tanggal", "Permintaan Aktual (cup)": "Permintaan (cup)"},
)
st.plotly_chart(fig_history, use_container_width=True)

st.header("6. Interpretasi untuk Operasional Mixue")
latest_actual = float(history.iloc[-1]["Permintaan Aktual (cup)"])
mean_actual = float(history["Permintaan Aktual (cup)"].mean())
selected_forecast = forecasts.get(best_model, forecasts["Simple Moving Average"]) if best_model else forecasts["Simple Moving Average"]
change_pct = ((selected_forecast - latest_actual) / latest_actual * 100.0) if latest_actual > 0 else float("nan")

interpretation_cols = st.columns(3)
with interpretation_cols[0]:
    st.metric("Aktual Hari Terakhir", f"{latest_actual:,.0f} cup")
with interpretation_cols[1]:
    st.metric("Rata-rata Aktual", f"{mean_actual:,.0f} cup/hari")
with interpretation_cols[2]:
    st.metric("Perubahan Forecast vs Aktual Terakhir", f"{change_pct:+.1f}%" if np.isfinite(change_pct) else "Tidak tersedia")

if np.isfinite(change_pct):
    if change_pct > 5:
        st.info("Forecast terpilih lebih tinggi dari aktual hari terakhir. Pertimbangkan menyiapkan bahan baku dan tenaga kerja secukupnya, lalu validasi dengan promosi, cuaca, dan kalender lokal.")
    elif change_pct < -5:
        st.info("Forecast terpilih lebih rendah dari aktual hari terakhir. Periksa stok agar tidak terjadi penumpukan bahan mudah rusak.")
    else:
        st.info("Forecast terpilih relatif dekat dengan aktual hari terakhir. Tetap evaluasi hasil setelah data aktual terbaru tersedia.")

st.markdown(
    "**Batasan interpretasi:** hasil model adalah perkiraan, bukan jaminan penjualan. "
    "Faktor seperti promosi, cuaca, lokasi gerai, hari libur, dan perubahan menu belum dimodelkan secara eksplisit."
)

# -----------------------------------------------------------------------------
# Download outputs
# -----------------------------------------------------------------------------
st.header("7. Unduh Hasil Analisis")
forecast_csv = forecast_table.to_csv(index=False).encode("utf-8-sig")
accuracy_csv = accuracy_table.to_csv(index=False).encode("utf-8-sig")
seasonal_csv = seasonal_table.to_csv(index=False).encode("utf-8-sig")
backtest_csv = backtest.to_csv(index=False).encode("utf-8-sig") if not backtest.empty else b""

download_cols = st.columns(4)
with download_cols[0]:
    st.download_button(
        "Unduh Forecast CSV", data=forecast_csv, file_name="mixue_forecast_chapter3.csv", mime="text/csv"
    )
with download_cols[1]:
    st.download_button(
        "Unduh Akurasi CSV", data=accuracy_csv, file_name="mixue_accuracy_chapter3.csv", mime="text/csv"
    )
with download_cols[2]:
    st.download_button(
        "Unduh Seasonal Index", data=seasonal_csv, file_name="mixue_seasonal_index.csv", mime="text/csv"
    )
with download_cols[3]:
    st.download_button(
        "Unduh Backtest CSV", data=backtest_csv, file_name="mixue_backtest_chapter3.csv", mime="text/csv"
    )

st.divider()
st.caption(
    "Mixue Demand Forecasting | Concepts aligned with Operations Management, Chapter 3: Forecasting. "
    "Data defaults are simulated for demonstration only."
)
