import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import ccxt
import yfinance as yf
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d
from datetime import datetime, timedelta

# =============================
# CONFIG & THEME
# =============================
st.set_page_config(
    page_title="Crypto Market Spectral Decomposition Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark quant theme for Plotly
PLOTLY_TEMPLATE = "plotly_dark"
SPECTRAL_COLORS = px.colors.sequential.Viridis
NEON_COLORS = ["#00ffe7", "#ff00c8", "#00ff6a", "#fffb00", "#ff0055", "#00baff"]

# =============================
# DATA LOADING
# =============================
def load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower() for c in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['date'] = pd.to_datetime(df.index)
    df = df.sort_values('date')
    return df

def fetch_ccxt(symbol, exchange_name, limit=1000):
    exchange = getattr(ccxt, exchange_name)()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def fetch_yfinance(symbol, period='60d', interval='1h'):
    data = yf.download(symbol, period=period, interval=interval)
    data = data.reset_index()
    data.columns = [c.lower() for c in data.columns]
    # Always use 'close' for crypto assets
    if 'close' not in data.columns:
        raise ValueError("Downloaded data does not contain 'close' column. Columns: {}".format(data.columns))
    data['date'] = pd.to_datetime(data['datetime'] if 'datetime' in data.columns else data['date'])
    # Only keep relevant columns
    keep_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    data = data[[c for c in keep_cols if c in data.columns]]
    return data

# =============================
# SIGNAL PROCESSING
# =============================
def compute_log_returns(df):
    df['log_return'] = np.log(df['close']).diff()
    return df

def rolling_normalized_returns(df, window):
    returns = df['log_return'].fillna(0)
    normed = (returns - returns.rolling(window).mean()) / (returns.rolling(window).std() + 1e-8)
    return normed

def rolling_fft(df, window, fft_res, smoothing):
    times = df['date'].values
    normed = rolling_normalized_returns(df, window)
    n_windows = len(normed) - window + 1
    freq_grid = np.fft.rfftfreq(window, d=1)
    spectral_surface = np.zeros((n_windows, len(freq_grid)))
    dom_freqs = []
    dom_amps = []
    for i in range(n_windows):
        segment = normed.iloc[i:i+window].values
        if smoothing:
            segment = gaussian_filter1d(segment, sigma=2)
        fft_vals = np.fft.rfft(segment, n=window)
        power = np.abs(fft_vals)**2
        spectral_surface[i, :] = power
        dom_idx = np.argmax(power)
        dom_freqs.append(freq_grid[dom_idx])
        dom_amps.append(power[dom_idx])
    return spectral_surface, freq_grid, dom_freqs, dom_amps, times[window-1:]

def latest_fft(df, window, smoothing):
    normed = rolling_normalized_returns(df, window)
    segment = normed.iloc[-window:].values
    if smoothing:
        segment = gaussian_filter1d(segment, sigma=2)
    fft_vals = np.fft.rfft(segment, n=window)
    power = np.abs(fft_vals)**2
    freq_grid = np.fft.rfftfreq(window, d=1)
    return freq_grid, power

# =============================
# PLOTLY VISUALIZATIONS
# =============================
def plot_3d_surface(spectral_surface, freq_grid, times):
    fig = go.Figure(data=[
        go.Surface(
            z=spectral_surface.T,
            x=np.arange(len(times)),
            y=freq_grid,
            colorscale=SPECTRAL_COLORS,
            showscale=True,
            lighting=dict(ambient=0.7, diffuse=0.8, specular=0.5, roughness=0.5, fresnel=0.2),
            hovertemplate="Time: %{x}<br>Freq: %{y:.4f}<br>Power: %{z:.2f}<extra></extra>",
        )
    ])
    fig.update_layout(
        title="3D Rolling Spectral Surface",
        autosize=True,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title="Time",
            yaxis_title="Frequency",
            zaxis_title="Spectral Power",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode="manual",
            aspectratio=dict(x=2, y=1, z=0.7),
        ),
        font=dict(size=18, color="#00ffe7"),
        paper_bgcolor="#111",
        plot_bgcolor="#111",
    )
    fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="#ff00c8", project_z=True))
    return fig

def plot_dominant_freq(times, dom_freqs, dom_amps):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=dom_freqs,
        mode="lines+markers",
        line=dict(color=NEON_COLORS[0], width=3),
        marker=dict(size=8, color=dom_amps, colorscale="Viridis", showscale=True),
        name="Dominant Frequency"
    ))
    fig.update_layout(
        title="Dominant Frequency Over Time",
        template=PLOTLY_TEMPLATE,
        font=dict(size=18, color="#ff00c8"),
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        margin=dict(l=0, r=0, b=0, t=40),
        xaxis_title="Time",
        yaxis_title="Frequency",
    )
    return fig

def plot_fft_spectrum(freq_grid, power):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=freq_grid,
        y=power,
        marker=dict(color=power, colorscale="Plasma"),
        name="Spectral Power"
    ))
    dom_idx = np.argmax(power)
    fig.add_trace(go.Scatter(
        x=[freq_grid[dom_idx]],
        y=[power[dom_idx]],
        mode="markers",
        marker=dict(size=16, color="#fffb00", symbol="star"),
        name="Dominant Frequency"
    ))
    fig.update_layout(
        title="FFT Spectrum (Latest Window)",
        template=PLOTLY_TEMPLATE,
        font=dict(size=18, color="#00ff6a"),
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        margin=dict(l=0, r=0, b=0, t=40),
        xaxis_title="Frequency",
        yaxis_title="Spectral Power",
    )
    return fig

# =============================
# STREAMLIT UI
# =============================
ASSETS = {
    "BTC/USDT (Binance)": {"type": "ccxt", "symbol": "BTC/USDT", "exchange": "binance"},
    "ETH/USDT (Binance)": {"type": "ccxt", "symbol": "ETH/USDT", "exchange": "binance"},
    "SOL/USDT (Binance)": {"type": "ccxt", "symbol": "SOL/USDT", "exchange": "binance"},
    "BTC-USD (Yahoo)": {"type": "yfinance", "symbol": "BTC-USD"},
    "ETH-USD (Yahoo)": {"type": "yfinance", "symbol": "ETH-USD"},
    "SOL-USD (Yahoo)": {"type": "yfinance", "symbol": "SOL-USD"},
}

st.sidebar.title("Crypto Market Spectral Decomposition Engine")
asset = st.sidebar.selectbox("Select Asset", list(ASSETS.keys()))
window = st.sidebar.slider("Rolling Window Size", min_value=32, max_value=512, value=128, step=8)
fft_res = st.sidebar.slider("FFT Resolution", min_value=32, max_value=512, value=128, step=8)
smoothing = st.sidebar.checkbox("Apply Smoothing", value=True)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload CSV (OHLCV)", type=["csv"])

st.title("Crypto Market Spectral Decomposition Engine")
st.markdown("""
<div style='font-size:2.2em; color:#00ffe7; font-weight:bold;'>
Analyze crypto price dynamics with rolling spectral decomposition and cinematic 3D visualizations.
</div>
""", unsafe_allow_html=True)

# =============================
# DATA INGESTION
# =============================
if uploaded_file:
    df = load_csv(uploaded_file)
    source = "CSV Upload"
else:
    info = ASSETS[asset]
    if info["type"] == "ccxt":
        df = fetch_ccxt(info["symbol"], info["exchange"], limit=1500)
        source = f"Live {info['symbol']} from {info['exchange'].capitalize()}"
    else:
        df = fetch_yfinance(info["symbol"], period="90d", interval="1h")
        source = f"Live {info['symbol']} from Yahoo Finance"

# =============================
# METRIC PANELS
# =============================
df = compute_log_returns(df)
latest_close = df['close'].iloc[-1]
latest_return = df['log_return'].iloc[-1]
volatility = df['log_return'].rolling(window).std().iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Latest Close", f"{latest_close:,.2f}")
col2.metric("Latest Log Return", f"{latest_return:.4f}")
col3.metric("Rolling Volatility", f"{volatility:.4f}")

st.markdown(f"<span style='color:#ff00c8;font-size:1.2em;'>Source: {source}</span>", unsafe_allow_html=True)

# =============================
# SPECTRAL ANALYSIS
# =============================
spectral_surface, freq_grid, dom_freqs, dom_amps, times = rolling_fft(df, window, fft_res, smoothing)
freq_grid_latest, power_latest = latest_fft(df, window, smoothing)

# =============================
# VISUALIZATION
# =============================
st.markdown("---")

# Cinematic 3D Spectral Surface
st.plotly_chart(plot_3d_surface(spectral_surface, freq_grid, times), use_container_width=True)

# Animated time slider for spectral evolution
frame_idx = st.slider("Spectral Evolution Time Slider", min_value=0, max_value=len(times)-1, value=len(times)-1, step=1)
frame_power = spectral_surface[frame_idx, :]
frame_time = times[frame_idx]

st.subheader(f"Spectral Power at {pd.to_datetime(frame_time).strftime('%Y-%m-%d %H:%M')}")
st.plotly_chart(plot_fft_spectrum(freq_grid, frame_power), use_container_width=True)

# Dominant frequency time series
st.subheader("Dominant Frequency Evolution")
st.plotly_chart(plot_dominant_freq(times, dom_freqs, dom_amps), use_container_width=True)

# FFT spectrum for latest window
st.subheader("FFT Spectrum (Latest Window)")
st.plotly_chart(plot_fft_spectrum(freq_grid_latest, power_latest), use_container_width=True)

st.markdown("""
<style>
    .stPlotlyChart {background: #111 !important;}
    .stApp {background: #111 !important;}
</style>
""", unsafe_allow_html=True)
