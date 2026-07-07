import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)


# ---------------- CACHED MODEL ---------------- #

@st.cache_resource
def load_stock_model():
    return load_model("keras_model.keras")


# ---------------- HELPER FUNCTIONS ---------------- #

def clean_yfinance_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean yfinance output and handle MultiIndex columns."""
    if dataframe.empty:
        return dataframe

    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = dataframe.columns.get_level_values(0)

    dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()]
    return dataframe


def get_close_series(dataframe: pd.DataFrame) -> pd.Series:
    """Return a clean closing price Series."""
    close_data = dataframe["Close"]

    if isinstance(close_data, pd.DataFrame):
        close_data = close_data.iloc[:, 0]

    close_data = pd.to_numeric(close_data, errors="coerce").dropna()
    return close_data


# ---------------- HEADER ---------------- #

st.markdown(
    """
    <h1 style='text-align:center; font-size:48px;'>📈 Stock Market Prediction Dashboard</h1>
    <p style='text-align:center; color:gray; font-size:18px;'>
        Analyze historical stock prices, moving averages, and AI-based price predictions.
    </p>
    """,
    unsafe_allow_html=True
)


# ---------------- INPUT SECTION ---------------- #

col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input(
        "Enter Stock Ticker",
        "AAPL",
        placeholder="Examples: AAPL, NVDA, TSLA, META"
    ).upper().strip()

with col2:
    st.write("")
    st.write("")
    analyze = st.button("Analyze", use_container_width=True)

if not analyze:
    st.stop()


# ---------------- DOWNLOAD DATA ---------------- #

start = dt(2015, 1, 1)
end = dt.today()

try:
    df = yf.download(
        user_input,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False
    )

    df = clean_yfinance_data(df)

    if df.empty or "Close" not in df.columns:
        st.error("❌ No stock data found. Please enter a valid ticker.")
        st.stop()

    close_data = get_close_series(df)

    if close_data.empty:
        st.error("❌ Closing price data is unavailable for this ticker.")
        st.stop()

except Exception as e:
    st.error(f"❌ Unable to fetch stock data: {e}")
    st.stop()


# ---------------- METRICS ---------------- #

latest_close = float(close_data.iloc[-1])
highest = float(pd.to_numeric(df["High"], errors="coerce").max())
lowest = float(pd.to_numeric(df["Low"], errors="coerce").min())

m1, m2, m3 = st.columns(3)

m1.metric("Latest Close", f"${latest_close:.2f}")
m2.metric("Highest Price", f"${highest:.2f}")
m3.metric("Lowest Price", f"${lowest:.2f}")


# ---------------- HISTORICAL DATA ---------------- #

st.subheader("Historical Data")
st.dataframe(df.tail(10), use_container_width=True)


# ---------------- CLOSING PRICE CHART ---------------- #

st.subheader("Closing Price Over Time")

fig_close = go.Figure()

fig_close.add_trace(
    go.Scatter(
        x=close_data.index,
        y=close_data.values,
        mode="lines",
        name="Close Price",
        line=dict(width=2)
    )
)

fig_close.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    height=500,
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig_close, use_container_width=True)


# ---------------- MOVING AVERAGES ---------------- #

st.subheader("100-Day & 200-Day Moving Averages")

ma100 = close_data.rolling(100).mean()
ma200 = close_data.rolling(200).mean()

fig_ma = go.Figure()

fig_ma.add_trace(
    go.Scatter(
        x=close_data.index,
        y=close_data.values,
        mode="lines",
        name="Close Price",
        line=dict(width=2)
    )
)

fig_ma.add_trace(
    go.Scatter(
        x=ma100.index,
        y=ma100.values,
        mode="lines",
        name="100-Day MA",
        line=dict(width=2)
    )
)

fig_ma.add_trace(
    go.Scatter(
        x=ma200.index,
        y=ma200.values,
        mode="lines",
        name="200-Day MA",
        line=dict(width=2)
    )
)

fig_ma.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    height=500,
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig_ma, use_container_width=True)


# ---------------- MODEL PREDICTION ---------------- #

st.subheader("AI Prediction vs Actual Price")

try:
    if len(close_data) < 250:
        st.error("❌ Not enough historical data for prediction. Try another ticker.")
        st.stop()

    scaler = MinMaxScaler(feature_range=(0, 1))

    training_data = pd.DataFrame(close_data.iloc[: int(len(close_data) * 0.70)])
    testing_data = pd.DataFrame(close_data.iloc[int(len(close_data) * 0.70):])

    if len(training_data) < 100 or len(testing_data) < 100:
        st.error("❌ Not enough data for prediction. Try another ticker.")
        st.stop()

    scaler.fit_transform(training_data)

    with st.spinner("Loading AI model and generating predictions..."):
        model = load_stock_model()

        past_100_days = training_data.tail(100)
        final_df = pd.concat([past_100_days, testing_data], ignore_index=True)

        input_data = scaler.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        prediction = model.predict(x_test, verbose=0)

        scale = scaler.scale_[0]
        prediction = prediction / scale
        y_test = y_test / scale

except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    st.stop()


# ---------------- PREDICTION CHART ---------------- #

fig_pred = go.Figure()

fig_pred.add_trace(
    go.Scatter(
        y=y_test,
        mode="lines",
        name="Actual Price",
        line=dict(width=2)
    )
)

fig_pred.add_trace(
    go.Scatter(
        y=prediction.flatten(),
        mode="lines",
        name="Predicted Price",
        line=dict(width=2)
    )
)

fig_pred.update_layout(
    template="plotly_dark",
    xaxis_title="Trading Days",
    yaxis_title="Price ($)",
    height=500,
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig_pred, use_container_width=True)

st.success("Analysis Complete ✅")
