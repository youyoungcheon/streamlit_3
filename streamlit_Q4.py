import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sunspot Forecast", layout="wide")
st.title("🌞 Prophet Forecast with Preprocessed Sunspot Data")

# ----------------------------------
# [1] CSV 불러오기 및 날짜 형식 변환
# ----------------------------------
df = pd.read_csv("./sunspots_for_prophet.csv")
df["ds"] = pd.to_datetime(df["ds"])  # datetime 형식 재지정
st.subheader("📄 불러온 데이터 미리보기")
st.write(df.head())

# ----------------------------------
# [2] Prophet 모델 학습
# ----------------------------------
model = Prophet(
    yearly_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_mode='additive'
)
model.add_seasonality(name='sunspot_cycle', period=11, fourier_order=5)
model.fit(df)

# ----------------------------------
# [3] 예측
# ----------------------------------
future = model.make_future_dataframe(periods=30, freq="Y")
forecast = model.predict(future)

# ----------------------------------
# [4] 기본 시각화
# ----------------------------------
st.subheader("📈 Prophet Forecast Plot")
fig1 = model.plot(forecast)
plt.title("Prophet Forecast Plot")
st.pyplot(fig1)

st.subheader("📊 Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# ----------------------------------
# [5] 잔차 분석 (Residual Analysis)
# ----------------------------------
st.subheader("📉 Residual Analysis (예측 오차 분석)")

# 예측값과 실제값 병합
merged = pd.merge(df, forecast[["ds", "yhat"]], on="ds", how="inner")
merged["residual"] = merged["y"] - merged["yhat"]

# 잔차 시각화
fig3, ax = plt.subplots(figsize=(14, 4))
ax.plot(merged["ds"], merged["residual"], marker="o", linestyle="-", color="purple", label="Residual")
ax.axhline(0, color="black", linestyle="--")
ax.set_title("Residuals Over Time (Actual - Predicted)")
ax.set_xlabel("Year")
ax.set_ylabel("Residual")
ax.legend()
ax.grid(True)
st.pyplot(fig3)

# 잔차 통계
st.subheader("📌 Residual Summary Statistics")
st.dataframe(merged["residual"].describe().to_frame())
