import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.title("🌞 Sunspot Forecast with Prophet")

# 1. 데이터 불러오기
df = pd.read_csv("sunspots_for_prophet.csv")
df["ds"] = pd.to_datetime(df["ds"])

st.write("데이터 미리보기", df.head())

# 2. Prophet 모델 학습
model = Prophet(
    yearly_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_mode='additive'
)
model.add_seasonality(name='sunspot_cycle', period=11, fourier_order=5)
model.fit(df)

# 3. 예측
future = model.make_future_dataframe(periods=30, freq="Y")
forecast = model.predict(future)

# 4. 시각화
fig1 = model.plot(forecast)
plt.title("Prophet Forecast Plot")
st.pyplot(fig1)

fig2 = model.plot_components(forecast)
st.pyplot(fig2)
