import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, gaussian_kde

# -----------------------------------
# 🌞 페이지 설정
# -----------------------------------
st.set_page_config(page_title="Sunspot Data Interactive Analysis", layout="wide")
st.title("🌞 Interactively Explore Sunspot Activity Data")

# -----------------------------------
# 📊 데이터 로딩 및 전처리
# -----------------------------------
df = pd.read_csv("./sunspots.csv")
df['y'] = df['SUNACTIVITY']
df["YEAR"] = df["YEAR"].astype(int)
df["date"] = pd.to_datetime(df["YEAR"], format="%Y")
df = df.set_index("date")
df = df.sort_index()
sunactivity = df["y"].dropna()

# -----------------------------------
# 🧭 사이드바 - 설정 UI
# -----------------------------------
st.sidebar.title("🔧 시각화 설정")

min_year = int(df["YEAR"].min())
max_year = int(df["YEAR"].max())

selected_years = st.sidebar.slider(
    "분석 연도 범위 선택", min_value=min_year, max_value=max_year, value=(1900, 2000)
)
bins = st.sidebar.slider("히스토그램 구간 수 (bins)", min_value=10, max_value=100, value=30)
show_ma = st.sidebar.checkbox("이동 평균선 표시", value=True)
ma_window = st.sidebar.slider("이동 평균 기간 (연)", min_value=5, max_value=50, value=11)

# -----------------------------------
# 📌 데이터 필터링
# -----------------------------------
df_filtered = df[(df["YEAR"] >= selected_years[0]) & (df["YEAR"] <= selected_years[1])]
sunactivity = df_filtered["SUNACTIVITY"].dropna()

# -----------------------------------
# 1. 기본 통계 요약
# -----------------------------------
st.header("1️⃣ 기본 통계 요약 및 분포 분석")
st.subheader("📊 통계 요약")
st.dataframe(df_filtered.describe())

if st.checkbox("📈 왜도/첨도 출력"):
    st.write(f"📉 왜도(Skewness): {skew(sunactivity):.4f}")
    st.write(f"📈 첨도(Kurtosis): {kurtosis(sunactivity):.4f}")

# -----------------------------------
# 2. 분포 시각화
# -----------------------------------
st.subheader("📈 분포 시각화")
fig1, ax1 = plt.subplots()
xs = np.linspace(sunactivity.min(), sunactivity.max(), 200)
density = gaussian_kde(sunactivity)

ax1.hist(sunactivity, bins=bins, density=True, alpha=0.6, label='Histogram', color='skyblue')
ax1.plot(xs, density(xs), color='red', label='Density')
ax1.set_title("Sunspot Activity Distribution")
ax1.set_xlabel("SUNACTIVITY")
ax1.set_ylabel("Density")
ax1.legend()
st.pyplot(fig1)

# -----------------------------------
# 3. 결측치 & 이상치 탐지
# -----------------------------------
st.header("2️⃣ 결측치 및 이상치 탐지")

st.subheader("🧩 결측치 확인")
st.write(df_filtered.isnull().sum())

Q1 = sunactivity.quantile(0.25)
Q3 = sunactivity.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df_filtered[(df_filtered["SUNACTIVITY"] < lower_bound) | (df_filtered["SUNACTIVITY"] > upper_bound)]

st.subheader("🚨 이상치 탐지 (IQR 방법)")
st.write(f"하한: {lower_bound:.2f}, 상한: {upper_bound:.2f}")
st.dataframe(outliers[["YEAR", "SUNACTIVITY"]])

# -----------------------------------
# 4. 심화 시각화: 다중 서브플롯
# -----------------------------------
st.header("3️⃣ 심화 시각화")

fig2, axs = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle("Sunspots Data Advanced Visualization", fontsize=20)

# (a) 시계열 라인차트 + MA
axs[0, 0].plot(df_filtered.index, sunactivity, label='Original', color='blue')
if show_ma:
    ma_series = sunactivity.rolling(window=ma_window, min_periods=1).mean()
    axs[0, 0].plot(df_filtered.index, ma_series, linestyle='--', color='orange', label=f"{ma_window}Y MA")
axs[0, 0].set_title("Sunspot Activity Over Time")
axs[0, 0].set_xlabel("Year")
axs[0, 0].set_ylabel("SUNACTIVITY")
axs[0, 0].grid(True)
axs[0, 0].legend()

# (b) 히스토그램 + KDE
axs[0, 1].hist(sunactivity, bins=bins, density=True, alpha=0.6, color='gray', label='Histogram')
axs[0, 1].plot(xs, density(xs), color='red', label='Density')
axs[0, 1].set_title("Distribution of Sunspot Activity")
axs[0, 1].set_xlabel("SUNACTIVITY")
axs[0, 1].set_ylabel("Density")
axs[0, 1].legend()
axs[0, 1].grid(True)

# (c) 박스플롯
axs[1, 0].boxplot(sunactivity, vert=False)
axs[1, 0].set_title(f"Boxplot of Sunspot Activity ({selected_years[0]}–{selected_years[1]})")
axs[1, 0].set_xlabel("SUNACTIVITY")

# (d) 산점도 + 회귀선
years = df_filtered["YEAR"]
axs[1, 1].scatter(years, sunactivity, s=10, alpha=0.5, label='Data Points')
coef = np.polyfit(years, sunactivity, 1)
trend = np.poly1d(coef)
axs[1, 1].plot(years, trend(years), color='red', linewidth=2, label='Trend Line')
axs[1, 1].set_title("Trend of Sunspot Activity")
axs[1, 1].set_xlabel("Year")
axs[1, 1].set_ylabel("SUNACTIVITY")
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
st.pyplot(fig2)
