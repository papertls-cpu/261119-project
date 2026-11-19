
# streamlit run 야구.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. 데이터 불러오기
df = pd.read_csv("야구.ver1.csv")

st.title("⚾ 야구 관중 수 예측 AI")

# 2. 학습 데이터 설정
features = [
    "홈팀", "원정팀", "인기도", "기온", "주말여부",
    "최근10경기팀타율", "최근10경기평균자책점", "구장수용인원"
]

X = pd.get_dummies(df[features], columns=["홈팀", "원정팀"])
y = df["관중수"]

# 3. 데이터 분리
split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# 4. 머신러닝 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 모델 평가
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)

st.subheader("📊 데이터 분석")
st.write("전체 경기 수:", len(df))
st.write("평균 관중 수:", round(df["관중수"].mean()), "명")
st.write("평균 예측 오차:", round(mae), "명")

# 6. 경기 정보 입력
st.subheader("🔮 관중 수 예측")

home = st.selectbox("홈팀", df["홈팀"].unique())
away = st.selectbox("원정팀", df["원정팀"].unique())
popularity = st.slider("인기도", 1, 10, 8)
temperature = st.number_input("기온", value=25)
weekend = st.selectbox("주말 여부", ["평일", "주말"])
batting = st.number_input("최근 10경기 팀타율", value=0.260)
era = st.number_input("최근 10경기 평균자책점", value=4.20)
capacity = st.selectbox(
    "구장 수용 인원",
    sorted(df["구장수용인원"].unique())
)

# 7. 관중 수 예측
if st.button("관중 수 예측하기"):

    new_data = pd.DataFrame([{
        "홈팀": home,
        "원정팀": away,
        "인기도": popularity,
        "기온": temperature,
        "주말여부": 1 if weekend == "주말" else 0,
        "최근10경기팀타율": batting,
        "최근10경기평균자책점": era,
        "구장수용인원": capacity
    }])

    new_data = pd.get_dummies(
        new_data, columns=["홈팀", "원정팀"]
    )
    new_data = new_data.reindex(columns=X.columns, fill_value=0)
    
    result = model.predict(new_data)[0]

    st.success(f"예상 관중 수: {max(0, result):,.0f}명")

# 8. 데이터 확인
st.subheader("📋 전체 경기 데이터")
st.dataframe(df)
