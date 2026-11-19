import pandas as pd
import streamlit as st

df = pd.read_csv("야구.ver4_구장수정.csv")

print(df.head())
print(df.shape)

# 유효 변수 4개
X_4 = df[[
    "인기도",
    "기온",
    "주말여부",
    "구장수용인원"
]]

# 허수 변수까지 포함한 6개
X_6 = df[[
    "인기도",
    "기온",
    "주말여부",
    "구장수용인원",
    "최근10경기팀타율",
    "최근10경기평균자책점"
]]

# 정답(예측하려는 관중 수)
y = df["관중수"]

print(X_4.head())
print(X_6.head())
print(y.head())

from sklearn.linear_model import LinearRegression

# 1. 머신러닝 모델 만들기
model_4 = LinearRegression()

# 2. 유효 변수 4개로 학습하기
model_4.fit(X_4, y)

# 3. 학습한 모델로 관중 수 예측하기
예측값_4 = model_4.predict(X_4)

# 4. 예측 결과 확인하기
print(예측값_4[:5])

# 학습용 데이터 (처음 576경기)
X_train = X_4.iloc[:576]
y_train = y.iloc[:576]

# 테스트용 데이터 (나머지 144경기)
X_test = X_4.iloc[576:]
y_test = y.iloc[576:]

print("학습 데이터:", X_train.shape)
print("테스트 데이터:", X_test.shape)

# 1. 새로운 모델 만들기
model_4 = LinearRegression()

# 2. 학습용 576경기로 학습
model_4.fit(X_train, y_train)

# 3. 테스트용 144경기 예측
예측값_4 = model_4.predict(X_test)

# 4. 예측 결과 5개 확인
print("AI 예측 관중 수:")
print(예측값_4[:5])

print("실제 관중 수:")
print(y_test.head())

from sklearn.metrics import mean_absolute_error

mae_4 = mean_absolute_error(y_test, 예측값_4)

print("4개 변수 모델 MAE:", round(mae_4, 2), "명")

# 1. 허수 포함 6개 변수 데이터 나누기
X_train_6 = X_6.iloc[:576]
X_test_6 = X_6.iloc[576:]

# 2. 6개 변수 모델 만들기
model_6 = LinearRegression()

# 3. 학습용 576경기로 학습
model_6.fit(X_train_6, y_train)

# 4. 테스트용 144경기 예측
예측값_6 = model_6.predict(X_test_6)

# 5. MAE 계산
mae_6 = mean_absolute_error(y_test, 예측값_6)

# 6. 결과 비교
print("4개 변수 MAE:", round(mae_4, 2), "명")
print("6개 변수 MAE:", round(mae_6, 2), "명")