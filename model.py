#1. model.py
# 도구 가겨오기
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# csv 파일 불러오기
df = pd.read_csv("야구.ver1.csv")

# 힌트와 정답 나누기
x = df[["인기도","기온","주말여부","최근10경기팀타율","최근10경기평균자책점","구장수용인원"]]
y = df["관중수"]

# 공부용과 시험용으로 나누기
x_train,x_test,y_train,y_test = train_test_split(x,
                                                 y,
                                                 test_size=0.2,
                                                 random_state=1)

# 기준선 세우기
기준값 = y_train.mean()

공부_기준예측 = y_train * 0 + 기준값
시험_기준예측 = y_test * 0 + 기준값

공부_오차 = mean_squared_error(y_train, 공부_기준예측)
시험_오차 = mean_squared_error(y_test, 시험_기준예측)
print(f"기준값: {기준값:,.0f}")
print(f"공부용 오차: {공부_오차:,.0f}")
print(f"시험용 오차: {시험_오차:,.0f}")

# 머신러닝 모델 가져오기
model = LinearRegression()
# 데이터 학습 시키기
model.fit(x_train,y_train)

# 공부용과 시험용 예측하기
공부_예측 = model.predict(x_train)
시험_예측 = model.predict(x_test)

# 머신러닝 오차 구하기
공부_MSE = mean_squared_error(y_train, 공부_예측)
시험_MSE = mean_squared_error(y_test, 시험_예측)

print("==== 머신러닝 예측 ====")
print(f"공부용 오차: {공부_MSE:,.0f}")
print(f"시험용 오차: {시험_MSE:,.0f}")

# 베이스라인과 머신러닝 비교하기
print()
print(f"베이스라인 시험 오차: {시험_오차:,.0f}")
print(f"머신러닝 시험 오차: {시험_MSE:,.0f}")

if 시험_MSE < 시험_오차:
    print("머신러닝이 베이스라인보다 예측을 잘했습니다.")
else:
    print("베이스라인이 머신러닝보다 예측을 잘했거나 비슷합니다.")


# 머신러닝이 학습한 규칙 확인하기
print()
print("===== 머신러닝이 학습한 규칙 =====")
print("기울기:", model.coef_)
print("절편:", model.intercept_)