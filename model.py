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
                                                 test_size=0.5,
                                                 random_state=1)

# 기준선 세우기
기준값 = y_train.mean()

공부_기준예측 = y_train * 0 + 기준값
시험_기준예측 = y_test * 0 + 기준값

공부_오차 = mean_squared_error(y_train, 공부_기준예측)
시험_오차 = mean_squared_error(y_test, 시험_기준예측)
print(f"{공부_오차:,.0f}, {시험_오차:,.0f}")