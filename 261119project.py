import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 0. 데이터 준비 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('야구.ver4_구장수정.csv')

기본특성 = ['인기도', '기온', '주말여부', '최근10경기팀타율',
            '최근10경기평균자책점', '구장수용인원']

# 특성공학
# 다항회귀: 기온의 비선형 관계 반영
# 교호작용: 구장 크기에 따른 주말 효과 반영
df['기온제곱'] = df['기온'] ** 2
df['수용인원_주말'] = df['구장수용인원'] * df['주말여부']
다항특성 = 기본특성 + ['기온제곱']
최종특성 = 다항특성 + ['수용인원_주말']

# 세 모델 모두 동일한 학습/테스트 데이터 사용
train, test = train_test_split(df, test_size=0.2, random_state=42)
X_train = train[기본특성]
y_train = train['관중수']
y_test = test['관중수']

model1 = LinearRegression().fit(X_train, y_train)
model2 = LinearRegression().fit(train[다항특성], y_train)
model3 = LinearRegression().fit(train[최종특성], y_train)

예측1 = model1.predict(test[기본특성])
예측2 = model2.predict(test[다항특성])
예측3 = model3.predict(test[최종특성])

# Baseline: 모델을 학습하지 않고 학습 데이터의 평균 관중수만 예측
baseline_평균 = y_train.mean()
baseline_예측 = pd.Series(baseline_평균, index=y_test.index)
baseline_MAE = mean_absolute_error(y_test, baseline_예측)
기존_MAE = mean_absolute_error(y_test, 예측1)

# 회귀계수에 변수 이름 붙이기
coef = pd.Series(model1.coef_, index=기본특성)
표준편차 = X_train.std(ddof=0)
영향력 = coef * 표준편차

# 그래프에 표시할 변수 이름만 짧게 변경
표시이름 = {
    "최근10경기팀타율": "팀타율",
    "최근10경기평균자책점": "평균자책점"
}

# 발표 화면
st.title('프로야구 관중수 예측 분석')
st.caption('교육용 모의 데이터 | 학습 80%, 테스트 20%')
단계 = st.sidebar.radio('발표 순서', [
    '1. 기존 회귀계수',
    '2. 표준편차 반영',
    '3. 변수별 산점도 분석',
    '4. 다항회귀 적용',
    '5. 교호작용 적용',
    '6. 최종 오차 비교',
    '7. 최종 관중수 예측 AI'
])

# 1. 기존 회귀계수
if 단계 == '1. 기존 회귀계수':
    st.header('1. 기존 선형회귀의 회귀계수')
    st.subheader('Baseline과 기존 선형회귀의 예측 오차')
    st.write('Baseline은 모든 경기의 관중수를 학습 데이터의 평균값으로 예측하는 가장 단순한 기준 모델입니다.')
    열1, 열2 = st.columns(2)
    열1.metric('Baseline MAE', f'{baseline_MAE:,.2f}명')
    열2.metric('기존 선형회귀 MAE', f'{기존_MAE:,.2f}명')
    st.metric('Baseline 대비 MAE 감소율', f'{(baseline_MAE - 기존_MAE) / baseline_MAE * 100:.1f}%')
    비교 = pd.Series(
        [baseline_MAE, 기존_MAE],
        index=['Baseline (평균값 예측)', '기존 선형회귀'],
        name='평균 절대 오차(MAE)'
    )
    st.bar_chart(비교, horizontal=True)
    st.caption(f'Baseline 예측값: 매 경기 {baseline_평균:,.0f}명 | 동일한 테스트 데이터 {len(y_test)}경기로 평가')
    st.divider()
    st.subheader('기존 선형회귀의 회귀계수')
    st.bar_chart(
        coef.abs().rename(표시이름).sort_values(ascending=False).rename('회귀계수 절댓값'),
        horizontal=True
    )
    st.dataframe(coef.rename('회귀계수'))
    st.write('변수마다 단위가 다르기 때문에 회귀계수만으로 영향력을 비교하기 어렵습니다.')

# 2. 표준편차를 곱한 선형 영향력
elif 단계 == '2. 표준편차 반영':
    st.header('2. 표준편차를 반영한 영향력')
    st.bar_chart(
        영향력.abs().rename(표시이름).sort_values(ascending=False).rename('예측 관중수 변화량(명)'),
        horizontal=True
    )
    st.dataframe(pd.DataFrame({
        '회귀계수': coef,
        '표준편차': 표준편차,
        '영향력(명)': 영향력
    }).rename(index=표시이름))
    st.write('다른 조건이 같을 때 변수가 1표준편차 증가하면 예측 관중수가 얼마나 변하는지 확인합니다.')

    # 표준편차 설명용 사진 (파이썬 파일과 같은 폴더에 저장)
    st.subheader('표준편차란 무엇인가?')
    st.write('표준편차는 데이터가 평균을 중심으로 얼마나 퍼져 있는지를 나타내는 숫자입니다.')
    st.image('standard_deviation.png')
    st.caption('실제 학습 데이터의 평균과 표준편차를 표시한 개념도입니다. 곡선은 실제 분포가 아닙니다.')

    st.subheader('회귀계수에 표준편차를 곱하면?')
    for 변수, 이름, 소수자리 in [
        ('최근10경기팀타율', '팀타율', 4),
        ('기온', '기온', 2)
    ]:
        평균 = X_train[변수].mean()
        편차 = 표준편차[변수]
        st.write(f"**{이름}**: {평균:.{소수자리}f} → {평균 + 편차:.{소수자리}f} (1표준편차 증가)")
        st.write(f"{coef[변수]:,.2f} × {편차:.6f} ≈ {영향력[변수]:+,.1f}명")
    st.caption('영향력 그래프는 변화량의 절댓값을 표시합니다. 이는 선형회귀 모델의 조건부 예측 변화량이며 인과효과가 아닙니다.')

# 3. 영향력이 작게 나타난 변수와 관중수의 관계
elif 단계 == '3. 변수별 산점도 분석':
    st.header('3. 변수별 관중수 관계 분석')
    st.write('선형 영향력이 작게 나타난 팀타율, 평균자책점, 기온을 산점도로 비교합니다.')

    # 팀타율 산점도
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(train['최근10경기팀타율'], y_train, alpha=0.4)
    ax.set_title('팀타율과 관중수')
    st.pyplot(fig)
    plt.close(fig)

    # 평균자책점 산점도
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(train['최근10경기평균자책점'], y_train, alpha=0.4)
    ax.set_title('평균자책점과 관중수')
    st.pyplot(fig)
    plt.close(fig)
    st.caption('팀타율과 평균자책점은 기온처럼 뚜렷한 포물선 관계가 보이지 않습니다. 산점도는 다른 변수의 영향까지 포함하므로 선형 영향력과 단순 상관관계는 다를 수 있습니다.')

    st.subheader('기온과 관중수의 관계')
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(train['기온'], y_train, alpha=0.4)
    ax.set_xlabel('기온(℃)')
    ax.set_ylabel('관중수(명)')
    ax.set_ylim(y_train.min() - 500, y_train.max() + 500)
    st.pyplot(fig)
    plt.close(fig)

    # 기온을 1도 단위로 묶어 전체 추세 확인
    평균 = train.groupby(train['기온'].round())['관중수'].mean()
    st.line_chart(평균.rename('평균 관중수'))
    st.write('기온은 선형 영향력이 작지만, 기온별 평균 관중수에서는 비선형 관계가 나타납니다. 이를 반영하기 위해 기온제곱을 추가합니다.')

    # 기온제곱이 2차함수를 만드는 과정 설명
    st.divider()
    st.subheader('왜 기온제곱을 추가할까요?')
    st.write('기존 모델에서는 기온에 대한 항이 하나뿐이므로, 다른 변수를 고정하면 직선으로 나타납니다.')
    st.latex(r'\hat{y}_{\text{선형}} = bx + C')
    st.write('하지만 기온제곱을 새로운 독립변수로 추가하면, 기온과 기온제곱에 각각 회귀계수가 붙습니다.')
    st.code("df['기온제곱'] = df['기온'] ** 2", language='python')

    # 작은 표로 기온이 바뀔 때 제곱값이 어떻게 바뀌는지 보여줌
    st.latex(r'\hat{y}_{\text{다항}} = ax^2 + bx + C')
    st.write('여기서 x는 기온, a는 기온제곱의 회귀계수, b는 기온의 회귀계수입니다. '
             'C는 인기도·주말여부·팀타율·평균자책점·구장수용인원과 절편을 합친 값입니다.')
    st.write('모델은 모든 변수를 동시에 학습하지만, 그래프를 그릴 때 다른 변수의 값을 고정하면 C가 상수가 되어 기온에 대한 2차함수를 볼 수 있습니다.')

    # 모델이 학습한 실제 계수로 포물선과 꼭짓점 계산
    다항계수 = pd.Series(model2.coef_, index=다항특성)
    a = 다항계수['기온제곱']
    b = 다항계수['기온']
    st.subheader('실제로 학습된 2차함수')
    st.write(f'기온제곱의 회귀계수 a = {a:,.2f}, 기온의 회귀계수 b = {b:,.2f}')
    st.latex(rf'\hat{{y}} = {a:.2f}x^2 + {b:.2f}x + C')

    if a != 0:
        꼭짓점기온 = -b / (2 * a)
        if a < 0:
            st.latex(r'x_{\text{최대}} = -\frac{b}{2a}')
            st.metric('모델이 계산한 관중수 최대 기온', f'{꼭짓점기온:.2f}℃')
            st.write('기온제곱의 계수가 음수이므로 아래쪽으로 열리는(∩ 모양) 포물선이 만들어지고, 꼭짓점에서 예측 관중수가 가장 많아집니다.')
        else:
            st.latex(r'x_{\text{최소}} = -\frac{b}{2a}')
            st.metric('포물선의 최솟값에 해당하는 기온', f'{꼭짓점기온:.2f}℃')
            st.write('기온제곱의 계수가 양수이므로 위쪽으로 열리는(∪ 모양) 포물선이 만들어집니다.')

    # 다른 조건을 동일하게 고정해서 선형회귀와 다항회귀의 기온 반응 비교
    고정조건 = X_train.mean().to_dict()
    고정조건['주말여부'] = 0  # 주말여부는 0 또는 1이므로 평일로 고정
    기온범위 = [15 + 0.25 * i for i in range(65)]
    그래프입력 = pd.DataFrame([고정조건.copy() for _ in 기온범위])
    그래프입력['기온'] = 기온범위
    그래프입력['기온제곱'] = 그래프입력['기온'] ** 2

    st.image("기온에 따른 예측 관중수 변화 그래프.png")
    st.caption('인기도·팀타율·평균자책점·구장수용인원은 학습 데이터 평균으로, 주말여부는 평일(0)로 고정했습니다. '
               '이는 다른 변수의 영향을 고정한 모델의 예측 곡선이며, 위의 기온별 실제 평균 관중수 그래프와는 다릅니다.')
    st.write('이처럼 기온제곱을 추가해 포물선 관계를 학습한 뒤, 다음 4번 화면에서 실제 예측 오차가 줄었는지 확인합니다.')

# 4. 다항회귀 적용 전후: 기온별 잔차 비교
elif 단계 == '4. 다항회귀 적용':
    st.header('4. 다항회귀로 기온에 따른 예측 오차 줄이기')
    st.write('기온과 기온제곱을 함께 사용하여 비선형 관계를 학습합니다.')
    st.code('df["기온제곱"] = df["기온"] ** 2')

    # 잔차 = 실제 관중수 - 예측 관중수
    잔차 = pd.DataFrame({
        '기온': test['기온'].round(),
        '기존 선형회귀': y_test - 예측1,
        '다항회귀': y_test - 예측2
    })
    기온별잔차 = 잔차.groupby('기온').mean()
    기온별잔차['오차 없음'] = 0
    st.subheader('기온별 평균 잔차 비교')
    st.line_chart(기온별잔차)
    st.caption('X축: 기온(℃) / Y축: 실제 관중수 - 예측 관중수(명), 1℃ 단위 평균')
    st.write('잔차가 양수이면 실제보다 적게 예측한 것이고, 음수이면 실제보다 많게 예측한 것입니다.')
    st.write('기존 모델은 낮거나 높은 기온에서 관중수를 과대예측하고 중간 기온에서는 과소예측합니다. 다항회귀 적용 후 이러한 오차 패턴이 줄어듭니다.')

    # 같은 테스트 데이터의 평균 절대 오차 비교
    오차1 = mean_absolute_error(y_test, 예측1)
    오차2 = mean_absolute_error(y_test, 예측2)
    감소율 = (오차1 - 오차2) / 오차1 * 100
    열1, 열2 = st.columns(2)
    열1.metric('기존 선형회귀 MAE', f'{오차1:,.2f}명')
    열2.metric('다항회귀 MAE', f'{오차2:,.2f}명')
    st.metric('다항회귀 적용 후 오차 감소율', f'{감소율:.1f}%')
    st.caption('평균 잔차는 양수와 음수가 상쇄될 수 있으므로, 전체 예측 오차는 MAE로 별도 비교합니다.')

# 5. 교호작용 적용 전후: 회귀계수로 주말 효과 계산
elif 단계 == '5. 교호작용 적용':
    st.header('5. 구장수용인원 × 주말여부')
    st.code('df["수용인원_주말"] = df["구장수용인원"] * df["주말여부"]')
    coef2 = pd.Series(model2.coef_, index=다항특성)
    coef3 = pd.Series(model3.coef_, index=최종특성)

    수용인원 = sorted(df['구장수용인원'].unique())
    비교 = pd.DataFrame({'구장수용인원': 수용인원})
    비교['적용 전'] = coef2['주말여부']
    비교['적용 후'] = (coef3['주말여부']
                     + coef3['수용인원_주말'] * 비교['구장수용인원'])
    비교 = 비교.set_index('구장수용인원')
    st.line_chart(비교)
    st.dataframe(비교.round(1))
    st.caption('Y축: 주말 예측 관중수 - 평일 예측 관중수(명)')
    st.write('교호작용을 추가하면 구장 크기에 따라 주말 관중 증가량이 달라집니다. 다른 조건은 두 예측에서 같으므로 차이를 계산할 때 상쇄됩니다.')

# 6. 최종 오차 비교
elif 단계 == '6. 최종 오차 비교':
    st.header('6. 특성공학 적용 전후 오차 비교')
    오차1 = mean_absolute_error(y_test, 예측1)
    오차2 = mean_absolute_error(y_test, 예측2)
    오차3 = mean_absolute_error(y_test, 예측3)
    비교 = pd.Series([오차1, 오차2, 오차3],
                    index=['1. 선형회귀', '2. 다항회귀', '3. 교호작용 추가'],
                    name='평균 절대 오차(MAE)')
    st.bar_chart(비교)
    st.dataframe(비교.round(2))
    감소율 = (오차1 - 오차3) / 오차1 * 100
    st.metric('전체 MAE 감소율', f'{감소율:.1f}%')
    st.write(f'다항회귀 적용 후: {(오차1 - 오차2) / 오차1 * 100:.1f}% 감소')
    st.write(f'교호작용 추가 후: {(오차2 - 오차3) / 오차2 * 100:.1f}% 감소 (직전 모델 대비)')
    st.caption('같은 테스트 데이터에서 평가했습니다. 모의 데이터에서의 성능이며 실제 경기 예측 성능을 의미하지 않습니다.')


# 7. 최종 관중수 예측 AI
elif 단계 == '7. 최종 관중수 예측 AI':
    st.header('7. ⚾ 야구 관중수 예측 AI')

    # 앞에서 학습한 최종 모델의 평가 결과를 그대로 사용
    오차1 = mean_absolute_error(y_test, 예측1)
    오차3 = mean_absolute_error(y_test, 예측3)
    감소율 = (오차1 - 오차3) / 오차1 * 100

    st.subheader('📊 최종 데이터 분석 결과')
    열1, 열2 = st.columns(2)
    열1.metric('전체 경기 수', f'{len(df):,}경기')
    열2.metric('평균 관중수', f'{df["관중수"].mean():,.0f}명')
    열3, 열4 = st.columns(2)
    열3.metric('특성공학 적용 전 MAE', f'{오차1:,.2f}명')
    열4.metric('최종 모델 MAE', f'{오차3:,.2f}명')
    st.metric('전체 오차 감소율', f'{감소율:.1f}%')
    st.caption('6번과 동일한 학습·테스트 분할 및 최종 모델로 계산한 결과입니다.')

    # 경기 정보 입력
    st.subheader('🔮 관중수 예측')
    home = st.selectbox('홈팀', df['홈팀'].unique())
    away = st.selectbox('원정팀', [팀 for 팀 in df['원정팀'].unique() if 팀 != home])

    # 홈팀에 따라 기존 데이터에 기록된 인기도와 구장수용인원 자동 선택
    홈팀정보 = df[df['홈팀'] == home].iloc[0]
    popularity = int(홈팀정보['인기도'])
    capacity = int(홈팀정보['구장수용인원'])
    st.write(f'인기도: {popularity} / 구장수용인원: {capacity:,}명')

    temperature = st.number_input('기온(℃)', min_value=15.0,
                                  max_value=31.0, value=23.0, step=0.1)
    weekend = st.selectbox('주말 여부', ['평일', '주말'])
    batting = st.number_input('최근 10경기 팀타율', min_value=0.239,
                              max_value=0.287, value=0.260, step=0.001, format='%.3f')
    era = st.number_input('최근 10경기 평균자책점', min_value=3.34,
                         max_value=5.0, value=4.20, step=0.01, format='%.2f')
    st.caption('입력 범위는 모의 데이터의 최솟값과 최댓값을 기준으로 제한했습니다. 원정팀은 경기 정보 표시용이며 최종 모델의 독립변수는 아닙니다.')

    if st.button('관중수 예측하기'):
        new_data = pd.DataFrame([{
            '인기도': popularity,
            '기온': temperature,
            '주말여부': 1 if weekend == '주말' else 0,
            '최근10경기팀타율': batting,
            '최근10경기평균자책점': era,
            '구장수용인원': capacity
        }])

        # 학습할 때와 동일한 특성공학 적용
        new_data['기온제곱'] = new_data['기온'] ** 2
        new_data['수용인원_주말'] = new_data['구장수용인원'] * new_data['주말여부']

        # 6번에서 평가한 최종 모델을 그대로 사용
        result = model3.predict(new_data[최종특성])[0]
        # 실제 구장에 입장 가능한 인원 범위로 표시
        표시관중수 = min(capacity, max(0, result))
        st.success(f'{home} vs {away} | 예상 관중수: {표시관중수:,.0f}명')
        if 표시관중수 != result:
            st.caption('모델의 원래 예측값을 0명~구장수용인원 범위로 제한해 표시했습니다.')

    st.subheader('📋 전체 경기 데이터')
    st.dataframe(df.drop(columns=['기온제곱', '수용인원_주말']))
    st.caption('교육용 모의 데이터로 학습한 결과이며, 실제 경기 관중수 예측 성능을 의미하지 않습니다.')
