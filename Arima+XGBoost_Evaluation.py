import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split


# 데이터 로드
file_path = 'test_dataset/after_handling_nan.csv.csv'
data = pd.read_csv(file_path, encoding='cp949')
print(data.info())

# 데이터 확인
print(data.head())

# 시간대 컬럼을 리스트로 만듭니다
time_columns = ['06시이전', '06-07시간대', '07-08시간대', '08-09시간대', '09-10시간대', '10-11시간대', 
                '11-12시간대', '12-13시간대', '13-14시간대', '14-15시간대', '15-16시간대', '16-17시간대', 
                '17-18시간대', '18-19시간대', '19-20시간대', '20-21시간대', '21-22시간대', '22-23시간대', 
                '23-24시간대', '24시이후']

# 필요한 컬럼만 선택
data = data[['호선', '승하차구분'] + time_columns]

# 결측치 확인
print(data.isnull().sum())

# 결측치가 있을 경우 이를 처리 (예: 0으로 채우기)
data = data.fillna(0)

# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 각 호선과 상하행 구분에 따라 시계열 데이터를 나눕니다.
lines = data['호선'].unique()
directions = data['승하차구분'].unique()

# 승하차구분을 '상차'와 '하차'로 변경하는 함수 정의
def map_boarding_alighting(value):
    if value == 0:
        return '상차'
    else:
        return '하차'
    
# MAPE, MPE 계산 함수 정의
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_percentage_error(y_true, y_pred):
    return np.mean((y_true - y_pred) / y_true) * 100


# ARIMA + XGBoost 하이브리드 모델 적용 및 예측
for line in lines:
    for direction in directions:
        subset = data[(data['호선'] == line) & (data['승하차구분'] == direction)]
        if not subset.empty:
            first_row = subset.iloc[0]
            time_series = first_row[time_columns].values
            time_series = np.array(time_series, dtype=float)

            # ARIMA 모델 적용
            model_arima = ARIMA(time_series, order=(8, 0, 35))
            model_fit_arima = model_arima.fit()

            # ARIMA 모델을 통해 예측 및 잔차 계산
            arima_pred = model_fit_arima.predict(start=1, end=len(time_series)-1)
            residuals = time_series[1:] - arima_pred

            # XGBoost 모델 학습
            X = np.arange(len(residuals)).reshape(-1, 1)
            y = residuals
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
            model_xgb.fit(X_train, y_train)

            # 잔차 예측
            residuals_pred = model_xgb.predict(np.arange(len(time_series)-1).reshape(-1, 1))

            # 최종 예측값 계산 (ARIMA 예측 + 잔차 예측)
            final_pred = arima_pred + residuals_pred

            # 평가 지표 계산
            mse = mean_squared_error(time_series[1:], final_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(time_series[1:], final_pred)
            mape = mean_absolute_percentage_error(time_series[1:], final_pred)
            mpe = mean_percentage_error(time_series[1:], final_pred)
            

            # 결과 시각화
            plt.figure(figsize=(10, 6))
            plt.plot(time_columns[1:], time_series[1:], marker='o', label='실제 데이터')
            plt.plot(time_columns[1:], final_pred, marker='o', color='red', label='예측 데이터')
            plt.title(f'{line}호선 {map_boarding_alighting(direction)} 시간대별 탑승객 수 예측 (ARIMA + XGBoost)')
            plt.xlabel('시간대')
            plt.ylabel('탑승객 수')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.show()

            # 평가
            rmse = np.sqrt(mean_squared_error(time_series[1:], final_pred))
            print(f'{line}호선 {map_boarding_alighting(direction)} MSE: {mse}')
            print(f'{line}호선 {map_boarding_alighting(direction)} RMSE: {rmse}')
            print(f'{line}호선 {map_boarding_alighting(direction)} MAE: {mae}')
            print(f'{line}호선 {map_boarding_alighting(direction)} MAPE: {mape}')
            print(f'{line}호선 {map_boarding_alighting(direction)} MPE: {mpe}')