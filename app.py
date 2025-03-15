import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 데이터 불러오기 (예제 CSV 파일 가정)
data = pd.read_csv("real_estate7.csv", parse_dates=["date"])
data.set_index("date", inplace=True)

# 예측할 변수
targets = ["apt2_price", "my_land_price"]

# 과거 예측 (백테스트)
def backtest(start_date):
    train = data.loc[:start_date].copy()
    test = data.loc[start_date:].copy()
    predictions = []
    
    for date in test.index:
        # 독립 변수 선택 (경제 지표 등)
        X_train = train.drop(columns=targets)
        y_train = train[targets]
        
        # 모델 학습 및 예측
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        X_test = test.loc[date].drop(targets).values.reshape(1, -1)
        pred = model.predict(X_test)[0]
        
        # 결과 저장
        predictions.append([date, pred[0], pred[1]])
        
        # 실제값 추가하여 점진적 예측
        train.loc[date, targets] = test.loc[date, targets]
    
    # 결과 DataFrame
    pred_df = pd.DataFrame(predictions, columns=["date"] + targets)
    pred_df.set_index("date", inplace=True)
    
    return pred_df

# 미래 예측 (2025-07-01 ~ 2028-01-01)
def forecast_future(start_date, end_date):
    # VAR 모델로 경제 지표 예측
    var_model = VAR(data.drop(columns=targets))
    var_results = var_model.fit()
    future_econ = var_results.forecast(data.drop(columns=targets).values, steps=6*6)  # 6개월 단위
    future_dates = pd.date_range(start=start_date, end=end_date, freq='6M')
    future_econ_df = pd.DataFrame(future_econ, index=future_dates, columns=data.drop(columns=targets).columns)
    
    # 머신러닝 모델로 부동산 가격 예측
    X_train = data.drop(columns=targets)
    y_train = data[targets]
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    future_preds = model.predict(future_econ_df)
    
    # 결과 DataFrame
    future_df = pd.DataFrame(future_preds, index=future_dates, columns=targets)
    return future_df

# 실행 예제
start_date = "2020-01-01"
end_date = "2028-01-01"
backtest_results = backtest(start_date)
future_forecast = forecast_future("2025-07-01", end_date)

# 그래프 시각화
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["apt2_price"], label="Actual apt2_price", color='blue')
plt.plot(backtest_results.index, backtest_results["apt2_price"], label="Backtest apt2_price", linestyle='dashed', color='red')
plt.plot(future_forecast.index, future_forecast["apt2_price"], label="Forecast apt2_price", linestyle='dotted', color='green')
plt.legend()
plt.show()



