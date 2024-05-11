#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler
import tensorflow as tf
from keras.layers import Dropout, TimeDistributed
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR

# 데이터 로드 및 전처리
data = pd.read_csv("real_estate7.csv")
data['time'] = pd.to_datetime(data['time'])

data.set_index(['time'], inplace=True)

# 경제지표 관련 열 추출
economic_columns = ["exchange_rate", "kr_interest_rate", "us_interest_rate", "oil_price", "kr_price_index"]
economic_data = data[economic_columns]

# 데이터 정규화
#scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()
economic_data_norm1 = scaler.fit_transform(economic_data)
economic_data_norm = pd.DataFrame(economic_data_norm1, columns=economic_columns, index=economic_data.index)

#미래경제데이터 생성 (VAR모델)
#maxlags = 17
maxlags = 19

# VAR 모델 학습
model = VAR(economic_data_norm)
results = model.fit(maxlags=maxlags)  # 과거 12개월의 데이터를 사용하여 학습

# 미래 36개월(3년) 예측
forecast_steps = 6
forecast = results.forecast(economic_data_norm.values[-maxlags:], forecast_steps)  # 과거 모든 데이터에서 예측

# 예측된 데이터 프레임으로 변환
future_months = pd.date_range(start="2024-07-01", periods=forecast_steps, freq='6MS')
predicted_economic_data = pd.DataFrame(forecast, columns=economic_columns, index=future_months)

predicted_economic_data_de = scaler.inverse_transform(predicted_economic_data)
predicted_economic_data_denorm =pd.DataFrame(predicted_economic_data_de, columns=economic_columns, index=future_months)

print("Predicted Economic Data for Next 3 Years:")
predicted_economic_data_denorm


predicted_economic_data_jan_norm = predicted_economic_data[predicted_economic_data.index.month == 1]
predicted_economic_data_jan = predicted_economic_data_denorm[predicted_economic_data_denorm.index.month == 1]

#data_df_apt2 = data.dropna(subset=["apt2_price_rate"])
data_df_apt2 = data.dropna(subset=["apt2_price"])
data_df_apt2 = data_df_apt2.drop([ "apt1_price", "apt1_price_rate","apt2_price_rate","my_land_price", "my_land_price_rate"], axis=1) 
data_df_myp = data.dropna(subset=["my_land_price"])
data_df_my = data.dropna(subset=["my_land_price_rate"])
data_df_my = data_df_my.drop([ "apt1_price", "apt1_price_rate","apt2_price", "apt2_price_rate","my_land_price"], axis=1) 
                          
                          
#scaler2 = StandardScaler()
scaler2 = RobustScaler()
data_df_apt2_norm1 = scaler2.fit_transform(data_df_apt2)
data_df_apt2_norm = pd.DataFrame(data_df_apt2_norm1, columns=[["exchange_rate", "kr_interest_rate", "us_interest_rate", "oil_price", "kr_price_index","apt2_price"]], index=data_df_apt2.index)

#scaler3 = StandardScaler()
scaler3 = RobustScaler()
data_df_my_norm1 = scaler3.fit_transform(data_df_my)
data_df_my_norm = pd.DataFrame(data_df_my_norm1, columns=[["exchange_rate", "kr_interest_rate", "us_interest_rate", "oil_price", "kr_price_index", "my_land_price_rate"]], index=data_df_my.index)


# 특성과 타겟 분리
X = data_df_apt2_norm[["exchange_rate", "kr_interest_rate", "us_interest_rate", "oil_price", "kr_price_index"]]
#y2 = data_df_apt2['apt2_price_rate']
y2 = data_df_apt2_norm['apt2_price']

Xmy = data_df_my_norm[["exchange_rate", "kr_interest_rate", "us_interest_rate", "oil_price", "kr_price_index"]]
ymy = data_df_my_norm['my_land_price_rate']  #*69*2.5/100000000



# Train/Test 분할
X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state = 20211227)

# 학습할 때 사용된 특성명 저장
feature_names = X.columns.tolist()

# 머신러닝 모델 비교
models2 = {
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Ridge': Ridge(),
    'SVR': SVR(),
    'XGBoost': XGBRegressor(objective='reg:squarederror'),
    #'LightGBM': LGBMRegressor()
}

best_model2 = None
best_mse2 = float('inf')

for name, model in models2.items():
    # 파이프라인 구성
    pipeline2 = Pipeline([
        #('imputer', SimpleImputer(strategy='mean')),
        #('scaler', StandardScaler()),
        ('model', model)
    ])

    
    # 모델 학습
    pipeline2.fit(X_train, y2_train)
       
    # 예측
    predictions2 = pipeline2.predict(X_test)
    
    # 모델 평가
    mse2 = mean_squared_error(y2_test, predictions2)
        
    # 최적의 모델 선택
    if mse2 < best_mse2:
        best_mse2 = mse2
        best_model2 = pipeline2    

    
# Train/Test 분할
Xmy_train, Xmy_test, ymy_train, ymy_test = train_test_split(Xmy, ymy, test_size=0.2, random_state = 881030)

# 학습할 때 사용된 특성명 저장
feature_names_my = Xmy.columns.tolist()

# 머신러닝 모델 비교
models_my = {
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Ridge': Ridge(),
    'SVR': SVR(),
    'XGBoost': XGBRegressor(objective='reg:squarederror'),
    #'LightGBM': LGBMRegressor()
}

best_model_my = None
best_mse_my = float('inf')

for name, model in models_my.items():
    # 파이프라인 구성
    pipeline_my = Pipeline([
        #('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # 모델 학습
    pipeline_my.fit(Xmy_train, ymy_train)
    
    # 예측
    predictions_my = pipeline_my.predict(Xmy_test)
    
    # 모델 평가
    mse_my = mean_squared_error(ymy_test, predictions_my)

    
    # 최적의 모델 선택
    if mse_my < best_mse_my:
        best_mse_my = mse_my
        best_model_my = pipeline_my

        
# 향후 시점 가격변동률 예측
#def predict_future_price(model, X):
 #   future_features = pd.DataFrame(columns=X.columns)
        
    # 예측할 때 사용된 특성명만 선택
  #  future_features = future_features[feature_names]
    
   # future_predictions = model.predict(future_features)
    #return future_predictions

#predicted_apt2_price_rate = best_model2.predict(predicted_economic_data)
#predicted_apt2_price_rate_df = pd.DataFrame(predicted_apt2_price_rate, index= predicted_economic_data.index)
predicted_apt2_price_norm = best_model2.predict(predicted_economic_data)
#predicted_apt2_price = predict_future_price(best_model2, predicted_economic_data)
predicted_apt2_price_norm_df = pd.DataFrame(predicted_apt2_price_norm, index= predicted_economic_data.index)
pred_apt2_df = pd.concat([predicted_economic_data,predicted_apt2_price_norm_df], axis =1)

predicted_myland_price_rate_norm = best_model_my.predict(predicted_economic_data_jan_norm)
#predicted_myland_price = predict_future_price(best_model_my, predicted_economic_data_jan)
predicted_myland_price_rate_norm_df = pd.DataFrame(predicted_myland_price_rate_norm, index= predicted_economic_data_jan_norm.index)
pred_myland_df = pd.concat([predicted_economic_data_jan_norm, predicted_myland_price_rate_norm_df], axis =1)


predicted_apt2_price_de = scaler2.inverse_transform(pred_apt2_df)
predicted_apt2_price_denorm =pd.DataFrame(predicted_apt2_price_de, index=predicted_economic_data.index)
predicted_apt2_price_denorm2 = predicted_apt2_price_denorm.drop(columns =[0,1,2,3,4])

predicted_myland_price_rate_de = scaler3.inverse_transform(pred_myland_df)
predicted_myland_price_rate_denorm =pd.DataFrame(predicted_myland_price_rate_de, index=predicted_economic_data_jan.index)
predicted_myland_price_rate_denorm2 = predicted_myland_price_rate_denorm.drop(columns =[0,1,2,3,4])

def calculate_future_price_my (initial_price, change_rates):
    future_prices = []
    last_price = initial_price
    
    for rate in change_rates:
        new_price = last_price*(1+rate)
        last_price = new_price
        future_prices.append(new_price)
        
    return future_prices


#current_apt2_price = data_df_apt2['apt2_price'].iloc[-1]
#predicted_apt2_price = calculate_future_price_my(current_apt2_price, predicted_apt2_price_rate)
#predicted_apt2_price_df = pd.DataFrame(predicted_apt2_price, index = predicted_apt2_price_rate_df.index)

current_land_price = data_df_myp['my_land_price'].iloc[-1]
predicted_my_land_price = calculate_future_price_my(current_land_price, predicted_myland_price_rate_denorm2.values)
predicted_my_land_price_df = pd.DataFrame(predicted_my_land_price, index = predicted_economic_data_jan.index)
predicted_my_land_price2 = predicted_my_land_price_df*69*2.2/100000000


# 그래프 시각화 (all)
plt.figure(figsize=(10, 6))

# 실제 시장가격 그래프에 표시
plt.plot(data_df_my.index, data_df_myp["my_land_price"]*69*2.2/100000000, label='Actual my_land_price', marker='o', color = "blue")
plt.plot(data_df_apt2.index, data_df_apt2['apt2_price'], label='Actual apt2_price', marker='o', color = "orange")

# 향후 24개월 동안 예측 시장가격 표시
plt.scatter(predicted_my_land_price_df.index, predicted_my_land_price_df*69*2.2/100000000, label='Future_my_land_price', marker='^', color = "blue")
plt.scatter(predicted_apt2_price_denorm2.index, predicted_apt2_price_denorm2 , label='Future_price_apt2', marker='^', color = "orange")


plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual and Predicted Prices')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

