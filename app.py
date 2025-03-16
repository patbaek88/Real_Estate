import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import streamlit as st
from dateutil.relativedelta import relativedelta

# 데이터 로드 및 전처리
data = pd.read_csv("real_estate8.csv")
data['time'] = pd.to_datetime(data['time'])

data.set_index(['time'], inplace=True)

# 경제지표 관련 열 추출
economic_columns = ["exchange_rate", "kr_interest_rate", "us_interest_rate", "oil_price", "kr_price_index"]
economic_data = data[economic_columns]

# 데이터 정규화
scaler = RobustScaler()
economic_data_norm1 = scaler.fit_transform(economic_data)
economic_data_norm = pd.DataFrame(economic_data_norm1, columns=economic_columns, index=economic_data.index)

#미래경제데이터 생성 (VAR모델)
maxlags = st.number_input('Max Lag 설정 (3 = 1년6개월)', 1,30, value=3)

# VAR 모델 학습
model = VAR(economic_data_norm)
lag_selection = model.select_order(maxlags=maxlags)
optimal_lag = lag_selection.selected_orders['aic']
st.write("Optimal Lag:", optimal_lag)

results = model.fit(optimal_lag)  # 과거 12개월의 데이터를 사용하여 학습

st.write("AIC:", results.aic)
st.write("BIC:", results.bic)

# 미래 36개월(3년) 예측
forecast_steps = 6
forecast = results.forecast(economic_data_norm.values[-maxlags:], forecast_steps)  # 과거 모든 데이터에서 예측


last_time = data.index[-1]
next_point = last_time + relativedelta(months=6)
next_point = next_point.strftime('%Y-%m-%d')


# 예측된 데이터 프레임으로 변환
future_months = pd.date_range(start=next_point, periods=forecast_steps, freq='6MS')
predicted_economic_data = pd.DataFrame(forecast, columns=economic_columns, index=future_months)

predicted_economic_data_de = scaler.inverse_transform(predicted_economic_data)
predicted_economic_data_denorm =pd.DataFrame(predicted_economic_data_de, columns=economic_columns, index=future_months)

st.write("Predicted Economic Data for Next 3 Years:")
st.write(predicted_economic_data_denorm)
st.link_button("exchange_rate Reference", url = "http://www.smbs.biz/ExRate/MonAvgStdExRate.jsp")
st.link_button("kr_interest_rate Reference", url = "https://www.bok.or.kr/portal/singl/baseRate/list.do?dataSeCd=01&menuNo=200643")
st.link_button("us_interest_rate Reference", url = "https://ko.tradingeconomics.com/united-states/interest-rate")
st.link_button("oil_price Reference", url = "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=RBRTE&f=M")
st.link_button("kr_price_index Reference", url = "https://tradingeconomics.com/south-korea/consumer-price-index-cpi")





predicted_economic_data_jan_norm = predicted_economic_data[predicted_economic_data.index.month == 1]
predicted_economic_data_jan = predicted_economic_data_denorm[predicted_economic_data_denorm.index.month == 1]

data_df_apt2 = data.dropna(subset=["apt2_price"])
data_df_apt2 = data_df_apt2.drop([ "apt1_price", "apt1_price_rate","apt2_price_rate","my_land_price", "my_land_price_rate"], axis=1) 
data_df_myp = data.dropna(subset=["my_land_price"])
data_df_my = data.dropna(subset=["my_land_price_rate"])
data_df_my = data_df_my.drop([ "apt1_price", "apt1_price_rate","apt2_price", "apt2_price_rate","my_land_price"], axis=1) 
                          
                          
scaler2 = RobustScaler()
data_df_apt2_norm1 = scaler2.fit_transform(data_df_apt2)
data_df_apt2_norm = pd.DataFrame(data_df_apt2_norm1, columns=[["exchange_rate", "kr_interest_rate", "us_interest_rate", "oil_price", "kr_price_index","apt2_price"]], index=data_df_apt2.index)


scaler3 = RobustScaler()
data_df_my_norm1 = scaler3.fit_transform(data_df_my)
data_df_my_norm = pd.DataFrame(data_df_my_norm1, columns=[["exchange_rate", "kr_interest_rate", "us_interest_rate", "oil_price", "kr_price_index", "my_land_price_rate"]], index=data_df_my.index)


# 특성과 타겟 분리
X = data_df_apt2_norm[["exchange_rate", "kr_interest_rate", "us_interest_rate", "oil_price", "kr_price_index"]]
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
 #   'Ridge': Ridge(),
    'SVR': SVR(),
 #   'XGBoost': XGBRegressor(objective='reg:squarederror'),
 #   'LightGBM': LGBMRegressor()
}

best_model2 = None
best_mse2 = float('inf')

for name, model in models2.items():
    # 파이프라인 구성
    pipeline2 = Pipeline([
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
#    'Ridge': Ridge(),
    'SVR': SVR(),
#    'XGBoost': XGBRegressor(objective='reg:squarederror'),
#    'LightGBM': LGBMRegressor()
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


# 초기 start_date 설정
start_date = pd.to_datetime("2017-01-01")

# 6개월씩 더해가며 반복할 때 2025-01-01까지 반복
end_date = last_time

# 예측 결과를 저장할 빈 데이터프레임 생성
predicted_apt2_price_denorm2_df = pd.DataFrame()

# 반복문을 사용하여 start_date가 2025-01-01이 될 때까지 6개월씩 더함
while start_date <= end_date:
    
    # 데이터 정규화
    scaler_t = RobustScaler()
    
    # start_date 기준으로 데이터를 트리밍
    economic_data_trimmed = economic_data[economic_data.index < start_date]

    
    economic_data_trimmed_norm1 = scaler_t.fit_transform(economic_data_trimmed)
    economic_data_trimmed_norm = pd.DataFrame(economic_data_trimmed_norm1, columns=economic_columns, index=economic_data_trimmed.index)

    # VAR 모델 학습
    model_t = VAR(economic_data_trimmed_norm)
    lag_selection_t = model_t.select_order(maxlags=maxlags)
    optimal_lag_t = lag_selection_t.selected_orders['aic']

    results_t = model_t.fit(optimal_lag_t)  


    # 미래 6개월 후 예측
    forecast_steps_t = 1
    forecast_t = results_t.forecast(economic_data_trimmed_norm.values[-maxlags:], forecast_steps_t)  

    # 예측된 데이터 프레임으로 변환
    future_months_t = pd.date_range(start=start_date, periods=forecast_steps_t, freq='6MS')
    predicted_economic_data_t = pd.DataFrame(forecast_t, columns=economic_columns, index=future_months_t)

    predicted_economic_data_t_de = scaler_t.inverse_transform(predicted_economic_data_t)
    predicted_economic_data_t_denorm = pd.DataFrame(predicted_economic_data_t_de, columns=economic_columns, index=future_months_t)

    # scaler2_t 설정 및 데이터 트리밍
    scaler2_t = RobustScaler()
    data_df_apt2_t = data_df_apt2[data_df_apt2.index < start_date]

    data_df_apt2_norm1_t = scaler2_t.fit_transform(data_df_apt2_t)
    data_df_apt2_norm_t = pd.DataFrame(data_df_apt2_norm1_t, columns=["exchange_rate", "kr_interest_rate", "us_interest_rate", "oil_price", "kr_price_index", "apt2_price"], index=data_df_apt2_t.index)

    # 특성과 타겟 분리
    X_t = data_df_apt2_norm_t[["exchange_rate", "kr_interest_rate", "us_interest_rate", "oil_price", "kr_price_index"]]
    y2_t = data_df_apt2_norm_t['apt2_price']

    X_trimmed = X_t[X_t.index < start_date]
    y2_trimmed = y2_t[y2_t.index < start_date]

    # Train/Test 분할
    X_trimmed_train, X_trimmed_test, y2_trimmed_train, y2_trimmed_test = train_test_split(X_trimmed, y2_trimmed, test_size=0.2, random_state=20211227)

    # 머신러닝 모델 비교
    models2_t = {
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'SVR': SVR(),
    }

    best_model2_t = None
    best_mse2_t = float('inf')

    for name, model in models2_t.items():
        # 파이프라인 구성
        pipeline2_t = Pipeline([
            ('model', model)
        ])

        # 모델 학습
        pipeline2_t.fit(X_trimmed_train, y2_trimmed_train)
           
        # 예측
        predictions2_t = pipeline2_t.predict(X_trimmed_test)
        
        # 모델 평가
        mse2_t = mean_squared_error(y2_trimmed_test, predictions2_t)
            
        # 최적의 모델 선택
        if mse2_t < best_mse2_t:
            best_mse2_t = mse2_t
            best_model2_t = pipeline2_t

    predicted_apt2_price_norm_t = best_model2_t.predict(predicted_economic_data_t)
    predicted_apt2_price_norm_df_t = pd.DataFrame(predicted_apt2_price_norm_t, index=predicted_economic_data_t.index)
    pred_apt2_df_t = pd.concat([predicted_economic_data_t, predicted_apt2_price_norm_df_t], axis=1)

    predicted_apt2_price_de_t = scaler2_t.inverse_transform(pred_apt2_df_t)
    predicted_apt2_price_denorm_t = pd.DataFrame(predicted_apt2_price_de_t, index=predicted_economic_data_t.index)
    predicted_apt2_price_denorm2_t = predicted_apt2_price_denorm_t.drop(columns=[0, 1, 2, 3, 4])

    # 예측 결과를 데이터프레임에 추가
    predicted_apt2_price_denorm2_df = pd.concat([predicted_apt2_price_denorm2_df, predicted_apt2_price_denorm2_t])


    # 6개월씩 더하기
    start_date += relativedelta(months=6)

# 최종 예측된 결과 데이터프레임 출력
st.write(predicted_apt2_price_denorm2_df)





def calculate_future_price_my (initial_price, change_rates):
    future_prices = []
    last_price = initial_price
    
    for rate in change_rates:
        new_price = last_price*(1+rate)
        last_price = new_price
        future_prices.append(new_price)
        
    return future_prices


# 예측 결과를 저장할 빈 데이터프레임 생성
predicted_myland_price_denorm2_df = pd.DataFrame()


# 초기 start_date_m 설정
start_date_m = pd.to_datetime("2017-01-01")

# 반복문을 사용하여 start_date가 2025-01-01이 될 때까지 12개월씩 더함
while start_date_m <= end_date:
    
    # 데이터 정규화
    scaler_m = RobustScaler()
    
    # start_date 기준으로 데이터를 트리밍
    economic_data_trimmed_m = economic_data[economic_data.index < start_date_m]

    
    economic_data_trimmed_norm1_m = scaler_m.fit_transform(economic_data_trimmed_m)
    economic_data_trimmed_norm_m = pd.DataFrame(economic_data_trimmed_norm1_m, columns=economic_columns, index=economic_data_trimmed_m.index)

    # VAR 모델 학습
    model_m = VAR(economic_data_trimmed_norm_m)
    lag_selection_m = model_t.select_order(maxlags=maxlags)
    optimal_lag_m = lag_selection_m.selected_orders['aic']

    results_m = model_t.fit(optimal_lag_m)  


    # 미래 12개월 후 예측
    forecast_steps_m = 1
    forecast_m = results_m.forecast(economic_data_trimmed_norm_m.values[-maxlags:], forecast_steps_m)  

    # 예측된 데이터 프레임으로 변환
    future_months_m = pd.date_range(start=start_date_m, periods=forecast_steps_m, freq='12MS')
    predicted_economic_data_m = pd.DataFrame(forecast_m, columns=economic_columns, index=future_months_m)

    predicted_economic_data_m_de = scaler_m.inverse_transform(predicted_economic_data_m)
    predicted_economic_data_m_denorm = pd.DataFrame(predicted_economic_data_m_de, columns=economic_columns, index=future_months_m)

    predicted_economic_data_m_jan_norm = predicted_economic_data_m[predicted_economic_data_m.index.month == 1]
    predicted_economic_data_m_jan = predicted_economic_data_m_denorm[predicted_economic_data_m_denorm.index.month == 1]
  
    # scaler2_m 설정 및 데이터 트리밍
    scaler2_m = RobustScaler()
    data_df_my_m = data_df_my[data_df_my.index < start_date_m]
    data_df_myp_m = data_df_myp[data_df_myp.index < start_date_m]

    data_df_myland_norm1_m = scaler2_m.fit_transform(data_df_my_m)
    data_df_myland_norm_m = pd.DataFrame(data_df_myland_norm1_m, columns=["exchange_rate", "kr_interest_rate", "us_interest_rate", "oil_price", "kr_price_index", "my_land_price"], index=data_df_my_m.index)

    # 특성과 타겟 분리
    X_m = data_df_myland_norm_m[["exchange_rate", "kr_interest_rate", "us_interest_rate", "oil_price", "kr_price_index"]]
    y2_m = data_df_myland_norm_m['my_land_price']

    X_trimmed_m = X_m[X_m.index < start_date_m]
    y2_trimmed_m = y2_m[y2_m.index < start_date_m]

    # Train/Test 분할
    X_trimmed_m_train, X_trimmed_m_test, y2_trimmed_m_train, y2_trimmed_m_test = train_test_split(X_trimmed_m, y2_trimmed_m, test_size=0.2, random_state=20211227)

    # 머신러닝 모델 비교
    models2_m = {
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'SVR': SVR(),
    }

    best_model2_m = None
    best_mse2_m = float('inf')

    for name, model in models2_m.items():
        # 파이프라인 구성
        pipeline2_m = Pipeline([
            ('model', model)
        ])

        # 모델 학습
        pipeline2_m.fit(X_trimmed_m_train, y2_trimmed_m_train)
           
        # 예측
        predictions2_m = pipeline2_m.predict(X_trimmed_m_test)
        
        # 모델 평가
        mse2_m = mean_squared_error(y2_trimmed_m_test, predictions2_m)
            
        # 최적의 모델 선택
        if mse2_m < best_mse2_m:
            best_mse2_m = mse2_m
            best_model2_m = pipeline2_m

    predicted_myland_price_norm_m = best_model2_m.predict(predicted_economic_data_m)
    predicted_myland_price_norm_df_m = pd.DataFrame(predicted_myland_price_norm_m, index=predicted_economic_data_m.index)
    pred_myland_df_m = pd.concat([predicted_economic_data_m, predicted_myland_price_norm_df_m], axis=1)

    predicted_myland_price_rate_de_m = scaler2_m.inverse_transform(pred_myland_df_m)
    predicted_myland_price_rate_denorm_m = pd.DataFrame(predicted_myland_price_rate_de_m, index=predicted_economic_data_m.index)
    predicted_myland_price_rate_denorm2_m = predicted_myland_price_rate_denorm_m.drop(columns=[0, 1, 2, 3, 4])


    current_land_price_m = data_df_myp_m['my_land_price'].iloc[-1]
    predicted_my_land_price_m = calculate_future_price_my(current_land_price_m, predicted_myland_price_rate_denorm2_m.values)
    predicted_my_land_price_df_m = pd.DataFrame(predicted_my_land_price_m, index = predicted_economic_data_m_jan.index)
    predicted_my_land_price2_m = predicted_my_land_price_df_m*69*2.2/100000000

  
    # 예측 결과를 데이터프레임에 추가
    predicted_myland_price_denorm2_df = pd.concat([predicted_myland_price_denorm2_df, predicted_my_land_price2_m])


    # 12개월씩 더하기
    start_date_m += relativedelta(months=12)

# 최종 예측된 결과 데이터프레임 출력
st.write(predicted_myland_price_denorm2_df)




predicted_apt2_price_norm = best_model2.predict(predicted_economic_data)
predicted_apt2_price_norm_df = pd.DataFrame(predicted_apt2_price_norm, index= predicted_economic_data.index)
pred_apt2_df = pd.concat([predicted_economic_data,predicted_apt2_price_norm_df], axis =1)

predicted_myland_price_rate_norm = best_model_my.predict(predicted_economic_data_jan_norm)
predicted_myland_price_rate_norm_df = pd.DataFrame(predicted_myland_price_rate_norm, index= predicted_economic_data_jan_norm.index)
pred_myland_df = pd.concat([predicted_economic_data_jan_norm, predicted_myland_price_rate_norm_df], axis =1)


predicted_apt2_price_de = scaler2.inverse_transform(pred_apt2_df)
predicted_apt2_price_denorm =pd.DataFrame(predicted_apt2_price_de, index=predicted_economic_data.index)
predicted_apt2_price_denorm2 = predicted_apt2_price_denorm.drop(columns =[0,1,2,3,4])

st.write(predicted_apt2_price_denorm2)

predicted_myland_price_rate_de = scaler3.inverse_transform(pred_myland_df)
predicted_myland_price_rate_denorm =pd.DataFrame(predicted_myland_price_rate_de, index=predicted_economic_data_jan.index)
predicted_myland_price_rate_denorm2 = predicted_myland_price_rate_denorm.drop(columns =[0,1,2,3,4])




current_land_price = data_df_myp['my_land_price'].iloc[-1]
predicted_my_land_price = calculate_future_price_my(current_land_price, predicted_myland_price_rate_denorm2.values)
predicted_my_land_price_df = pd.DataFrame(predicted_my_land_price, index = predicted_economic_data_jan.index)
predicted_my_land_price2 = predicted_my_land_price_df*69*2.2/100000000


# 그래프 시각화 (all)
graph = plt.figure(figsize=(10, 6))

# 실제 시장가격 그래프에 표시
plt.plot(data_df_my.index, data_df_myp["my_land_price"]*69*2.2/100000000, label='Actual my_land_price', marker='o', color = "blue")
plt.plot(data_df_apt2.index, data_df_apt2['apt2_price'], label='Actual apt2_price', marker='o', color = "orange")

# 향후 36개월 동안 예측 시장가격 표시
plt.scatter(predicted_my_land_price_df.index, predicted_my_land_price_df*69*2.5/100000000, label='Future_my_land_price', marker='^', color = "blue")
plt.scatter(predicted_apt2_price_denorm2.index, predicted_apt2_price_denorm2 , label='Predicted_price_apt2', marker='x', color = "orange")
plt.scatter(predicted_apt2_price_denorm2_df.index, predicted_apt2_price_denorm2_df, label='Predicted_price_apt2', marker='x', color = "orange")


plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual and Predicted Prices')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
#plt.show()

#st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(graph)

st.write("APT2 model: ", best_model2)
st.write("My land model: ", best_model_my)


