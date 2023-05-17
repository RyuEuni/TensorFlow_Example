import mysql.connector
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional

# MySQL 연결 설정
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0000",
    database="stock"
)

#==============csv파일을 읽어 DB 테이블에 저장하는 코드=================================================================
# # CSV 파일 경로
# csv_file = './data_5417_20230517.csv'

# # CSV 파일 읽기 및 데이터 삽입
# with open(csv_file, 'r') as file:
#     reader = csv.reader(file)
#     next(reader)  # 첫 줄은 헤더이므로 건너뜁니다.
#     cursor = conn.cursor()  # 커서 생성
#     for row in reader:
#         code = row[0]  # 종목코드
#         name = row[1]  # 종목 명
#         category = row[2]  # 시장 구분
#         close_price = row[3]  # 종가
#         open_price = row[4]  # 시가
#         high_price = row[5]  # 고가
#         low_price = row[6]  # 저가
#         volume = row[7]  # 거래량
#         value = row[8]  # 거래대금
#         cap = row[9]  # 시가총액
#         shares = row[10]  # 상장 주식 수

#         # 데이터 삽입 쿼리 실행
#         query = "INSERT INTO stocks (stock_code, name, category, closing_price, opening_price, highest_price, lowest_price, volume, value, market_cap, listed_shares) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
#         values = (code, name, category, close_price, open_price, high_price, low_price, volume, value, cap, shares)
#         cursor.execute(query, values)
    
#     conn.commit()  # 변경사항 저장
#     cursor.close()  # 커서 닫기
#======================================================================================================================

# SQL 쿼리 실행하여 데이터 가져오기
query = "SELECT closing_price, opening_price, highest_price, lowest_price, volume FROM stocks"
# 주식 데이터를 DataFrame으로 읽어와서 df에 저장
df = pd.read_sql_query(query, conn) 
# print("주식 데이터: ", df)

# 훈련시킬 데이터의 비율 및 크기 설정
total_rows = 2713
train_ratio = 0.8
train_size = int(total_rows * train_ratio)

train_data = df[:train_size]  # 훈련용 데이터
test_data = df[train_size:]  # 테스트용 데이터
# print("훈련 데이터: ", train_data)
# print("테스트 데이터: ", test_data)

timesteps = 1
features = 1

# 모델 구성
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(timesteps, features)))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1))

X_train = []
y_train = []

for i in range(timesteps, len(train_data)):
    X_train.append(train_data[i - timesteps:i])
    y_train.append(train_data.loc[i, 'closing_price'])

X_train = np.array(X_train)
y_train = np.array(y_train)

# 데이터 타입 변환
X_train = X_train.astype(float)
y_train = y_train.astype(float)

# y_train의 차원 값 변경. 1차원 -> 2차원
y_train = y_train.reshape(-1, 1)

# print("X_train 데이터 타입:", type(X_train))
# print("y_train 데이터 타입:", type(y_train))

# print("X_train 차원:", X_train.shape)
# print("y_train 차원:", y_train.shape)

# 모델 컴파일
model.compile(loss='mean_squared_error', optimizer='adam')

# 모델 학습
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 모델 예측
predictions = model.predict(X_train)

# 모델 평가
evaluation = model.evaluate(X_train, y_train)

print("모델 예측: ", predictions)

# 추가 작업 및 결과 분석
# ...

# 전체 코드 정리
# ...

