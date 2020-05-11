#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 18:13:19 2020

@author: ziyunchebn
"""

import ccxt
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import goog_trend
import prediction_model
from Binance import Binance
from TradingModel import TradingModel
import sys
import json
import schedule 
import time

binance_keys = {
	'apiKey': "Wiv78wwuMopBnvRqf0g76l8PGNHEZoC6REqa1y5boJJ6tX7Oz6KXfuxSRW9qJBcz",
	'secret': "XEPd9axQNEqdkpyr5txuRukYtWANoc5gSsqfBrL5f8TMsaGrTttCbxPooEbe57U7"
}


# collect the candlestick data from Binance



def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['price'].rolling(window=7).mean()
    dataset['ma21'] = dataset['price'].rolling(window=21).mean()
    
    # Create MACD
    #dataset['26ema'] = pd.ewma(dataset['price'], span=26)
    #dataset['12ema'] = pd.ewma(dataset['price'], span=12)
    dataset['26ema'] = dataset['price'].ewm(span=26).mean()
    dataset['12ema'] = dataset['price'].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    # Create Bollinger Bands
    dataset['20sd'] = dataset['price'].rolling(window=7).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Create Exponential moving average
    dataset['ema'] = dataset['price'].ewm(com=0.5).mean()
    
    # Create Momentum
    dataset['momentum_1'] = dataset['price'].shift(1)
    dataset['momentum_2'] = dataset['price'].shift(2)
    dataset['momentum_3'] = dataset['price'].shift(3)
    dataset['momentum_4'] = dataset['price'].shift(4)
    dataset['momentum_5'] = dataset['price'].shift(5)
    
    #dataset["momentum return"]=dataset['Close']/dataset['momentum']
    dataset["momentum_1 return"]=dataset['price']/dataset['momentum_1']
    dataset["momentum_2 return"]=dataset['price']/dataset['momentum_2']
    dataset["momentum_3 return"]=dataset['price']/dataset['momentum_3']
    dataset["momentum_4 return"]=dataset['price']/dataset['momentum_4']
    dataset["momentum_5 return"]=dataset['price']/dataset['momentum_5']
    
    return dataset

def load_bitcoin(binance):
    trading_pair = 'BTC/USDT'
    candles = binance.fetch_ohlcv(trading_pair, '1h')
    ohlcv = [[ccxt.Exchange.iso8601(candle[0])] + candle[1:] for candle in candles] 
    bitcoin_hourly=pd.DataFrame(ohlcv)
    bitcoin_hourly.columns=["timestamp","Open","High","Low","Close","volume"]
    #bitcoin_hourly.dropna(inplace=True)

    #bitcoin_hourly.drop(columns=['timestamp'],inplace=True)
    bitcoin_hourly['Open']= pd.to_numeric(bitcoin_hourly['Open'],errors='coerce')
    bitcoin_hourly['High']= pd.to_numeric(bitcoin_hourly['High'],errors='coerce') 
    bitcoin_hourly['Low']= pd.to_numeric(bitcoin_hourly['Low'],errors='coerce') 
    bitcoin_hourly['Close']= pd.to_numeric(bitcoin_hourly['Close'],errors='coerce') 
    bitcoin_hourly['price']=bitcoin_hourly['Close']
    bitcoin_hourly.drop(columns=['Close'],inplace=True)
    #bitcoin_hourly.reset_index()
    dataset=get_technical_indicators(bitcoin_hourly)
    dataset['timestamp']=pd.to_datetime(dataset['timestamp'])
    bitcoin_hourly.set_index("timestamp",inplace=True)
    #print(dataset.dtypes)
    return dataset.tz_convert(None).reset_index()


def predict_trades(binance):
    bitcoin_hourly=load_bitcoin(binance)
    #print(bitcoin_hourly.shape)
    
    google_hourly=goog_trend.get_trend()
    #print(google_hourly.shape)
    dataset_search_hourly=bitcoin_hourly.merge(google_hourly,left_on='timestamp', right_on='Timestamp',how='left')
    dataset_search_hourly.drop(columns=['timestamp','Timestamp'],inplace=True)
    nn_features,nn_labels,nn_output,predict_features=prediction_model.load_data(dataset_search_hourly)
    #return_array=rebal_model_consecutive(nn_features,nn_labels)
    accuracy_score,predictions,prediction_dummy,labels_test,prediction_label=prediction_model.fit_rf(nn_features,nn_labels,len(nn_labels)-10,predict_features)

    print('model accuracy: ',accuracy_score)
    print('prediction: ',prediction_label)
    return prediction_label[0]

def execute_trades(binance,predict_dummy):
    balance=binance.fetchBalance()
    BTC_balance=round(balance['BTC']['free'],4)
    USDT_balance=round(balance['USDT']['free'],4)
    print("Account Balance BTC ", BTC_balance)
    print("Account Balance USDT ", USDT_balance)
    symbol='BTCUSDT'
    interval='1h'
    exchange = Binance()
    model = TradingModel(symbol=symbol, timeframe=interval)
        
    if predict_dummy:
        USDT_quantity=min(5,USDT_balance)
        print("Model predicts BUY, we are buying ", USDT_quantity)
        if USDT_quantity==0:
            sys.exit("USDT balance in account not sufficient")
        order_result = model.exchange.PlaceOrder(model.symbol, "BUY", "MARKET", quantity=10, test=False)
    else:
        BTC_quantity=min(.001,BTC_balance)
        print("Model predicts SELL, we are selling ", BTC_quantity)
        if BTC_quantity==0:
            sys.exit("BTC balance in account not sufficient")
        order_result = model.exchange.PlaceOrder(model.symbol, "SELL", "MARKET", quantity=10, test=False) 
          
    if "code" in order_result:
        print("\nERROR.")
        print(order_result)
    else:
        print("\nSUCCESS.")
        print(order_result)
        
    with open('trade_history.txt','a') as outfile:
        json.dump(order_result,outfile)
    
def Main():
    print(datetime.now())
    binance = ccxt.binance(binance_keys)
    predict_dummy=predict_trades(binance)
    execute_trades(binance,predict_dummy)

#schedule.every().hour.do(Main) 
#schedule.every(10).minutes.do(job)
schedule.every().minute.do(Main)

while True: 
    # Checks whether a scheduled task  
    # is pending to run or not 
    schedule.run_pending() 
    time.sleep(1) 

'''
if __name__ == '__main__':
	Main()
'''