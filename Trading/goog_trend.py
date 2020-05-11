#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:55:57 2020

@author: ziyunchebn
"""
#import numpy as np
#import pytrends
#import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from pytrends.request import TrendReq
#import time
import pandas as pd
#import matplotlib

#plt.style.use('seaborn-darkgrid')
#matplotlib.rcParams['font.family'] = ['Heiti TC']


def rmax(maxrow: int=50):
    pd.set_option('display.max_rows', maxrow)

def cmax(maxcol: int=50):
    pd.set_option('display.max_columns', maxcol)
    

def get_trend():
    pytrends = TrendReq(hl='en-US', tz=0)
    kw_list=['bitcoin',"btc","blockchain","crypto","cryptocurrency"]
    
    today_obj=date.today()
    end_year=today_obj.year
    end_month=today_obj.month
    end_day=today_obj.day
    
    T_20=date.today()-timedelta(30)
    
    start_year=T_20.year
    start_month=T_20.month
    start_day=T_20.day
    
    hourly_data_gtrend=pytrends.get_historical_interest(kw_list, year_start=start_year, month_start=start_month, day_start=start_day, hour_start=0, year_end=end_year, month_end=end_month, day_end=end_day, hour_end=23, cat=0, geo='', gprop='', sleep=0)
    goog_hourly=hourly_data_gtrend.reset_index()
    goog_hourly['Timestamp']=pd.to_datetime(goog_hourly['date'])
    goog_hourly.set_index("Timestamp",inplace=True)
    goog_hourly['bitcoin']=goog_hourly['bitcoin']+1
    prev_time=0
    multiplier=1
    prev_trans=1
    timestamp=[]
    bitcoin_trend=[]
    for index, row in goog_hourly.iterrows():
        if row['date']!=prev_time:
            timestamp.append(row['date'])
            curr_level_trans=row['bitcoin']*multiplier
            bitcoin_trend.append(curr_level_trans)
            prev_time=row['date']
            prev_trans=curr_level_trans
        else:
            multiplier=prev_trans/row['bitcoin']
    goog_hourly_trans=pd.DataFrame({'Timestamp':timestamp, 'google':bitcoin_trend})
    #print(goog_hourly_trans)
    return goog_hourly_trans
