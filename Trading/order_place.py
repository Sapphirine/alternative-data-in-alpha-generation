#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:59:16 2020

@author: ziyunchebn
"""

from StrategyEvaluator import StrategyEvaluator
from Strategies import Strategies

from Binance import Binance
from TradingModel import TradingModel

import json

from decimal import Decimal, getcontext





def Main():
    exchange = Binance()
    symbols = exchange.GetTradingSymbols(quoteAssets=["ETH"])
    
    strategy_evaluators = [
        StrategyEvaluator(strategy_function=Strategies.bollStrategy),
        StrategyEvaluator(strategy_function=Strategies.maStrategy),
        StrategyEvaluator(strategy_function=Strategies.ichimokuBullish)
        ]
    opening_text = "\nWelcome to Tudor's Crypto Trading Bot. \n \
	Press 'b' (ENTER) to backtest all strategies \n \
	Press 'e' (ENTER) to execute the strategies on all coins \n \
	Press 'q' (ENTER) to quit. "
    
    print(opening_text)
    
    '''
    answer = input()
	while answer not in ['b', 'e', 'q']:
		print(opening_text)
		answer = input()

	if answer == 'e':
		EvaluateStrategies(symbols=symbols, interval='5m', strategy_evaluators=strategy_evaluators)
	if answer == 'b':
		BacktestStrategies(symbols=symbols, interval='5m', plot=True, strategy_evaluators=strategy_evaluators)
	if answer == 'q':
		print("\nBYE!\n")
    '''
    answer = input()
    symbol='ETCBTC'
    interval='1h'
    
    model = TradingModel(symbol=symbol, timeframe=interval)
    
    if answer=='buy':
        order_result = model.exchange.PlaceOrder(model.symbol, "BUY", "MARKET", quantity=0.02, test=False)
    elif answer=='sell':
        order_result = model.exchange.PlaceOrder(model.symbol, "SELL", "MARKET", quantity=0.01, test=False)       

        if "code" in order_result:
            print("\nERROR.")
            print(order_result)
        else:
            print("\nSUCCESS.")
            print(order_result)
        
    
if __name__ == '__main__':
	Main()