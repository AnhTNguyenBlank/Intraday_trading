import MetaTrader5 as mt
import pandas as pd
import ml_collections
import yaml
import json

from datetime import datetime
import sys

sys.path.insert(0, 'D:/Intraday_trading')

from src.support import *
from src.models import *
from src.models_support import *
from src.trade_support import *

import time


if __name__ == '__main__':
    config_dir = 'D:/Intraday_trading/Live_trading/configs/config.yaml'
    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))

    # Login to metatrader
    acc_config = config.demo_acc
    print(acc_config)
    login_metatrader(acc_config = acc_config)
    
    # define alpha, trading configs
    alpha_config = {'RSI_PARAMS': {'INPUT_PARAM': 7, 'CUTOFF_BUY': 30, 'CUTOFF_SELL': 70}, 
                    'base_SL': 10, 
                    'base_TP': 20
    }

    max_existing_position = 3

    # Import alpha
    strat = RSI_strat(config = alpha_config)

    # set up asset sender
    XAU_sender = positions_sender(
        symbol = 'XAUUSDm',
        action = mt.TRADE_ACTION_DEAL,
        volume = 0.01,
        deviation = 20,
        comment = 'RSI_strat live test',
        type_filling = mt.ORDER_FILLING_FOK,
        ref_profit = 20,
        ref_loss = 10
        # type = mt.ORDER_TYPE_BUY, #***
        # price = None,
        # sl = None,
        # tp = None,
        # time_end = mt.ORDER_TIME_GTC, # mt.ORDER_TIME_DAY
    )

    BTC_sender = positions_sender(
        symbol = 'BTCUSDm',
        action = mt.TRADE_ACTION_DEAL,
        volume = 0.01,
        deviation = 20,
        comment = 'RSI_strat live test',
        type_filling = mt.ORDER_FILLING_FOK,
        ref_profit = 20,
        ref_loss = 10
        # type = mt.ORDER_TYPE_BUY, #***
        # price = None,
        # sl = None,
        # tp = None,
        # time_end = mt.ORDER_TIME_GTC, # mt.ORDER_TIME_DAY
    )
    
    for _ in range(120):
        print('='*100)  
        print('='*50)  
        print('Signals for XAUUSD')
        XAU_signal = strat.live_signal(asset = 'XAUUSD', time_frame = '1M')
        print(f'New signal: {XAU_signal}')
        
        num_current_XAU_positions = len(mt.positions_get(symbol = 'XAUUSDm'))
        if num_current_XAU_positions < max_existing_position:
            if XAU_signal[0][0] != 0:
                if XAU_signal[0][0] == -1:
                    type = mt.ORDER_TYPE_SELL
                elif XAU_signal[0][0] == 1:
                    type = mt.ORDER_TYPE_BUY

                time_end = mt.ORDER_TIME_GTC # mt.ORDER_TIME_DAY

                sl = XAU_signal[0][1]
                tp = XAU_signal[0][2]

                order_sent = XAU_sender.send_order(type = type, price = None, sl = sl, tp = tp, time_end = time_end)
            else:
                print('No signals for XAUUSD')
        else:
            print('Reached maximum number of existing positions for XAUUSD')


        print('='*50)  
        print('Signals for BTCUSD')
        BTC_signal = strat.live_signal(asset = 'BTCUSD', time_frame = '1M')
        print(f'New signal: {BTC_signal}')

        num_current_BTC_positions = len(mt.positions_get(symbol = 'BTCUSDm'))
        if num_current_BTC_positions < max_existing_position:
            if BTC_signal[0][0] != 0:
                if BTC_signal[0][0] == -1:
                    type = mt.ORDER_TYPE_SELL
                elif BTC_signal[0][0] == 1:
                    type = mt.ORDER_TYPE_BUY

                time_end = mt.ORDER_TIME_GTC # mt.ORDER_TIME_DAY

                sl = BTC_signal[0][1]
                tp = BTC_signal[0][2]

                order_sent = BTC_sender.send_order(type = type, price = None, sl = sl, tp = tp, time_end = time_end)
            else:
                print('No signals for BTCUSD')
        else:
            print('Reached maximum number of existing positions for BTCUSD')
        

        time.sleep(60)
        
