import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 999)
from datetime import datetime

import ta

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser


plt.style.use('classic')

import MetaTrader5 as mt
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta, timezone

from tqdm import tqdm
import contextlib
import os
import json


from src.support import *
from src.models_support import *


# =================================== Trading meta trader 5 support =================================== #


def login_metatrader(acc_config):
    
    # runs MetaTrader5 client
    # MetaTrader5 platform must be installed
    if not mt.initialize(path = acc_config.path,
                login = acc_config.login,
                password = acc_config.password,
                server = acc_config.server
                ):
        return(mt.last_error())
    else:
        # log in to trading account
        mt.login(acc_config.login, acc_config.password, acc_config.server) 
 

def acc_info_rt(df_acc):
    account_info_dict = mt.account_info()._asdict()
    temp = pd.DataFrame(list(account_info_dict.items())).transpose()
    temp.columns = temp.iloc[0, :]
    temp = temp.drop(index = 0)
    temp['updated_at'] = datetime.now()

    df_acc = pd.concat([df_acc, temp], axis = 0)
    return(df_acc.reset_index())


def plot_acc_info_rt(df_acc):
    # Define subplot heights and widths
    subplot_heights = [100, 100, 100]  # Adjust these values based on your preferences
    subplot_widths = [1]  # Only one column

    # Create subplot with 2 rows and 1 column
    fig = make_subplots(rows=3,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=('Balance, Equity', 'Margin, Free Margin', 'Margin level'),
                        row_heights=subplot_heights,
                        column_widths=subplot_widths,
                        vertical_spacing = 0.05,  # Set the spacing between rows,
                        specs=[
                            [{"secondary_y": True}], 
                            [{"secondary_y": True}],
                            [{"secondary_y": True}],
                            ]
                        )

    # Subplot 1: Balance, Profit, Equity
    balance = go.Bar(x=df_acc['updated_at'],
                                y=df_acc['balance'],
                                name='Balance',
                                # width = 200,
                                marker=dict(color='blue') ,
                                text=df_acc['balance'].values,
                                textposition="outside"
                                )

    equity = go.Bar(x=df_acc['updated_at'],
                                y=df_acc['equity'],
                                name='Equity',
                                # width = 200,
                                marker=dict(color='red'),
                                text=df_acc['equity'].values,
                                textposition="outside"
                                )

    profit = go.Scatter(x=df_acc['updated_at'],
                                y=df_acc['profit'],
                                mode='lines+markers+text',
                                name='Profit',
                                line=dict(color='green', width = 5),
                                text = df_acc['profit'].astype(float).round(2).values,
                                textposition = 'top center')
                                

    fig.add_trace(balance, 
                  row=1, 
                  col=1,
                  secondary_y=False
                  )
    
    fig.add_trace(equity,
                  row=1,
                  col=1,
                  secondary_y=False)
    
    fig.add_trace(profit,
                  row=1,
                  col=1,
                  secondary_y = True)
    
    # # Subplot 2: Margin, Free Margin
    margin = go.Scatter(x=df_acc['updated_at'],
                                y=df_acc['margin'],
                                mode='lines+markers+text',
                                name='Margin',
                                line=dict(color='blue', width = 2),
                                text=df_acc['margin'].values,
                                textposition="top center")

    margin_free = go.Scatter(x=df_acc['updated_at'],
                                y=df_acc['margin_free'],
                                mode='lines+markers+text',
                                name='Margin_free',
                                line=dict(color='yellow', width = 2),
                                text=df_acc['margin_free'].values,
                                textposition="top center")


    fig.add_trace(margin, row=2, col=1)
    fig.add_trace(margin_free,
                    row=2,
                    col=1)

    # # Subplot 3: Margin level
    margin_level = go.Bar(x=df_acc['updated_at'],
                                y=df_acc['margin_level'],
                                name='Margin_level',
                                # width = 200,
                                marker=dict(color='red'),
                                text=df_acc['margin_level'].astype(float).round(2).values,
                                textposition="outside"
                                )

    fig.add_trace(margin_level,
                    row=3,
                    col=1)
    
    # Add slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=False,
                thickness=0.05,  # Adjust the thickness of the slider
                bgcolor='rgba(0,0,0,0.1)',  # Set the background color of the slider
            ),
            type='date',
            ),

        height = 800,
        width = 1300,
        plot_bgcolor='black',  # Transparent background
        paper_bgcolor='black',  # Transparent paper background
        font = dict(color = 'white'),
        legend = dict(x = 1.01, y = 1),

        barmode='group', bargap=0.30,bargroupgap=0.0
        
    )

    # Fix y-axis range for each subplot
    fig.update_yaxes(
        # autorange = True, 
        range=[df_acc['balance'].values[0]*0.9, df_acc['balance'].values[0]*1.05], 
        row=1, col=1, fixedrange= False, secondary_y = False)  # Adjust as needed
    
    fig.update_yaxes(
        autorange = True, 
        # range=[df_acc['profit'].min() - 5, df_acc['profit'].max() + 5], 
        # row=1, col=1, fixedrange= False, 
        secondary_y = True
        )  # Adjust as needed

    fig.update_yaxes(
        # autorange = True, 
        range=[-10, df_acc[['margin', 'margin_free']].max().max()*1.5], 
        row=2, col=1, fixedrange= False)  # Adjust as needed
    
    fig.update_yaxes(
        range=[0, df_acc['margin_level'].max()*1.5], 
        row=3, col=1, fixedrange= False)  # Adjust as needed


    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='white',
        gridcolor='grey',
    )

    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='white',
        gridcolor='grey'
    )

    return(fig)


def positions_rt(df_positions):
    positions = mt.positions_get()
    temp = pd.DataFrame(positions, columns = positions[0]._asdict().keys())
    temp['time'] = pd.to_datetime(temp['time'], unit = 's')
    temp['time_msc'] = pd.to_datetime(temp['time_msc'], unit = 'ms')
    temp['time_update'] = pd.to_datetime(temp['time_update'], unit = 's')
    temp['time_update_msc'] = pd.to_datetime(temp['time_update_msc'], unit = 'ms')
    temp['VN_time'] = temp['time'] + pd.Timedelta('7 hours')
    temp['VN_time_msc'] = temp['time_msc'] + pd.Timedelta('7 hours')
    temp['VN_time_update'] = temp['time_update'] + pd.Timedelta('7 hours')
    temp['VN_time_update_msc'] = temp['time_update_msc'] + pd.Timedelta('7 hours')
    temp['updated_at'] = datetime.now()
    df_positions = pd.concat([df_positions, temp], axis = 0)
    trading_symbols = list(df_positions['symbol'].unique())
    return(df_positions, trading_symbols)


def plot_positions_rt(trading_symbols, df_positions):
    for symbol in trading_symbols:
        print('='*100)
        print(symbol)
        print('='*100)
        tickets = df_positions[['ticket', 'time', 'price_open', 'sl', 'tp', 'type', 'reason', 'symbol']].drop_duplicates()
        df_price = pd.DataFrame(mt.copy_rates_range(symbol, mt.TIMEFRAME_M1, df_positions['time'].min() - pd.Timedelta('7 hours'), datetime.now()))

        df_price['time'] = pd.to_datetime(df_price['time'], unit = 's')
        df_price.index = df_price['time']
        df_price = df_price[['open', 'high', 'low', 'close', 'tick_volume']]
        df_price.columns = ['open', 'high', 'low', 'close', 'tick_vol']
        df_price.columns = [col.upper() for col in df_price.columns]
        df_price = prepare_df(df_price, timeframe = '5min')

        fig = plot_df(df_price, path = None, open_tab = False)

        buy_tickets = tickets[tickets['type'] == 0]
        sell_tickets = tickets[tickets['type'] == 1]

        for idt in buy_tickets.index:
            fig.add_annotation(
                text=buy_tickets['price_open'][idt],
                x=buy_tickets['time'][idt],  # X-coordinate of the text box
                y=buy_tickets['price_open'][idt],  # Y-coordinate of the text box
                showarrow=True,
                arrowhead = 2,
                arrowwidth = 2,
                arrowcolor = 'green',
                arrowside = 'end',
                opacity = 1,
                ax=0,
                ay=100,
                font=dict(
                family="Arial, sans-serif",
                size=15,
                color="white",
                ),
                bordercolor="green",
                borderwidth=2,
                borderpad=4,
                bgcolor="green",
            )

            sl = dict(type='line',
                        x0=df_price.index.min(),
                        x1=df_price.index.max(),
                        y0=buy_tickets['sl'][idt],
                        y1=buy_tickets['sl'][idt],
                        line=dict(color='red', width=1, dash='dash'))

            tp = dict(type='line',
                        x0=df_price.index.min(),
                        x1=df_price.index.max(),
                        y0=buy_tickets['tp'][idt],
                        y1=buy_tickets['tp'][idt],
                        line=dict(color='green', width=1, dash='dash'))

            fig.add_shape(sl,
                        row=1,
                        col=1)

            fig.add_shape(tp, 
                        row=1, 
                        col=1)

        for idt in sell_tickets.index:
            fig.add_annotation(
                text=sell_tickets['price_open'][idt],
                x=sell_tickets['time'][idt],  # X-coordinate of the text box
                y=sell_tickets['price_open'][idt],  # Y-coordinate of the text box
                showarrow=True,
                arrowhead = 2,
                arrowwidth = 2,
                arrowcolor = 'red',
                arrowside = 'end',
                opacity = 1,
                ax=0,
                ay=-100,
                font=dict(
                family="Arial, sans-serif",
                size=15,
                color="white",
                ),
                bordercolor='red',
                borderwidth=2,
                borderpad=4,
                bgcolor="red",
            )

            sl = dict(type='line',
                        x0=df_price.index.min(),
                        x1=df_price.index.max(),
                        y0=buy_tickets['sl'][idt],
                        y1=buy_tickets['sl'][idt],
                        line=dict(color='red', width=1, dash='dash'))

            tp = dict(type='line',
                        x0=df_price.index.min(),
                        x1=df_price.index.max(),
                        y0=buy_tickets['tp'][idt],
                        y1=buy_tickets['tp'][idt],
                        line=dict(color='green', width=1, dash='dash'))

            fig.add_shape(sl,
                        row=1,
                        col=1)

            fig.add_shape(tp, 
                        row=1, 
                        col=1)
            
        return(fig)


class positions_sender:
    def __init__(self, 
                symbol,
                action,
                volume,
                deviation,
                comment,
                type_filling,
                ref_profit = 20,
                ref_loss = 10,
                ):

        self.symbol = symbol
        self.action = action
        self.volume = volume
        self.deviation = deviation
        self.comment = comment
        self.type_filling = type_filling
        self.ref_profit = ref_profit
        self.ref_loss = ref_loss
        
        
    def get_market_price(self, type):
        '''
        type = 0: ORDER_TYPE_BUY
        type = 1: ORDER_TYPE_SELL
        type = 2: ORDER_TYPE_BUY_LIMIT
        type = 3: ORDER_TYPE_SELL_LIMIT
        type = 4: ORDER_TYPE_BUY_STOP
        type = 5: ORDER_TYPE_SELL_STOP
        '''
        if type == 0 or type == 2 or type == 4:
            return mt.symbol_info(self.symbol).ask
        elif type == 1 or type == 3 or type == 5:
            return mt.symbol_info(self.symbol).bid


    def get_direction_multiplier(self, type):
        if type == 0 or type == 2 or type == 4:
            return 1
        elif type == 1 or type == 3 or type == 5:
            return -1


    def get_profit_multiplier(self):
        if self.symbol[0:3] == 'BTC':
            return 100
        elif self.symbol[0:3] == 'XAU':
            return 10


    def send_order(self, type, price, sl, tp, time_end):
        
        '''
        auto set up desired profit and loss if not provided sl, tp level: profit 20 usd and loss 10 usd
        type = 0: ORDER_TYPE_BUY
        type = 1: ORDER_TYPE_SELL
        type = 2: ORDER_TYPE_BUY_LIMIT
        type = 3: ORDER_TYPE_SELL_LIMIT
        type = 4: ORDER_TYPE_BUY_STOP
        type = 5: ORDER_TYPE_SELL_STOP

        action = 1: TRADE_ACTION_DEAL
        action = 5: TRADE_ACTION_PENDING
        '''

        if self.symbol is None:
            raise AssertionError('Missing symbol!')
        
        if self.action is None:
            raise AssertionError('Missing action!')
            
        if self.volume is None:
            self.volume = 0.01

        if type is None:
            raise AssertionError('Missing action type (buy/ sell by market/ stop/ limit)!')
        elif (type == 0 or type == 1) and (self.action != 1):
            raise AssertionError('Action and type are not valid. May be the action = TRADE_ACTION_DEAL')
        elif (type == 2 or type == 3 or type == 4 or type == 5) and (self.action != 5):
            raise AssertionError('Action and type are not valid. May be the action = TRADE_ACTION_PENDING')    

        if (price is None) and (self.action == 1):
            price = self.get_market_price(type = type)
        elif (price is None) and (self.action == 5):
            raise AssertionError('Action and price are not valid. Price should not be null in PENDING orders')    
        elif (price is not None) and (self.action == 5) and (type == 2 or type == 5) and (price > self.get_market_price(type = type)):
            raise AssertionError('Price is not valid. Input price should be LOWER than current asset price in BUY LIMIT/ SELL STOP orders')
        elif (price is not None) and (self.action == 5) and (type == 3 or type == 4) and (price < self.get_market_price(type = type)):
            raise AssertionError('Price is not valid. Input price should be HIGHER than current asset price in BUY STOP/ SELL LIMIT orders')
        
        
        direction_multiplier = self.get_direction_multiplier(type)
        profit_multiplier = self.get_profit_multiplier()
        
        if sl is None:
            sl = price - mt.symbol_info(self.symbol).point * direction_multiplier * self.ref_loss * profit_multiplier / self.volume
        else:
            sl = price - mt.symbol_info(self.symbol).point * direction_multiplier * sl * profit_multiplier / self.volume

        if tp is None:
            tp = price + mt.symbol_info(self.symbol).point * direction_multiplier * self.ref_profit * profit_multiplier / self.volume
        else: 
            tp = price + mt.symbol_info(self.symbol).point * direction_multiplier * tp * profit_multiplier / self.volume

        if time_end is None:
            time_end = mt.ORDER_TIME_GTC
        else:
            time_end = time_end

        if self.type_filling is None:
            self.type_filling = mt.ORDER_FILLING_FOK
        
        request = {
            "action": self.action,
            "symbol": self.symbol,
            "volume": self.volume,
            "type": type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.deviation,
            "comment": self.comment,
            "time_end": time_end, # mt.ORDER_TIME_GTC, # mt.ORDER_TIME_DAY
            "type_filling": self.type_filling,
        }

        # send a trading request
        result = mt.order_send(request)

        if result.retcode != mt.TRADE_RETCODE_DONE:
            print("order_send failed, retcode={}".format(result.retcode))
                

        # Buy stop/ limit

        # symbol = 'BTCUSDm'

        # action = mt.TRADE_ACTION_PENDING
        # volume = 0.03
        # type = mt.ORDER_TYPE_SELL_LIMIT
        # deviation = 20
        # comment = 'python script open'
        # time_end = mt.ORDER_TIME_GTC # mt.ORDER_TIME_DAY
        # type_filling = mt.ORDER_FILLING_FOK

        # price = get_market_price(symbol, type) + 2000
            
        # sl = None
        # tp = None
        

        # Partial close

        # close position
        # positions = mt.positions_get()
        # print('open positions', positions)

        # # Working with 1st position in the list and closing it
        # pos1 = positions[0]

                
        # def reverse_type(type):
        #     # to close a buy positions, you must perform a sell position and vice versa
        #     if type == mt.ORDER_TYPE_BUY:
        #         return mt.ORDER_TYPE_SELL
        #     elif type == mt.ORDER_TYPE_SELL:
        #         return mt.ORDER_TYPE_BUY


        # def get_close_price(symbol, type):
        #     if type == mt.ORDER_TYPE_BUY:
        #         return mt.symbol_info(symbol).bid
        #     elif type == mt.ORDER_TYPE_SELL:
        #         return mt.symbol_info(symbol).ask

        # request = {
        #     "action": mt.TRADE_ACTION_DEAL,
        #     "position": pos1.ticket,
        #     "symbol": pos1.symbol,
        #     "volume": 0.02,
        #     "type": reverse_type(pos1.type),
        #     "price":get_close_price(pos1.symbol, pos1.type),
        #     "deviation": 20,
        #     "magic": 0,
        #     "comment": "python close order",
        #     "time_end": mt.ORDER_TIME_GTC,
        #     "type_filling": mt.ORDER_FILLING_IOC,  # some brokers accept mt.ORDER_FILLING_FOK only
        # }

        # result = mt.order_send(request)
        # result

        if result.retcode != mt.TRADE_RETCODE_DONE:
            status = 'FAILED'
        else:
            status = 'SUCCEED'

        log_entry = {
            "time_send_order": datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
            "retcode": result.retcode,
            "deal": result.deal,
            "order": result.order,
            "volume": result.volume,
            "price": result.price,
            "bid": result.bid,
            "ask": result.ask,
            "request_id": result.request_id,
            "retcode_external": result.retcode_external,
            "status": status,
            "comment": result.comment,
            
            # Flatten request content
            "request": {
                "symbol": result.request.symbol,
                "volume": result.request.volume,
                "price": result.request.price,
                "sl": result.request.sl,
                "tp": result.request.tp,
                "type": result.request.type,
                "action": result.request.action,
                "deviation": result.request.deviation,
                "comment": result.request.comment,
                "magic": result.request.magic,
                "type_filling": result.request.type_filling,
                "type_time": result.request.type_time,
                "expiration": result.request.expiration
            }
        }

        dt_obj = datetime.strptime(log_entry['time_send_order'], "%Y-%m-%d-%H:%M:%S")
        # Format it however you want
        formatted = dt_obj.strftime("%Y_%m_%d")

        with open(f"D:/Intraday_trading/Live_trading/logs/deal_logs_{formatted}.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # with open(f"D:/Intraday_trading/Live_trading/logs/deal_logs_{formatted}.jsonl", "r") as f:
        #     logs = [json.loads(line) for line in f]


        return(result)


def update_data(asset, time_frame):
    
    if time_frame == '1M':
        time_dummy = timedelta(minutes = 1)
        time_var = mt.TIMEFRAME_M1
    elif time_frame == '15M':
        time_dummy = timedelta(minutes = 15)
        time_var = mt.TIMEFRAME_M15
    elif time_frame == 'H4':
        time_dummy = timedelta(hours = 4)
        time_var = mt.TIMEFRAME_H4
    elif time_frame == 'D1':
        time_dummy = timedelta(days = 1)
        time_var = mt.TIMEFRAME_D1
    
    ohlc = pd.read_pickle(f'D:/Intraday_trading/Live_trading/data/{asset}_ohlc_{time_frame}.pkl')
    ohlc = ohlc[ohlc['flag_candle_end'] == 1]

    # Compute now date
    from_date = ohlc['time'].max()
    to_date = datetime.now()

    rates = mt.copy_rates_range(f"{asset}m", time_var, from_date, to_date)
    df_rates = pd.DataFrame(rates)

    df_rates["time"] = pd.to_datetime(df_rates["time"], unit="s")

    df_rates['flag_candle_end'] = df_rates['time'] + timedelta(hours = 7) + time_dummy
    df_rates['flag_candle_end'] = df_rates['flag_candle_end'].apply(lambda x: 1 if x < datetime.now() else 0)

    ohlc = pd.concat([ohlc, df_rates], axis = 0, ignore_index = True)
    ohlc = ohlc.drop_duplicates(subset='time', keep='last')
    
    ohlc.to_pickle(f'D:/Intraday_trading/Live_trading/data/{asset}_ohlc_{time_frame}.pkl')



