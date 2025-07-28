# Intraday_trading

## General introduction

This is a general framework for live, automatic trading system that connects to metatrader5 and (as I used) Exness broker. Metatrader does assist automatic live trading through their desktop apps, but, for me, I used Python for most of the backtesing and brainstorming then this repo is more useful for me than adapting to their system. Moreover, if I ever have trading bot that based on machine learning algorithms whether for signals or supports (such as using LLMs for market sentiment), then this repo will be even more helpful.

## Repo structure

This repo will be structured as:

1. Live_trading: 
    * configs: contains any configuration (account log-in info);
    * data: serves as the database for the system (there will be a function to update the data to the latest data);
    * logs: serves as history records for any trading signals that our bots gave. We can access this with metatrader built-in functions ([refer to doc](https://www.mql5.com/en/docs/python_metatrader5/mt5historydealsget_py)), but for errors or issues such as lost connections, not turning on the 'Auto trading' function on the MetaTrader, they will not be recorded;
    * workplace: serves as main workplace for live_trading.

2. src: contains functions, modules, models, etc.

## Results' details

The results after running the file .\Intraday_trading\Live_trading\workplace\RSI_live_trade.py is showed (in short) as:
<pre>
login: 123
password: pass
path: C:/Program Files/MetaTrader 5 EXNESS/terminal64.exe
server: Exness-MT5Trial14

====================================================================================================
==================================================
Signals for XAUUSD
New signal: [[ 1 10 20]]
order_send failed, retcode=10027
==================================================
Signals for BTCUSD
New signal: [[0 0 0]]
No signals for BTCUSD
====================================================================================================
==================================================
Signals for XAUUSD
New signal: [[ 1 10 20]]
==================================================
Signals for BTCUSD
New signal: [[0 0 0]]
No signals for BTCUSD
</pre>

In this example, the first time the bot sent order to buy XAUUSD, it returned the code: 10027, ie the 'Auto trading' function is not turned on, we can just simply turn it on at the MetaTrader interface and do not need to shut down and run the program again.

## Prerequisites

For this repo to work:
1. Download the MetaTrader desktop app (they are different based on the broker; for exness, use this [link](https://expness.com/mt5/#:~:text=There%20is%20a%20direct%20link%20to%20download%20metatrader,terminals%20and%20find%20links%20to%20the%20installation%20files.))

2. Log-in to your account. Every time you run the function: login_metatrader() in the trade_support.py, the metatrader window will appear, and make sure to turn on the 'Algo Trading'.

3. The main function of this repo is the class **position_sender** in trade_support.py and its function send_order(type = type, price = None, sl = sl, tp = tp, time_end = time_end):

    Type will paired 1 with buy and -1 with sell; 
    If price is None, then the current market price will be used;
    SL and TP will be the desired money of SL and TP (for instance, sl is $10 and tp is $20);
    time_end has 2 options: good-till-close and end-of-day, refer the doc for more infos.

    Therefore, if your bots have different structures than the example ones I have in the models.py, then just make sure there outputs will have [Signals, SL, TP], SL and TP can be null as my sender will set a referenced SL and TP for safety.

## Disclaimers

This repository serves purely as a technical signal-sending engine for MetaTrader 5 (MT5). It is intended for personal engineering exploration and automation purposes only.

This project does not constitute financial advice.

I am not responsible for any financial losses or outcomes resulting from the use of this code or signals.

Use at your own risk. Always perform your own due diligence before making trading decisions.

This repo will be updated periodically to better adapt to the systems (logging more errors, more support functions, etc.).
Check out my other repos related to intraday_trading:
* [Trading_backtest_framework](https://github.com/AnhTNguyenBlank/Trading_backtest_framework): for backtesting strategies;
* [BERT_in_intraday_trading](https://github.com/AnhTNguyenBlank/BERT_in_intraday_trading): for the applications of BERT large language models even as a stand-alone or supported strategies;
* (To be updated more).