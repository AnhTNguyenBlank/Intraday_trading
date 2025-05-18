import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 999)
<<<<<<< Updated upstream
=======
from scipy.stats import levy_stable

from datetime import datetime
from scipy.stats import kstest
from scipy.stats import jarque_bera
# from arch.unitroot import ADF
from scipy.stats import kurtosis
from scipy.stats import skew
# from arch import arch_model

import pickle
>>>>>>> Stashed changes

import ta

import matplotlib.pyplot as plt

<<<<<<< Updated upstream
plt.style.use('classic')

=======
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope

from scipy.signal import argrelmin
from scipy.signal import argrelmax

plt.style.use('classic')


import MetaTrader5 as mt
import pandas as pd
import ml_collections
import yaml

from datetime import datetime
import pytz
import sys


>>>>>>> Stashed changes
#=======================ALPHAS=======================


class Base_Alpha:
    def __init__(self, strat, prepare_data):
        # Prepare function
        self.prepare_data = prepare_data
        # Strategy function
        self.strat = strat 

        self.config = None

    def signal(self, df_observe):
        '''
        Return the tested dataframe with additional columns: signal (buy/ sell), stop loss, take profit.
        In case eval = False --> live trading, return only the vector of (signal, stop_loss, take_profit).
        '''
        df_prepare = self.prepare_data(df_observe)           
        # Apply the strategy to the dataframe and return the result in IS
        df_result = df_prepare.copy().apply(self.strat, axis=1)

        return(df_result)



class RSI_strat(Base_Alpha):
    def __init__(self, config):
        super().__init__(
            strat = self._RSI_strat,
            prepare_data = self._prepare_data
        )  # Pass the strategy function to the parent constructor
        
        '''
<<<<<<< Updated upstream
        config = {'RSI_PARAMS': {'INPUT_PARAM': 14, 'CUTOFF_BUY': 30, 'CUTOFF_SELL': 70}, 
                'base_SL': 10, 
                'base_TP': 20
                }
        '''
        self.RSI_param = config['RSI_PARAMS']
=======
        config = {'RSI_param': 14, 'base_SL': 10, 'base_TP': 20}
        '''
        self.RSI_param = config['RSI_param']
>>>>>>> Stashed changes
        self.base_SL = config['base_SL']
        self.base_TP = config['base_TP']
        
        
    def _prepare_data(self, df_observe):
<<<<<<< Updated upstream
        df_observe['RSI'] = ta.momentum.RSIIndicator(df_observe['CLOSE'], window = self.RSI_param['INPUT_PARAM']).rsi()
=======
        df_observe['RSI'] = ta.momentum.RSIIndicator(df_observe['CLOSE'], window = self.RSI_param).rsi()
>>>>>>> Stashed changes
        return(df_observe)

    def _RSI_strat(self, row):
        '''Strategy for RSI, calculates the signal based on the RSI value.'''
<<<<<<< Updated upstream
        if row['RSI'] < self.RSI_param['CUTOFF_BUY']:
            return pd.Series([1, self.base_SL, self.base_TP], index=['SIGNAL', 'SL', 'TP']) # Buy signal
        elif row['RSI'] > self.RSI_param['CUTOFF_BUY']:
            return pd.Series([-1, self.base_SL, self.base_TP], index=['SIGNAL', 'SL', 'TP']) # Sell signal
        else:
            return pd.Series([0, 0, 0], index=['SIGNAL', 'SL', 'TP']) # No action (neutral)
        

class MCMC_strat_v1(Base_Alpha):
    def __init__(self, config):
        super().__init__(
            strat = self._MCMC_strat_v1,
            prepare_data = self._prepare_data
        )

        '''
        config = {
            'base_SL': 10, 
            'base_TP': 20,
            'DECISION_CUTOFF': {'ENTRY_BUY_CUTOFF': 1, 'ENTRY_SELL_CUTOFF': -1}
        }

        Conduct on simple transition 1-step matrix between the previous and current log return
        '''
        self.base_SL = config['base_SL']
        self.base_TP = config['base_TP']
        self.DECISION_CUTOFF = config['DECISION_CUTOFF']
        
        
    def _prepare_data(self, df_observe):
        #returns
        df_observe['RET(T)'] = 100*(df_observe['CLOSE'] - df_observe['CLOSE'].shift(1))/df_observe['CLOSE'].shift(1)
        return(df_observe)

    def _MCMC_strat_v1(self, row):
        if ((row['RET(T)'] > self.DECISION_CUTOFF['ENTRY_BUY_CUTOFF']) & (pd.isnull(row['RET(T)']) == False)):
            return pd.Series([1, self.base_SL, self.base_TP], index=['SIGNAL', 'SL', 'TP']) # Buy signal
        elif ((row['RET(T)'] < self.DECISION_CUTOFF['ENTRY_SELL_CUTOFF']) & (pd.isnull(row['RET(T)']) == False)):
=======
        if row['RSI'] < 30:
            return pd.Series([1, self.base_SL, self.base_TP], index=['SIGNAL', 'SL', 'TP']) # Buy signal
        elif row['RSI'] > 70:
>>>>>>> Stashed changes
            return pd.Series([-1, self.base_SL, self.base_TP], index=['SIGNAL', 'SL', 'TP']) # Sell signal
        else:
            return pd.Series([0, 0, 0], index=['SIGNAL', 'SL', 'TP']) # No action (neutral)
        

