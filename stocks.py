import numpy as np
import pandas as pd
import tushare as ts

data = ts.get_hist_data('sh',start='2017-01-01',end='2019-07-07')
money = ts.get_money_supply()
