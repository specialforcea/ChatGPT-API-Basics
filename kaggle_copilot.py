import lightgbm as lgb 
import xgboost as xgb 
import catboost as cbt 
import numpy as np 
import joblib 
import os 
import time
import pickle
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import gc
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import warnings

warnings.filterwarnings('ignore')

def timeit(func):
    def wrapper(*args, **kwargs):
        #calculate the execution time        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} with {kwargs}: {execution_time} seconds")
        return result
    return wrapper



model_path ='/kaggle/input/cross-sectional/models_2'

out_path = f'./models_2'

os.system('mkdir models_2')

N_fold = 5

TRAINING =  False

@timeit
def get_stock_clusters(df):
    
    df_cor = df[['stock_id','date_id','seconds_in_bucket','wap']]
    df_cor = df_cor.dropna()
    pivot_df = df_cor.pivot_table(values='wap', index=['date_id','seconds_in_bucket'], columns='stock_id')
    correlation_matrix = pivot_df.pct_change(10).corr()
    distance_matrix = 1 - abs(correlation_matrix)
    Z = linkage(squareform(distance_matrix), method='ward')
    threshold = 0.95
    clusters = fcluster(Z, threshold, criterion='distance')
    df_stock_groups = pd.DataFrame({'stock': correlation_matrix.columns, 'group': clusters})
    return df_stock_groups


# All feature selections
@timeit
def group_features(df, df_stock_groups):
    
    df['group'] = df['stock_id'].replace(df_stock_groups['group'])
    temp = df.groupby(['date_id','seconds_in_bucket','group']).agg(
        gp_wap=('wap','mean'),
        gp_bid_price=('bid_price','mean'),
        gp_ask_price=('ask_price','mean'),
        gp_bid_size=('bid_size','sum'),
        gp_ask_size=('ask_size','sum'),        
    ).reset_index()

    df = df.merge(temp, on=['date_id','seconds_in_bucket','group'], how='left')
    return df

@timeit
def group_excess_return(df, price, periods):

    for p in periods:
        df_temp = df.groupby(['stock_id','date_id'])[[price, f'gp_{price}']].pct_change(p)
        df_temp[f'gp_ex_{price}_ret_{p}'] = df_temp[price] - df_temp[f'gp_{price}']
        df_temp = df_temp.drop([price, f'gp_{price}'], axis=1)
        df = df.join(df_temp)
    return df

@timeit
def idx_feats(df, price):

    weights = [
        0.004, 0.001, 0.002, 0.006, 0.004, 0.004, 0.002, 0.006, 0.006, 0.002, 0.002, 0.008,
        0.006, 0.002, 0.008, 0.006, 0.002, 0.006, 0.004, 0.002, 0.004, 0.001, 0.006, 0.004,
        0.002, 0.002, 0.004, 0.002, 0.004, 0.004, 0.001, 0.001, 0.002, 0.002, 0.006, 0.004,
        0.004, 0.004, 0.006, 0.002, 0.002, 0.04 , 0.002, 0.002, 0.004, 0.04 , 0.002, 0.001,
        0.006, 0.004, 0.004, 0.006, 0.001, 0.004, 0.004, 0.002, 0.006, 0.004, 0.006, 0.004,
        0.006, 0.004, 0.002, 0.001, 0.002, 0.004, 0.002, 0.008, 0.004, 0.004, 0.002, 0.004,
        0.006, 0.002, 0.004, 0.004, 0.002, 0.004, 0.004, 0.004, 0.001, 0.002, 0.002, 0.008,
        0.02 , 0.004, 0.006, 0.002, 0.02 , 0.002, 0.002, 0.006, 0.004, 0.002, 0.001, 0.02,
        0.006, 0.001, 0.002, 0.004, 0.001, 0.002, 0.006, 0.006, 0.004, 0.006, 0.001, 0.002,
        0.004, 0.006, 0.006, 0.001, 0.04 , 0.006, 0.002, 0.004, 0.002, 0.002, 0.006, 0.002,
        0.002, 0.004, 0.006, 0.006, 0.002, 0.002, 0.008, 0.006, 0.004, 0.002, 0.006, 0.002,
        0.004, 0.006, 0.002, 0.004, 0.001, 0.004, 0.002, 0.004, 0.008, 0.006, 0.008, 0.002,
        0.004, 0.002, 0.001, 0.004, 0.004, 0.004, 0.006, 0.008, 0.004, 0.001, 0.001, 0.002,
        0.006, 0.004, 0.001, 0.002, 0.006, 0.004, 0.006, 0.008, 0.002, 0.002, 0.004, 0.002,
        0.04 , 0.002, 0.002, 0.004, 0.002, 0.002, 0.006, 0.02 , 0.004, 0.002, 0.006, 0.02,
        0.001, 0.002, 0.006, 0.004, 0.006, 0.004, 0.004, 0.004, 0.004, 0.002, 0.004, 0.04,
        0.002, 0.008, 0.002, 0.004, 0.001, 0.004, 0.006, 0.004,
    ]

    weights = np.array(weights)
    df_weights = pd.DataFrame(weights, columns=['weights'])
    df_weights.index.name = 'stock_id'
    df_weights = df_weights.reset_index()

    temp = df[['stock_id', 'date_id', 'seconds_in_bucket', price]].merge(df_weights, on='stock_id', how='left').fillna(0.)
    # groupby date_id and seconds_in_bucket, sum the product of wap and weights
    temp = temp.groupby(['date_id', 'seconds_in_bucket'])[[price, 'weights']].apply(lambda x: (x[price] * x['weights']).sum())
    temp = temp.reset_index()
    temp.columns = ['date_id', 'seconds_in_bucket', f'idx_{price}']
    # merge with df
    df = df.merge(temp, on=['date_id', 'seconds_in_bucket'], how='left')
    return df

@timeit
def idx_excess_return(df, price, periods):
    
    for p in periods:
        df_temp = df.groupby(['stock_id','date_id'])[[price, f'idx_{price}']].pct_change(p)
        df_temp[f'idx_ex_{price}_ret_{p}'] = df_temp[price] - df_temp[f'idx_{price}']
        df_temp = df_temp.drop([price, f'idx_{price}'], axis=1)
        df = df.join(df_temp)
    return df

# calculate technical indicators of prices such as SMA, EMA, Bollinger Bands
@timeit
def technical_indicators_feats(df, price, periods=[10, 50, 100]):

    for p in periods:
        # calculate simple moving average with window size p
        df[f'sma_{price}_{p}'] = df.groupby(['stock_id','date_id'])[price].transform(lambda x: x.rolling(window=p).mean())
        # calculate exponential moving average with window size p
        df[f'ema_{price}_{p}'] = df.groupby(['stock_id','date_id'])[price].transform(lambda x: x.ewm(span=p, adjust=False).mean())
        # calculate Bollinger Bands
        df[f'bb_{price}_{p}'] = df[f'sma_{price}_{p}'] + 2 * df.groupby(['stock_id','date_id'])[price].transform(lambda x: x.rolling(window=p).std())
        # calculate log return
        df[f'log_ret_{price}'] = df.groupby(['stock_id','date_id'])[price].transform(lambda x: np.log(x) - np.log(x.shift(1)))
        # calculate volatility
        df[f'volatility_{price}_{p}'] = df.groupby(['stock_id','date_id'])[f'log_ret_{price}'].transform(lambda x: x.rolling(window=p).std())
        # calculate skewness
        df[f'skewness_{price}_{p}'] = df.groupby(['stock_id','date_id'])[price].transform(lambda x: x.rolling(window=p).skew())
        # calculate kurtosis
        df[f'kurtosis_{price}_{p}'] = df.groupby(['stock_id','date_id'])[price].transform(lambda x: x.rolling(window=p).kurt())
        # calculate autocorrelation
        df[f'autocorr_{price}_{p}'] = df.groupby(['stock_id','date_id'])[price].transform(lambda x: x.rolling(window=p).apply(lambda x: pd.Series(x).autocorr(), raw=False))
    
    return df

@timeit
def imb1_feats(df, cols):
    for i,a in enumerate(cols):
        for j,b in enumerate(cols):
            if (i>j):
                df[f'{a}_{b}_imb'] = df.eval(f'({a}-{b})/({a}+{b})')
    return df

@timeit
def pct_chg_feats(df, cols, periods):
    df_temp = df.groupby(['stock_id','date_id'])[cols].pct_change(periods)
    df_temp.columns += f'_chg{periods}'
    return df.join(df_temp)

@timeit
def diff_feats(df, cols, periods):
    df_temp = df.groupby(['stock_id','date_id'])[cols].diff(periods)
    df_temp.columns += f'_diff{periods}'
    return df.join(df_temp)

@timeit
def imb2_feats(df, cols):
    for i,a in enumerate(cols):
        for j,b in enumerate(cols):
            for k,c in enumerate(cols):
                if i>j and j>k :
                    max_ = df[[a,b,c]].max(axis=1)
                    min_ = df[[a,b,c]].min(axis=1)
                    mid_ = df[[a,b,c]].sum(axis=1)-min_-max_

                    df[f'{a}_{b}_{c}_imb2'] = (max_-mid_)/(max_-min_)
    return df

@timeit
def flow_and_flagged_size_feats(df):

    df['imb_size_flaged'] = df.eval('imbalance_size * imbalance_buy_sell_flag')
    df['mat_size_flaged'] = df.eval('matched_size * imbalance_buy_sell_flag')
    df['bid_size_flaged'] = df.eval('bid_size * imbalance_buy_sell_flag')
    df['ask_size_flaged'] = df.eval('ask_size * imbalance_buy_sell_flag')
    
    df['bid_flow'] = df.groupby(['stock_id','date_id']).apply(
        lambda x: x['bid_price'].gt(x['bid_price'].shift(1)).astype(float).sub(0.5).mul(2).mul(x['bid_size'].shift(1))    
    ).droplevel(['stock_id', 'date_id']).sort_index()
    
    df['ask_flow'] = df.groupby(['stock_id','date_id']).apply(
        lambda x: x['ask_price'].gt(x['ask_price'].shift(1)).astype(float).sub(0.5).mul(2).mul(x['ask_size'].shift(1))    
    ).droplevel(['stock_id', 'date_id']).sort_index()

    df['imb_size_2'] = df.eval('bid_size - ask_size * imb_size_flaged')
    
    return df

 

@timeit
def generate_all_features(df, df_stock_groups):
    
    # Generate group features and index features
    df = group_features(df, df_stock_groups)
    df = group_excess_return(df, 'wap', [1, 2, 5, 10, 50])
    df = idx_feats(df, 'wap')
    df = idx_excess_return(df, 'wap', [1, 2, 5, 10, 50])

    # Generate imbalance features
    prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
              'gp_wap', 'gp_bid_price', 'gp_ask_price', 'idx_wap'
             ]
    sizes = ['imbalance_size', 'matched_size', 'bid_size', 'ask_size']
    
    df = imb1_feats(df, prices)
    df = imb1_feats(df, sizes)
    
    df = imb2_feats(df, ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap'])
    df = imb2_feats(df, ['gp_wap', 'gp_bid_price', 'gp_ask_price'])
    df = imb2_feats(df, ['gp_wap', 'wap', 'idx_wap'])

    # Generate pct change features
    sizes = ['imbalance_size', 'matched_size', 'bid_size', 'ask_size', 'gp_bid_size', 'gp_ask_size']
    df = pct_chg_feats(df, prices + sizes, 1)
    df = pct_chg_feats(df, prices + sizes, 2)
    df = pct_chg_feats(df, prices + sizes, 5)
    df = pct_chg_feats(df, prices + sizes, 10)

    # Generate flow and flagged size features
    df = flow_and_flagged_size_feats(df)

    # Generate diff features
    diff1_cols = ['matched_size_imbalance_size_imb', 'ask_size_bid_size_imb', 'bid_flow', 'ask_flow', 'bid_price_ask_price_imb', 'ask_price_reference_price_imb',
                  'bid_price_far_price_imb', 'wap_near_price_imb', 'wap_ask_price_imb', 'ask_price_near_price_imb',
                  'near_price_reference_price_imb', 'bid_price_reference_price_imb', 'wap_far_price_imb', 'wap_reference_price_imb', 
                  'wap_bid_price_imb', 
                 ]
    df = diff_feats(df, diff1_cols, 1)
    df = technical_indicators_feats(df, 'wap', [10, 50])

    gc.collect()  
    feature_name = [i for i in df.columns if i not in ["row_id", "target", "time_id",  
                                                       "date_id", 'group', 'stock_id', 'seconds_in_bucket',
                                                       ]]
        
    df = df.loc[df['target'].notna(),:]
    
    
    return df[feature_name]


if TRAINING:
    df_train = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')
    df_train = df_train.loc[df_train['date_id']>=400,:]
    df_stock_groups = get_stock_clusters(df_train)
    df_ = generate_all_features(df_train, df_stock_groups)
    print(f"feature number: {len(df_.columns)}")
    with open(f'{out_path}/df_stock_groups.pkl', 'wb') as f:
        pickle.dump(df_stock_groups, f)
    
else:
    with open(f'{model_path}/df_stock_groups.pkl', 'rb') as f:
        df_stock_groups = pickle.load(f)


def technical_indicators_feats(df, price, periods=[10, 50]):
    for p in periods:
        df[[f'sma_{price}_{p}', f'ema_{price}_{p}', f'bb_{price}_{p}', f'log_ret_{price}', f'volatility_{price}_{p}', f'skewness_{price}_{p}', f'kurtosis_{price}_{p}', f'autocorr_{price}_{p}']] = np.nan

    df = df.set_index(['stock_id','date_id'], append=True).unstack(['stock_id','date_id'])
    for p in periods:
        # calculate simple moving average with window size p
        df[f'sma_{price}_{p}'] = df[price].rolling(window=p).mean()
        # calculate exponential moving average with window size p
        df[f'ema_{price}_{p}'] = df[price].ewm(span=p, adjust=False).mean()
        # calculate Bollinger Bands
        df[f'bb_{price}_{p}'] = df[f'sma_{price}_{p}'] + 2 * df[price].rolling(window=p).std()
        # calculate log return
        df[f'log_ret_{price}'] = np.log(df[price]) - np.log(df[price].shift(1))
        # calculate volatility
        df[f'volatility_{price}_{p}'] = df[f'log_ret_{price}'].rolling(window=p).std()
        # calculate skewness
        df[f'skewness_{price}_{p}'] = df[price].rolling(window=p).skew()
        # calculate kurtosis
        df[f'kurtosis_{price}_{p}'] = df[price].rolling(window=p).kurt()
        # calculate autocorrelation
        df[f'autocorr_{price}_{p}'] = df[price].rolling(window=p).apply(lambda x: pd.Series(x).autocorr(), raw=False)
    df = df.stack(['stock_id','date_id']).reset_index(['stock_id','date_id'])
    df[[f'sma_{price}_{p}', f'ema_{price}_{p}', f'bb_{price}_{p}', f'log_ret_{price}', 
        f'volatility_{price}_{p}', f'skewness_{price}_{p}', f'kurtosis_{price}_{p}', 
        f'autocorr_{price}_{p}']] 
        
    
    return df