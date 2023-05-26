import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import talib

T2306_df1 = pd.read_csv('/Users/tuxun/work/jnwork/建模大赛/HFtrading/data/future/T2306.CFE_20230228_20230306_new.csv')
T2306_df2 = pd.read_csv('/Users/tuxun/work/jnwork/建模大赛/HFtrading/data/future/T2306.CFE_20230323_20230331_new.csv')
T2306_df = T2306_df1.append(T2306_df2)
B220220_df = pd.read_csv('/Users/tuxun/work/jnwork/建模大赛/HFtrading/data/bond/220220_t1_cash_bond_ano_odm_cfets_xbond_orderbook_tick.csv')
B220025_df = pd.read_csv('/Users/tuxun/work/jnwork/建模大赛/HFtrading/data/bond/220025_t1_cash_bond_ano_odm_cfets_xbond_orderbook_tick.csv')


T2306_df = T2306_df.rename(columns={'Unnamed: 0':'datatime'})
T2306_df['datatime'] = T2306_df['datatime'].apply(lambda x: datetime.datetime.strptime(x, r'%Y-%m-%d %H:%M:%S.%f'))
T2306_df.index = pd.to_datetime(T2306_df['datatime'])
T2306_df['midprice'] = (T2306_df['ASK1'] + T2306_df['BID1']) / 2


B220220_df = B220220_df.rename(columns={'ts':'datatime'})
B220220_df['datatime'] = B220220_df['datatime'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000))
B220220_df.index = pd.to_datetime(B220220_df['datatime'])
B220220_df['midprice'] = (B220220_df['pask1'] + B220220_df['pbid1']) / 2


B220025_df = B220025_df.rename(columns={'ts':'datatime'})
B220025_df['datatime'] = B220025_df['datatime'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000))
B220025_df.index = pd.to_datetime(B220025_df['datatime'])
B220025_df['midprice'] = (B220025_df['pask1'] + B220025_df['pbid1']) / 2


date_str = '2023-03-01'
start_time = date_str + ' 10:53:00'
end_time = date_str + ' 10:54:00'
part_t = T2306_df.loc[start_time:end_time]
part_b02 = B220220_df.loc[start_time:end_time]
part_b00 = B220025_df.loc[start_time:end_time]


#以同一条时间轴画出t0301和b0301的折线图
fig, ax1 = plt.subplots(figsize=(20, 10))
ax2 = ax1.twinx()
ax1.plot(part_t['midprice'], 'r-', marker='o', markersize=2, linewidth=0.5)
ax2.plot(part_b02['midprice'], 'g-', marker='o', markersize=2, linewidth=0.5)
#加上图例
ax1.legend(['future'], loc=2)
ax2.legend(['bond'], loc=1)
#加上标题
ax1.set_title('future and bond midprice')

plt.show()