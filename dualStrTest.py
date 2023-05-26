from simulator import MdUpdate, Order, OwnTrade, Sim
from dualStrategy import BestPosStrategy
from get_info import get_pnl
from load_data import load_md_from_file
import datetime
import talib


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


plt.rcParams['font.sans-serif'] = ['Heiti TC'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


path = '/Users/tuxun/work/jnwork/建模大赛/HFtrading/data/bond/'

begin_time = "2023-03-23 09:30:00.0"
end_time = "2023-04-18 15:15:00.0"


md = load_md_from_file(path, begin_time, end_time)

latency = 0
md_latency = 0

sim = Sim(md, latency, md_latency)

#delay between orders
delay = 30
hold_time = 30
min_pos = 100

strategy = BestPosStrategy(delay, hold_time, min_pos=min_pos)

trades_list, md_list, updates_list, all_orders = strategy.run(sim)

print(len(trades_list))
print(len(all_orders))
df = get_pnl(updates_list, ifcost=True)

print(df)



df['exchange_ts'] = df['exchange_ts'].apply(lambda x: datetime.datetime.fromtimestamp(x))
df['receive_ts'] = df['receive_ts'].apply(lambda x: datetime.datetime.fromtimestamp(x))

dt = pd.to_datetime(df.receive_ts)

#画出dt为x轴，y轴为total的曲线，时间不需要连续


df = df.resample('1D', on='receive_ts').last()
df = df.dropna()

#计算夏普和最大回撤
sharpe_ratio = (252 ** 0.5) * df['total'].mean() / df['total'].std()

# 计算收益率
df['returns'] = df['total'].pct_change()

# 计算累计收益率
df['cum_returns'] = (1 + df['returns']).cumprod()

# 计算最大回撤
cum_max = df['cum_returns'].cummax()
drawdown = (df['cum_returns'] - cum_max) / cum_max

max_drawdown = drawdown.min()

# 打印计算结果
print('sharp ratio:', sharpe_ratio)
print('Max Drawdown:', max_drawdown)


dt = pd.to_datetime(df.receive_ts).apply(lambda x: x.strftime('%Y-%m-%d'))
dt = dt.tolist()

print(len(dt))

plt.figure(figsize=(10, 5))
plt.xticks(list(range(len(dt))), dt, rotation=30)
plt.plot(dt, df.total, 'r-', marker='o', markersize=2, linewidth=0.5)
plt.ylabel("PnL", fontsize=13)
plt.title("双边挂单策略 vol=100 PnL", fontsize=15)
plt.savefig('双边挂单策略 vol=100 PnL.png', dpi=300)
#plt.show()

'''


#日内

plt.figure(figsize=(10, 5))
plt.plot(df.receive_ts, df.total, 'r-', linewidth=0.5)
plt.ylabel("PnL", fontsize=13)
plt.title("双边挂单策略  vol=10 日内 PnL", fontsize=15)
plt.savefig('双边挂单策略  vol=10 日内 PnL.png', dpi=300)
#plt.show()
'''






