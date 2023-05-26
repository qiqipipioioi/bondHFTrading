import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import talib
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Heiti TC'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

T2306_df = pd.read_csv('/Users/tuxun/work/jnwork/建模大赛/HFtrading/data/bond/future.csv')
B220220_df = pd.read_csv('/Users/tuxun/work/jnwork/建模大赛/HFtrading/data/bond/lobs.csv')

T2306_df['ts'] = T2306_df['ts'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000))
T2306_df.index = pd.to_datetime(T2306_df['ts'])
T2306_df['midprice'] = (T2306_df['ask1'] + T2306_df['bid1']) / 2
T2306_df = T2306_df[~T2306_df.index.duplicated()]

B220220_df['ts'] = B220220_df['ts'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000))
B220220_df.index = pd.to_datetime(B220220_df['ts'])
B220220_df['midprice'] = (B220220_df['pask1'] + B220220_df['pbid1']) / 2
B220220_df['ymidprice'] = (B220220_df['yask1'] + B220220_df['ybid1']) / 2
B220220_df['yspread'] = B220220_df['ybid1'] - B220220_df['yask1']
B220220_df['pspread'] = B220220_df['pask1'] - B220220_df['pbid1']
B220220_df = B220220_df[~B220220_df.index.duplicated()]

start_time = '2023-03-23 9:30:00'
end_time = '2023-04-19 15:30:00'
T2306_df = T2306_df.loc[start_time: end_time]
B220220_df = B220220_df.loc[start_time: end_time]



def filter_index(df, time_diff):
    selected_index = []
    prev_index = None

    for index, row in df.iterrows():
        if prev_index is None:
            selected_index.append(index)
            prev_index = index
        else:
            time_diff = (index - prev_index).total_seconds()
            if time_diff >= 10:
                selected_index.append(index)
                prev_index = index
            else:
                continue
    return df.loc[selected_index]



def order_signal(x, sig_len):
    before = x[-sig_len:-sig_len//2]
    after = x[-sig_len//2:]
    b = min(after) - max(before)
    s = min(before) - max(after)
    if b >= 0.0049:
        return 1
    elif s >= 0.0049:
        return -1
    else:  
        return 0


def cal_result(T, B, sig_len, after_n):

    T_alldays = None
    n = 0

    for oneday in T['datatime'].apply(lambda x: x.date()).unique():
        t_oneday = T.loc[oneday.strftime('%Y-%m-%d')]
        t_oneday['order_sig'] = t_oneday['midprice'].rolling(sig_len).apply(lambda x: order_signal(x, sig_len))
        if n == 0:
            T_alldays = t_oneday
            n = 1
        else:
            T_alldays = T_alldays.append(t_oneday)

    before_filter_number = len(T_alldays)   
    T_alldays = T_alldays.loc[T_alldays['order_sig'] != 0]
    T_alldays = filter_index(T_alldays, after_n)
    after_filter_number = len(T_alldays)


    B_buy_points = B.iloc[B.index.get_indexer(T_alldays[T_alldays['order_sig'] == 1].index.to_list(), method='nearest')]
    B_buy_points_after = B.iloc[B.index.get_indexer(T_alldays[T_alldays['order_sig'] == 1].index + pd.Timedelta(seconds=after_n), method='nearest')]

    B_sell_points = B.iloc[B.index.get_indexer(T_alldays[T_alldays['order_sig'] == -1].index.to_list(), method='nearest')]
    B_sell_points_after = B.iloc[B.index.get_indexer(T_alldays[T_alldays['order_sig'] == -1].index + pd.Timedelta(seconds=after_n), method='nearest')]

    Buy_point_number = len(B_buy_points) + len(B_sell_points)

    buy_price_delta = B_buy_points_after['midprice'].reset_index() - B_buy_points['midprice'].reset_index()
    buy_price_delta = buy_price_delta['midprice']
    sell_price_delta = B_sell_points['midprice'].reset_index() - B_sell_points_after['midprice'].reset_index()
    sell_price_delta = sell_price_delta['midprice']
    price_delta = buy_price_delta.append(sell_price_delta)
    q_low = price_delta.quantile(0.01)
    q_high = price_delta.quantile(0.99)
    df_filtered = price_delta[(price_delta >= q_low) & (price_delta <= q_high)]
    df_mean = df_filtered.mean()
    df_sum = df_filtered.sum()
    win_rate = df_filtered[df_filtered > 0].count() / df_filtered.count()
    loss_rate = df_filtered[df_filtered < 0].count() / df_filtered.count()
    return df_mean, win_rate, loss_rate, df_sum, before_filter_number, after_filter_number, Buy_point_number


result = []
for sig_len in range(6, 16, 2):
    for after_n in range(5, 60, 5):
        print('sig_len: ', sig_len, 'after_n: ', after_n)
        df_mean, win_rate, loss_rate, df_sum, before_filter_number, after_filter_number, Buy_point_number = cal_result(T2306_df, B220220_df, sig_len, after_n)
        print(df_mean, win_rate, loss_rate, df_sum, before_filter_number, after_filter_number, Buy_point_number)
        result.append([sig_len, after_n, df_mean, win_rate, loss_rate, df_sum, before_filter_number, after_filter_number, Buy_point_number])


result_df = pd.DataFrame(result, columns=['sig_len', 'after_n', 'mean', 'win_rate', 'loss_rate', 'sum', 'before_filter_number', 'after_filter_number', 'Buy_point_number'])

#找出mean最大的行
print('平均值最大的行')
print(result_df[result_df['mean'] == result_df['mean'].max()])

#找出win_rate最大的行
print('胜率最大的行')
print(result_df[result_df['win_rate'] == result_df['win_rate'].max()])

#找出sum最大的行
print('总和最大的行')
print(result_df[result_df['sum'] == result_df['sum'].max()])



#把三个曲面图都画出来, 且给每个曲面加上标题
fig = plt.figure(figsize=(20, 10))
fig.suptitle('寻找最佳参数 sig_len, after_n')
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223, projection='3d')
ax4 = fig.add_subplot(224, projection='3d')

#把三个曲面图的标题分别设置为'loss_rate', 'win_rate', 'mean'
ax1.set_title('loss_rate')
ax2.set_title('win_rate')
ax3.set_title('mean')
ax4.set_title('sum')

#把三个曲面图的x轴, y轴, z轴分别设置为'sig_len', 'after_n', 'loss_rate'
ax1.set_xlabel('sig_len')
ax1.set_ylabel('after_n')
ax1.set_zlabel('loss_rate')


#把三个曲面图的x轴, y轴, z轴分别设置为'sig_len', 'after_n', 'win_rate'
ax2.set_xlabel('sig_len')
ax2.set_ylabel('after_n')
ax2.set_zlabel('win_rate')

#把三个曲面图的x轴, y轴, z轴分别设置为'sig_len', 'after_n', 'mean'
ax3.set_xlabel('sig_len')
ax3.set_ylabel('after_n')
ax3.set_zlabel('mean')

#把三个曲面图的x轴, y轴, z轴分别设置为'sig_len', 'after_n', 'sum'
ax4.set_xlabel('sig_len')
ax4.set_ylabel('after_n')
ax4.set_zlabel('sum')


#把三个曲面图的x轴, y轴, z轴分别设置为'sig_len', 'after_n', 'loss_rate'
ax1.plot_trisurf(result_df['sig_len'], result_df['after_n'], result_df['loss_rate'], cmap='rainbow')

#把三个曲面图的x轴, y轴, z轴分别设置为'sig_len', 'after_n', 'win_rate'
ax2.plot_trisurf(result_df['sig_len'], result_df['after_n'], result_df['win_rate'], cmap='rainbow')

#把三个曲面图的x轴, y轴, z轴分别设置为'sig_len', 'after_n', 'mean'
ax3.plot_trisurf(result_df['sig_len'], result_df['after_n'], result_df['mean'], cmap='rainbow')

#把三个曲面图的x轴, y轴, z轴分别设置为'sig_len', 'after_n', 'sum'
ax4.plot_trisurf(result_df['sig_len'], result_df['after_n'], result_df['sum'], cmap='rainbow')


plt.show()

fig.savefig('/Users/tuxun/work/jnwork/建模大赛/HFtrading/寻找最佳参数_micro_sig_len_after_n.png')

