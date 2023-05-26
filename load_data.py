from typing import List
import datetime
import time
import random

import pandas as pd

from simulator import AnonTrade, MdUpdate, OrderbookSnapshotUpdate, FutureSnapshotUpdate



def load_before_time(path, begin_time, end_time):
    chunksize = 10 ** 5
    chunks = []
    #begin_time str转timestamp
    begin_time = datetime.datetime.strptime(begin_time, '%Y-%m-%d %H:%M:%S.%f')
    begin_timestamp = int(begin_time.timestamp())
    end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S.%f')
    end_timestamp = int(end_time.timestamp())
    
    
    for chunk in pd.read_csv(path, chunksize=chunksize): 
        chunks.append(chunk)
        t1 = chunk['ts'].iloc[-1]
        t1 = t1 / 1000  
        if t1 >= end_timestamp:
            break
    df = pd.concat(chunks)
    df['ts'] = df['ts'].apply(lambda x: x / 1000)
    mask = df['ts'].apply(lambda x: x >= begin_timestamp and x <= end_timestamp)
 
    df = df.loc[mask]
    
    df['datatime'] = df['ts'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    df.index = pd.to_datetime(df['datatime'])
    df = df.between_time('9:30', '11:30').append(df.between_time('13:30', '15:15'))
    #去掉非工作日的数据
    df = df[df['datatime'].apply(lambda x: x.weekday() < 5)]
    #去掉3月28日数据
    df = df[df['datatime'].apply(lambda x: x.date() != datetime.date(2023, 3, 28))]
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['datatime'], inplace=True)
    print(df)
    
    return df


def load_future(path:str, begin_time, end_time) -> List[FutureSnapshotUpdate]:
    '''
        This function downloads future data

        Args:
            path(str): path to file
            T(int): max timestamp from the first one in nanoseconds

        Return:
            trades(List[AnonTrade]): list of trades 
    '''
    future = load_before_time(path + 'future.csv', begin_time, end_time)
    
    future['receive_ts'] = future['ts']
    future['exchange_ts'] = future['ts'] - 0.003

    future = future[ ['exchange_ts', 'receive_ts', 'ask1', 'bid1', 'volume' ] ].sort_values(["exchange_ts", 'receive_ts'])
    receive_ts = future.receive_ts.values
    exchange_ts = future.exchange_ts.values 
    future = [ FutureSnapshotUpdate(*args) for args in future.values]
    return future


def load_trades(path:str, begin_time, end_time) -> List[AnonTrade]:
    '''
        This function downloads trades data

        Args:
            path(str): path to file
            T(int): max timestamp from the first one in nanoseconds

        Return:
            trades(List[AnonTrade]): list of trades 
    '''
    trades = load_before_time(path + 'trades.csv', begin_time, end_time)
    
    #переставляю колонки, чтобы удобнее подавать их в конструктор AnonTrade
    trades['receive_ts'] = trades['ts']
    trades['exchange_ts'] = trades['ts'] - 0.003
    trades['size'] = trades['qty']
    trades['price'] = trades['ytm']
    trades = trades[ ['exchange_ts', 'receive_ts', 'size', 'price' ] ].sort_values(["exchange_ts", 'receive_ts'])
    receive_ts = trades.receive_ts.values
    exchange_ts = trades.exchange_ts.values 
    trades = [ AnonTrade(*args) for args in trades.values]
    return trades


def load_books(path:str, begin_time, end_time) -> List[OrderbookSnapshotUpdate]:
    '''
        This function downloads orderbook market data

        Args:
            path(str): path to file
            T(int): max timestamp from the first one in nanoseconds

        Return:
            books(List[OrderbookSnapshotUpdate]): list of orderbooks snapshots 
    '''
    lobs   = load_before_time(path + 'lobs.csv', begin_time, end_time)
    
    #rename columns
    # names = lobs.columns.values
    # ln = len('btcusdt:Binance:LinearPerpetual_')
    # renamer = { name:name[ln:] for name in names[2:]}
    # renamer[' exchange_ts'] = 'exchange_ts'
    # lobs.rename(renamer, axis=1, inplace=True)
    lobs['receive_ts'] = lobs['ts']
    lobs['exchange_ts'] = lobs['ts'] - 0.003
    #timestamps
    receive_ts = lobs.receive_ts.values
    exchange_ts = lobs.exchange_ts.values 
    #список ask_price, ask_vol для разных уровней стакана
    #размеры: len(asks) = 10, len(asks[0]) = len(lobs)
    asks = [list(zip(lobs[f"yask{i}"],lobs[f"vask{i}"])) for i in range(1, 11)]
    #транспонируем список
    asks = [ [asks[i][j] for i in range(len(asks))] for j in range(len(asks[0]))]
    #тоже самое с бидами
    bids = [list(zip(lobs[f"ybid{i}"],lobs[f"vbid{i}"])) for i in range(1, 6)]
    bids = [ [bids[i][j] for i in range(len(bids))] for j in range(len(bids[0]))]
    
    books = list( OrderbookSnapshotUpdate(*args) for args in zip(exchange_ts, receive_ts, asks, bids) )
    return books


def merge_books_and_trades(books : List[OrderbookSnapshotUpdate], trades: List[AnonTrade], future: List[FutureSnapshotUpdate]) -> List[MdUpdate]:
    '''
        This function merges lists of orderbook snapshots and trades 
    '''
    trades_dict = { (trade.exchange_ts, trade.receive_ts) : trade for trade in trades }
    books_dict  = { (book.exchange_ts, book.receive_ts) : book for book in books }
    future_dict = { (f.exchange_ts, f.receive_ts) : f for f in future }
    
    ts = sorted(trades_dict.keys() | books_dict.keys() | future_dict.keys())

    md = [MdUpdate(*key, books_dict.get(key, None), trades_dict.get(key, None), future_dict.get(key, None) ) for key in ts]
    return md


def load_md_from_file(path: str, begin_time, end_time) -> List[MdUpdate]:
    '''
        This function downloads orderbooks ans trades and merges them
    '''
    books  = load_books(path, begin_time, end_time)
    trades = load_trades(path, begin_time, end_time)
    future = load_future(path, begin_time, end_time)
    return merge_books_and_trades(books, trades, future)