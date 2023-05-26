from typing import List, Optional, Tuple, Union, Dict
from collections import deque

import numpy as np
import pandas as pd

from simulator import MdUpdate, Order, OwnTrade, Sim, update_best_positions


class BestPosStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, delay: float, hold_time:Optional[float], min_pos = 0.001, window=8) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time

        self.min_pos = min_pos
        
        self.window = window
        
        self.last_best_p = None
        
        
    def order_signal(self, x, sig_len):
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


    def run(self, sim: Sim ) ->\
        Tuple[List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order]]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''
        future_data_q = deque(maxlen=self.window)
        last_future_ts = None
        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        best_bid = np.inf
        best_ask = -np.inf
        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    if update.future:
                        if last_future_ts == None:
                            last_future_ts = receive_ts
                        elif receive_ts - last_future_ts >= 60:
                            last_future_ts = receive_ts
                            future_data_q = deque(maxlen=self.window)
                        else:
                            future_data_q.append((update.future.ask1 + update.future.bid1)/2)
                            last_future_ts = receive_ts
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'
                    
                                       
            if best_ask in (np.inf, -np.inf) or best_bid in (np.inf, -np.inf):
                continue
            
            if self.last_best_p is None:
                self.last_best_p = (best_ask + best_bid) / 2  
            
            if abs((best_ask + best_bid) / 2 - self.last_best_p) / self.last_best_p >= 0.02:
                continue
            else:
                self.last_best_p = (best_ask + best_bid) / 2
                    
            #对主动发起的单向订单进行平仓
            keys = list(sim.wait_close_orders.keys())
            for k in keys:
                if sim.wait_close_orders[k][1] == 1:
                    side = sim.wait_close_orders[k][0].side
                    q = sim.wait_close_orders[k][0].size
                    if side == 'BID':
                        ask_order = sim.place_order(receive_ts, q, 'ASK', best_ask, 'close', k)
                        ongoing_orders[ask_order.order_id] = ask_order
                        all_orders += [ask_order]
                    elif side == 'ASK':
                        bid_order = sim.place_order(receive_ts, q, 'BID', best_bid, 'close', k)
                        ongoing_orders[bid_order.order_id] = bid_order
                        all_orders += [bid_order]
                    sim.wait_close_orders.pop(k)        
            
            if receive_ts - prev_time >= self.delay:
                #place order
                if len(future_data_q) == self.window:
                    sig = self.order_signal(list(future_data_q), self.window)
                    if sig == 1:
                        bid_order = sim.place_order(receive_ts, self.min_pos, 'BID', best_bid, 'open')
                        ongoing_orders[bid_order.order_id] = bid_order
                        all_orders += [bid_order]
                        sim.wait_close_orders[bid_order.order_id] = [bid_order, 0]
                        prev_time = receive_ts
                    elif sig == -1:
                        ask_order = sim.place_order(receive_ts, self.min_pos, 'ASK', best_ask, 'open')
                        ongoing_orders[ask_order.order_id] = ask_order
                        all_orders += [ask_order]
                        sim.wait_close_orders[ask_order.order_id] = [ask_order, 0]
                        prev_time = receive_ts
                    # else:
                    #     bid_order = sim.place_order( receive_ts, self.min_pos, 'BID', best_bid )
                    #     ask_order = sim.place_order( receive_ts, self.min_pos, 'ASK', best_ask )
                    #     ongoing_orders[bid_order.order_id] = bid_order
                    #     ongoing_orders[ask_order.order_id] = ask_order
                    #     all_orders += [bid_order, ask_order]
            
            to_cancel = []
            add_pair = []
            
            for ID, order in ongoing_orders.items():
                if order.order_type == 'open':
                    if order.place_ts < receive_ts - self.hold_time:
                        sim.cancel_order(receive_ts, ID)
                        to_cancel.append(ID)
                elif order.order_type == 'close':
                    if order.place_ts < receive_ts - self.hold_time:
                        sim.cancel_order(receive_ts, ID)
                        to_cancel.append(ID)
                        if order.side == 'BID':
                            bid_order = sim.place_order(receive_ts, order.size, 'BID', best_ask, 'close', order.compaire_id)
                            add_pair.append([bid_order.order_id, bid_order])
                            all_orders += [bid_order]
                        elif order.side == 'ASK':
                            ask_order = sim.place_order(receive_ts, order.size, 'ASK', best_bid, 'close', order.compaire_id)
                            add_pair.append([ask_order.order_id, ask_order])
                            all_orders += [ask_order]
                            
            for p in add_pair:
                ongoing_orders[p[0]] = p[1]
                
            for ID in to_cancel:
                ongoing_orders.pop(ID)
                
        return trades_list, md_list, updates_list, all_orders
