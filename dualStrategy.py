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
    def __init__(self, delay: float, hold_time:Optional[float], min_pos = 0.001) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time
        
        self.pair_id = 1

        self.min_pos = min_pos
        

        
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
                        
            
            if receive_ts - prev_time >= self.delay:
                #check order status
                keys = list(sim.wait_close_orders.keys())
                assert len(keys) <= 2, 'invalid number of orders'
                excuted_keys = []
                not_excuted_keys = []
                for k in keys:
                    if sim.wait_close_orders[k][1] == 1:
                        excuted_keys.append(k)
                    else:
                        not_excuted_keys.append(k)
                        
                #都没有成交，则全部取消
                if len(not_excuted_keys) == 2:
                    for k in not_excuted_keys:
                        sim.cancel_order(receive_ts, k)
                        sim.wait_close_orders.pop(k)
                        ongoing_orders.pop(k)
                #有一个成交，则另一个强平
                elif len(excuted_keys) == 1 and len(not_excuted_keys) == 1:
                    not_excuted_id = not_excuted_keys[0]
                    side = sim.wait_close_orders[not_excuted_id][0].side
                    q = sim.wait_close_orders[not_excuted_id][0].size
                    
                    sim.wait_close_orders.pop(excuted_keys[0])
                    
                    sim.cancel_order(receive_ts, not_excuted_id)
                    sim.wait_close_orders.pop(not_excuted_id)
                    ongoing_orders.pop(not_excuted_id)
                    
                    if side == 'ASK':
                        ask_order = sim.place_order(receive_ts, q, 'ASK', best_bid, 'close', not_excuted_id)
                        ongoing_orders[ask_order.order_id] = ask_order
                        all_orders += [ask_order]
                    elif side == 'BID':
                        bid_order = sim.place_order(receive_ts, q, 'BID', best_ask, 'close', not_excuted_id)
                        ongoing_orders[bid_order.order_id] = bid_order
                        all_orders += [bid_order]
                elif len(excuted_keys) == 2:
                    for k in excuted_keys:
                        sim.wait_close_orders.pop(k)
                
                
                #新的双边订单
                bid_order = sim.place_order( receive_ts, self.min_pos, 'BID', best_bid, 'open', self.pair_id )
                ask_order = sim.place_order( receive_ts, self.min_pos, 'ASK', best_ask, 'open', self.pair_id )
                self.pair_id += 1
                sim.wait_close_orders[bid_order.order_id] = [bid_order, 0]
                sim.wait_close_orders[ask_order.order_id] = [ask_order, 0]
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order
                all_orders += [bid_order, ask_order]
                prev_time = receive_ts
            

                
        return trades_list, md_list, updates_list, all_orders
