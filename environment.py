from enum import Enum
import numpy as np


class Position(Enum):
    LONG = 0
    NONE = 1
    SHORT = 2

class FEE(Enum):
    TRADING = 0.0005    # Binance Trading fee : 0.05%
    SLIPPAGE = 0.0005   # slippage : 0.05% per trade

class Action(Enum):
    LONG = 0
    HOLD = 1
    SHORT = 2



class Environment():
    # Agent Balance State : [포지션/자금 비율, 손익, 평균 수익률, 현재 포지션]
    B_STATE_DIM = 4
    NUM_ACTIONS = len(Action)
    CLOSE_PRICE_IDX = 4

    def __init__(self, chart_data, trading_data, initial_balance, min_trading_budget, max_trading_budget):
        # 초기 자본금 설정
        self.initial_balance = initial_balance

        # 최대/최소 단일 매매 금액 설정. (최소 : 70 달러)
        self.min_trading_budget = min_trading_budget
        self.max_trading_budget = max_trading_budget

        # chart 정보
        self.chart_data = chart_data
        self.trading_data = trading_data
        self.observation = None
        self.idx = -1

        # balance : 잔고 내역 및 거래 정보
        self.balance = initial_balance   # 현재 현금 잔고
        self.num_stocks = 0              # 보유 주식 수
        self.portfolio_value = 0         # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.num_long = 0                # 매수 횟수
        self.num_short = 0               # 매도 횟수
        self.num_hold = 0                # 관망 횟수

        # balance : agent의 state 정보
        self.hold_ratio = 0              # 주식 보유 비율
        self.profitloss = 0              # 현재 손익
        self.avg_position_price = 0      # 주당 매수 단가
        self.position = 1                # 현재 포지션 (0 : Long, 1 : None, 2 : Short)


    def reset(self):
        self.observation = None
        self.idx = -1
        self.balance = self.initial_balance
        self.portfolio_value = self.portfolio_value
        self.num_stocks = 0
        self.hold_ratio = 0.0
        self.profitloss = 0.0
        self.avg_position_price = 0.0
        self.position = 1
        self.num_long = 0
        self.num_short = 0
        self.num_hold = 0

    def observe(self):
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None
    
    def get_price(self):
        return self.observation[self.CLOSE_PRICE_IDX]
    
    # 결정된 Action(Long, Short)을 수행할 수 있는 최소 조건을 확인.
    def validate_action(self, action):
        if action == Action.LONG.value:
            # 숏 포지션 일 때 : Position Value로 확인 
            if self.position == Position.SHORT.value:
                if self.portfolio_value < self.min_trading_budget * (1 + FEE.TRADING.value):
                    return False
            
            # 롱 or 무 포지션 일 때 : Balance로 구매할 수 있는 자금 확인.
            else:
                # 적어도 1주(거래 최소 금액)를 살 수 있는지 확인
                if self.balance < self.min_trading_budget * (1 + FEE.TRADING.value):
                    return False
        
        elif action == Action.SHORT.value:
            # 롱 or 무 포지션 일 때 : Position Value로 구매할 수 있는 자금 확인.
            if self.position == Position.LONG.value:
                if self.portfolio_value < self.min_trading_budget * (1 + FEE.TRADING.value):
                    return False
            # 숏 포지션 일 때 : Balance로 공매도 할 수 있는 자금 확인.
            else: 
                # 적어도 1주(거래 최소 금액)를 팔 수 있는지 확인
                if self.balance < self.min_trading_budget * (1 + FEE.TRADING.value):
                    return False
        return False
    
    # Action을 수행할 수 있을 때 진입 포지션의 양을 반환해주는 함수.
    def decide_trading_unit(self, confidence):
        # [TODO] 이거 logging 하자. 얼마나 nan 많이 나오는지 체크해야 함.
        if np.isnan(confidence):
            trading_qty = self.min_trading_budget/self.get_price()
            return  round(max(trading_qty, 0), 4)
        
        added_trading_budget = max(min(confidence*(self.max_trading_budget - self.min_trading_budget),
                                                    self.max_trading_budget - self.min_trading_budget),0)
        trading_budget = self.min_trading_budget + added_trading_budget
        trading_qty = trading_budget/self.get_price()
        return round(max(trading_qty, 0), 4)

    
    

            
    



