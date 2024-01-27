import numpy as np


class Position:
    LONG = 0
    NONE = 1
    SHORT = 2

class FEE:
    TRADING = 0.0005    # Binance Trading fee : 0.05%
    SLIPPAGE = 0.0005   # slippage : 0.05% per trade

class Action:
    LONG = 0
    HOLD = 1
    SHORT = 2



class Environment():
    # Agent Balance State : [포지션/자금 비율, 손익, 평균 수익률, 현재 포지션]
    B_STATE_DIM = 4
    NUM_ACTIONS = len([Action.LONG, Action.HOLD, Action.SHORT])
    CLOSE_PRICE_IDX = 4

    def __init__(self, chart_data, training_data, initial_balance, min_trading_budget, max_trading_budget):
        # 초기 자본금 설정
        self.initial_balance = initial_balance

        # 최대/최소 단일 매매 금액 설정. (최소 : 70 달러)
        self.min_trading_budget = min_trading_budget
        self.max_trading_budget = max_trading_budget

        # chart 정보
        self.chart_data = chart_data
        self.training_data = training_data
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
        if action == Action.LONG:
            # 숏 포지션 일 때 : Position Value로 확인 
            if self.position == Position.SHORT:
                if self.portfolio_value < self.min_trading_budget * (1 + FEE.TRADING + FEE.SLIPPAGE):
                    return False
            
            # 롱 or 무 포지션 일 때 : Balance로 구매할 수 있는 자금 확인.
            else:
                # 적어도 1주(거래 최소 금액)를 살 수 있는지 확인
                if self.balance < self.min_trading_budget * (1 + FEE.TRADING + FEE.SLIPPAGE):
                    return False
        
        elif action == Action.SHORT:
            # 롱 or 무 포지션 일 때 : Position Value로 구매할 수 있는 자금 확인.
            if self.position == Position.LONG:
                if self.portfolio_value < self.min_trading_budget * (1 + FEE.TRADING + FEE.SLIPPAGE):
                    return False
            # 숏 포지션 일 때 : Balance로 공매도 할 수 있는 자금 확인.
            else: 
                # 적어도 1주(거래 최소 금액)를 팔 수 있는지 확인
                if self.balance < self.min_trading_budget * (1 + FEE.TRADING + FEE.SLIPPAGE):
                    return False
        return False
    
    # Action을 수행할 수 있을 때 진입 포지션의 양을 반환해주는 함수.
    def decide_trading_unit(self, confidence):
        # [TODO] 이거 logging 하자. 얼마나 nan 많이 나오는지 체크해야 함.
        if np.isnan(confidence):
            trading_unit = self.min_trading_budget/self.get_price()
            return  round(max(trading_unit, 0), 4)
        
        added_trading_budget = max(min(confidence*(self.max_trading_budget - self.min_trading_budget),
                                                    self.max_trading_budget - self.min_trading_budget),0)
        trading_budget = self.min_trading_budget + added_trading_budget
        trading_unit = trading_budget/self.get_price()
        return round(max(trading_unit, 0), 4)

    
    # action을 수행하고 환경 정보를 업데이트
    # Input : (action, confidence), Output : (reward(self.profitloss), trading_unit)
    def act(self, action, confidence):
        trading_unit = 0
        # Action을 수행할 수 있는지 잔고 확인 : 수행할 수 없다면 HOLD 포지션
        if not self.validate_action(action):
            action = Action.HOLD
        curr_price = self.get_price()

        # (1) Position : NONE 
        if self.position == Position.NONE:
            # (1.1) Action : HOLD
            if action == Action.HOLD:
                self.num_hold += 1
            
            # (1.2) Action : LONG or SHORT
            else:
                # (1.2.1) 진입 유닛 설정
                trading_unit = self.decide_trading_unit(confidence)
                # (1.2.2) 보유 현금 검증
                remain_balance = self.balance - (curr_price * (1 + FEE.TRADING + FEE.SLIPPAGE) * trading_unit)
                # (1.2.3) 보유 현금 부족시 금액&진입 유닛 재산정.
                if remain_balance < 0:
                    possible_budget = min(self.balance, self.max_trading_budget)
                    trading_unit = round(possible_budget/(curr_price * (1 + FEE.TRADING + FEE.SLIPPAGE)), 4)
                # (1.2.4) 진입 금액 산정
                trading_budget = curr_price * (1 + FEE.TRADING + FEE.SLIPPAGE) * trading_unit 
                # (1.2.5) 진입 금액 존재시 정보 갱신
                if trading_budget > 0:
                    self.avg_position_price = curr_price
                    self.balance -= trading_budget
                    # (1.2.5.1) Action : LONG
                    if action == Action.LONG:
                        self.num_stocks += trading_unit
                        self.num_long += 1
                    # (1.2.5.2) Action : SHORT
                    elif action == Action.SHORT:
                        self.num_stocks -= trading_unit
                        self.num_short += 1

        # (2) Position : LONG
        elif self.position == Position.LONG:
            # (2.1) Action : HOLD
            if action == Action.HOLD:
                self.num_hold += 1
            
            # (2.2) Action : LONG
            elif action == Action.LONG:
                trading_unit = self.decide_trading_unit(confidence)                                             # (2.2.1) 진입 유닛 설정 
                remain_balance = self.balance - (curr_price * (1 + FEE.TRADING + FEE.SLIPPAGE)*trading_unit)    # (2.2.2) 보유 현금 검증

                if remain_balance < 0:                                                                          # (2.2.3) 보유 현금 부족시 진입 금액&유닛 재산정
                    possible_budget = min(self.balance, self.max_trading_budget)
                    trading_unit = round(possible_budget/(curr_price * (1 + FEE.TRADING + FEE.SLIPPAGE)), 4)    # (2.2.4) 진입 금액 산정

                trading_budget = curr_price * (1 + FEE.TRADING + FEE.SLIPPAGE) * trading_unit 

                if trading_budget > 0:
                    self.avg_position_price = (self.avg_position_price * self.num_stocks + curr_price * trading_unit) \
                                                / (self.num_stocks + trading_unit)  # 평균 포지션 가격 업데이트
                    self.balance -= trading_budget                                  # 보유 현금을 갱신
                    self.num_long += 1                                              # long 횟수 증가
                    self.num_stocks += trading_unit                                 # 보유 주식수 추가

            # (2.3) Action : SHORT
            elif action == Action.SHORT:
                trading_unit = self.decide_trading_unit(confidence)         # (2.3.1) 진입 유닛 결정
                remain_unit = self.num_stocks - trading_unit                # (2.3.2) 보유 물량 검증
                
                # (2.3.3) 보유 물량 부족 -> 보유 현금 확인
                if remain_unit < 0:
                    # (2.3.3.2) 잔여 balance 계산(= 기존 Balance + 자산 청산 금액 - 신규 공매도 금액)
                    asset_sell_amount = (curr_price * (1 - (FEE.TRADING + FEE.SLIPPAGE)) * self.num_stocks)
                    remain_balance = self.balance + asset_sell_amount - (curr_price * (1 + FEE.TRADING + FEE.SLIPPAGE)) * abs(remain_unit)

                    # (2.3.3.1) 보유 현금 부족시 진입 금액&유닛 재설정.
                    if remain_balance < 0:
                        possible_budget = min(self.balance + asset_sell_amount, self.max_trading_budget)
                        trading_unit = round(possible_budget / (curr_price * (1 + FEE.TRADING + FEE.SLIPPAGE)), 4)
                    # (2.3.3.2) 기존 물량(self.num_stocks) 모두 매도 + 보유 현금으로 나머지 Short 포지션 진입
                    trading_budget = curr_price * (1 + FEE.TRADING + FEE.SLIPPAGE) * trading_unit
                    self.balance = self.balance + asset_sell_amount - trading_budget
                
                # (2.3.4) 보유 물량 부족 X
                else:
                    trading_budget = curr_price * (1 - (FEE.TRADING + FEE.SLIPPAGE)) * trading_unit
                    self.balance += trading_budget
                
                # (2.3.5) 진입 금액 존재시 정보 업데이트
                if trading_budget > 0:
                    self.num_stocks -= trading_unit
                    # 평균 포지션 가격 & 보유 현금 업데이트
                    if self.num_stocks > 0:
                        self.avg_position_price = self.avg_position_price
                    elif self.num_stocks < 0:
                        self.avg_position_price = curr_price
                    else:
                        self.avg_position_price = 0
                    
                    self.num_short += 1
        
        # (3) Position : SHORT
        elif self.position == Position.SHORT:
            # Action : HOLD
            if action == Action.HOLD:
                self.num_hold += 1
            
            # Action : LONG
            if action == Action.LONG :
                # (3.2.1) 진입 유닛 설정 
                trading_unit = self.decide_trading_unit(confidence)
                # (3.2.2) 공매도 물량 초과 확인
                remain_unit = self.num_stocks + trading_unit
                # (3.2.3) 공매도 물량 소진 O + 초과 매수
                if remain_unit > 0 :
                    # [TODO] 변경함. (3.2.3.1) 잔여 balance 계산(=기존 balance + 공매도 포지션 정리 금액 - remain_unit * 현재 가격)
                    asset_buy_amount = (2*self.avg_position_price - curr_price) * (1 - (FEE.TRADING + FEE.SLIPPAGE)) * abs(self.num_stocks)
                    remain_balance = self.balance + asset_buy_amount - (curr_price * (1 + (FEE.TRADING + FEE.SLIPPAGE)) * remain_unit)
                    # (3.2.3.2) 보유 현금 부족시 진입 금액&유닛 재산정
                    if remain_balance < 0 :
                        possible_budget = min(self.balance + asset_buy_amount, self.max_trading_budget)
                        trading_unit = round(possible_budget / (curr_price * (1 + (FEE.TRADING + FEE.SLIPPAGE))), 4)
                    # (3.2.3.3) 기존 공매도 물량(self.num_stocks) 모두 매수 + 보유 현금으로 나머지 Long 포지션 진입
                    trading_budget = curr_price * (1 + (FEE.TRADING + FEE.SLIPPAGE)) * trading_unit
                    self.balance = self.balance + asset_buy_amount - trading_budget  # balance 정보 갱신(*)
                # (3.2.4) 공매도 물량 소진 X
                else :
                    asset_buy_amount =  (2*self.avg_position_price - curr_price) * (1 - (FEE.TRADING + FEE.SLIPPAGE)) * trading_unit
                    self.balance += asset_buy_amount  # balance 정보 갱신(*)

                # (3.2.5) 진입 금액 존재시 정보 갱신
                if trading_budget > 0 :
                    self.num_stocks += trading_unit         # 보유 주식수 추가
                    # 평균 포지션 가격 & 보유 현금 갱신
                    if self.num_stocks < 0 :
                        self.avg_position_price = self.avg_position_price
                    elif self.num_stocks > 0 :
                        self.avg_position_price = curr_price
                    else :
                        self.avg_position_price = 0
                    self.num_long += 1                      # long 횟수 증가
                

            # (3.2) SHORT 진입
            elif action == Action.SHORT :
                trading_unit = self.decide_trading_unit(confidence)                                              # (3.1.1) 진입 유닛 설정 
                remain_balance = self.balance - (curr_price * (1 + FEE.TRADING + FEE.SLIPPAGE) * trading_unit)   # (3.1.2) 보유 현금 검증

                # (3.1.3) 보유 현금 부족시 진입 금액&유닛 재산정
                if remain_balance < 0 :
                    possible_budget = min(self.balance, self.max_trading_budget)
                    trading_unit = round(possible_budget / (curr_price * (1 + FEE.TRADING + FEE.SLIPPAGE)), 4)
                
                trading_budget = curr_price * (1 + FEE.TRADING + FEE.SLIPPAGE) * trading_unit                    # (3.1.4) 진입 금액 산정
                # (3.1.5) 진입 금액 존재시 정보 갱신
                if trading_budget > 0 :
                    self.avg_position_price = (self.avg_position_price * abs(self.num_stocks) + curr_price * trading_unit) \
                                                / (abs(self.num_stocks) + trading_unit)      # 평균 포지션 가격 업데이트
                    self.balance -= trading_budget                                      # 보유 현금을 갱신
                    self.num_short += 1                                                 # short 횟수 증가
                    self.num_stocks -= trading_unit                                     # 보유 주식수 차감

        # (4) 포지션 업데이트
        if self.num_stocks > 0:
            self.postion = Position.LONG
        elif self.num_stocks < 0:
            self.position = Position.SHORT
        else:
            self.position = Position.NONE
        
        # (5) 포트폴리오 가치 갱신
        if self.position in [Position.LONG, Position.NONE]:
            self.portfolio_value = self.balance + curr_price * abs(self.num_stocks)
        else:
            self.portfolio_value = self.balance + (2*self.avg_position_price - curr_price) * abs(self.num_stocks)

        # (6) 손익 갱신
        self.profitloss = self.portfolio_value / self.initial_balance -1

        # (7) 포지션 보유 비율 갱신
        self.hold_ratio = (self.portfolio_value - self.balance) / self.portfolio_value
        return self.profitloss, trading_unit
        


    # Input : Agent's action space | Output : (Observed Chart, Observed Balance, Reward, Done, Info)
    def step(self, action=None, policy=None):
        observation = self.observe()
        # 다음 훈련 데이터가 없을 경우.
        if (np.array(observation) == None).all():
            done = True
            return None, None, 0, done, None
        
        # 훈련 시작 전 초기 데이터 반환
        if action == None:
            reward, avg_return, done  = 0, 0, False
            chart_next_state = self.training_data[self.idx]
            balance_next_state = (self.hold_ratio, self.profitloss, avg_return, self.position)
            return chart_next_state, balance_next_state, reward, done, None
        
        # Agent의 행동에 따라서 다음 환경 정보를 반환.
        else:
            # action에 대한 신뢰도를 계산.
            confidence = policy[action]
            # 행동 수행 및 보상 출력
            reward, trading_unit = self.act(action, confidence)
            # 현재 종가 대비 평균 수익률
            if self.position == Position.LONG:
                # [TODO] : 수정 했음
                avg_return = (self.get_price()/ self.avg_position_price) - 1
            
            elif self.position == Position.NONE:
                avg_return = 0
            
            else:
                # [TODO] : 수정 했음
                avg_return = 1- (self.avg_position_price / self.get_price())
            
            # chart state, balance state 계산.
            chart_next_state = self.training_data[self.idx]
            balance_next_state = (self.hold_ratio, self.profitloss, avg_return, self.position)
            done = False

            # 원금 대비 -80% 손실 나면 epoch 종료
            if self.portfolio_value < self.initial_balance*0.20:
                done = True
            
            return chart_next_state, balance_next_state, reward, done, trading_unit