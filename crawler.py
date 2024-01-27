import time
import itertools
import asyncio
import aiohttp
import orjson
from loguru import logger

import datetime
import numpy as np
import pandas as pd


from typing import List, Union
from fake_useragent import UserAgent
from enum import Enum


class Endpoints(Enum):
    BINANCE_FUTURES_CANDLESTICK_API = 'https://fapi.binance.com/fapi/v1/klines'



class RestClient:
    def __init__(self, loop) -> None:
        self.ip_address : List[str] = ['0.0.0.0']
        user_agent = UserAgent()

        self.sessions = itertools.cycle([aiohttp.ClientSession(loop=loop,
                                                               headers={'User-Agent':user_agent.random},
                                                               json_serialize=orjson.dumps,
                                                               connector=aiohttp.TCPConnector(local_addr=(ip, 0))) for ip in self.ip_address]
                                        )
        
    async def get(self,
                  url,
                  params=None,
                  timeout=1,
                  headers=None) -> dict:
        async with next(self.sessions).get(url=url, params=params, timeout=timeout, headers=headers) as response:
            return await response.json()



class Crawler:
    
    def __init__(self):
        loop = asyncio.get_event_loop()
        self.client = RestClient(loop)

    async def get_coin_candle_data(self, url, symbol:str, interval:str, startTime:int, limit:int=1500)->List:
        """
        [
          [
            1499040000000,      // Open time
            "0.01634790",       // Open
            "0.80000000",       // High
            "0.01575800",       // Low
            "0.01577100",       // Close
            "148976.11427815",  // Volume
            1499644799999,      // Close time
            "2434.19055334",    // Quote asset volume
            308,                // Number of trades
            "1756.87402397",    // Taker buy base asset volume
            "28.46694368",      // Taker buy quote asset volume
            "17928899.62484339" // Ignore.
          ]
        ]

        """
        response = []
        try:
            params = {'symbol' : symbol,
                    'interval' : interval,
                    'startTime' : startTime,
                    'limit' : limit}
            
            response = await self.client.get(url = url, params = params)
            
            if response:
                startTime, endTime = response[0][0], response[-1][0]
                logger.info(f"SYMBOL :{symbol} | startTime : {datetime.datetime.utcfromtimestamp(startTime/1000).strftime('%Y-%m-%d %H:%M:%S')}, endTime : {datetime.datetime.utcfromtimestamp(endTime/1000).strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            logger.error(e)
        return response



    async def get_coin_candle_all(self, url, symbol:str, interval:str, startTime:int, limit:int=1500, save:bool=False, save_path:str='./') -> pd.DataFrame:
        logger.info(f"CRAWLING START | SYMBOL :{symbol}, INTERVAL : {interval}")
        startTime = int(datetime.datetime.strptime(startTime, "%Y-%m-%d").timestamp() * 1000)
        
        candle_list, iter = [], 0
        
        while True:
            response = await self.get_coin_candle_data(url, symbol, interval, startTime, limit)
            
            # 반환 데이터 없으면 종료
            if len(response) == 0:
                break

            startTime = response[-1][6] + 1
            candle_list.extend(response)
            iter += 1
            await asyncio.sleep(1)


        logger.info(f"CRAWLING END |SYMBOL :{symbol}, ITER : {iter}, DATA LENGTH : {len(candle_list)}")
        # dataframe 정리
        df_response = pd.DataFrame(candle_list, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        df_response.drop(columns=['Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'], axis=1, inplace=True)
        
        if save:
            df_response.to_csv(path_or_buf=save_path, index=False)

        return df_response


async def main():
    cralwer_btc = Crawler()
    cralwer_eth = Crawler()

    await asyncio.gather(
        cralwer_btc.get_coin_candle_all(url=Endpoints.BINANCE_FUTURES_CANDLESTICK_API.value, 
                                                  symbol='BTCUSDT', 
                                                  interval='1h',
                                                  startTime='2018-01-01', 
                                                  limit=1500,
                                                  save=True,
                                                  save_path='./data/btc_1h.csv'),

        cralwer_eth.get_coin_candle_all(url=Endpoints.BINANCE_FUTURES_CANDLESTICK_API.value, 
                                                  symbol='ETHUSDT', 
                                                  interval='1h',
                                                  startTime='2018-01-01', 
                                                  limit=1500,
                                                  save=True,
                                                  save_path='./data/eth_1h.csv')                          
    )


if __name__ == '__main__':
    asyncio.run(main())

    
