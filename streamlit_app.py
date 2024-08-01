import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
import ta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize
import requests
import time

@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    attempt = 0
    while attempt < 3:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            info = stock.info
            return hist, info
        except Exception as e:
            attempt += 1
            st.warning(f"Error al obtener datos para {ticker}: {e}. Reintentando {attempt}/3...")
            time.sleep(5)
    st.error(f"No se pudo obtener datos para {ticker} después de varios intentos.")
    return pd.DataFrame(), {}

@st.cache_data
def calculate_technical_indicators(hist):
    data = hist.copy()
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['SMA_100'] = ta.trend.sma_indicator(data['Close'], window=100)
    data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    data['EMA_50'] = ta.trend.ema_indicator(data['Close'], window=50)
    bollinger = ta.volatility.BollingerBands(close=data['Close'])
    data['BB_High'] = bollinger.bollinger_hband()
    data['BB_Low'] = bollinger.bollinger_lband()
    data['BB_Middle'] = bollinger.bollinger_mavg()
    stoch = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['Momentum'] = ta.momentum.roc(data['Close'], window=10)
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Histogram'] = macd.macd_diff()
    adx = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'])
    data['ADX'] = adx.adx()
    data['ADX_Pos'] = adx.adx_pos()
    data['ADX_Neg'] = adx.adx_neg()
    data['CCI'] = ta.trend.CCIIndicator(high=data['High'], low=data['Low'], close=data['Close']).cci()
    obv = ta.volume.OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume'])
    data['OBV'] = obv.on_balance_volume()
    vwap = ta.volume.VolumeWeightedAveragePrice(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'])
    data['VWAP'] = vwap.volume_weighted_average_price()
    return data

def download_data_with_retries(tickers, start_date, end_date, max_retries=3):
    attempt = 0
    while attempt < max_retries:
        try:
            data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
            return data
        except Exception as e:
            attempt += 1
            st.warning(f"Error al descargar datos: {e}. Reintentando {attempt}/{max_retries}...")
            time.sleep(5)
    st.error("No se pudo descargar los datos después de varios intentos.")
    return pd.DataFrame()

def check_internet_connection():
    try:
        requests.get('https://www.google.com', timeout=5)
        return True
    except requests.ConnectionError:
        return False

def filter_valid_tickers(data):
    if data.empty or data.isnull().all().all():
        raise ValueError("No se descargaron datos válidos para los tickers proporcionados.")
    valid_tickers = [ticker for ticker in data.columns if not data[ticker].isnull().all()]
    if '^GSPC' not in valid_tickers:
        raise ValueError("No se encontraron datos para el índice de mercado (^GSPC).")
    data = data[valid_tickers]
    return data, valid_tickers

def calculate_portfolio_metrics(tickers, weights):
    tickers_with_market = tickers + ['^GSPC']
    if not check_internet_connection():
        st.error("No hay conexión a Internet.")
        return None, None, None, None, None, None, None, None

    data = download_data_with_retries(tickers_with_market, '2020-01-01', datetime.today().strftime('%Y-%m-%d'))
    if data.empty:
        return None, None, None, None, None, None, None, None

    data, valid_tickers = filter_valid_tickers(data)
    returns = data.pct_change(fill_method=None).dropna()

    if returns.shape[0] < 2:
        raise ValueError("Los datos descargados no tienen suficientes retornos.")

    market_returns = returns['^GSPC']
    portfolio_returns = returns[tickers].dot(weights)

    if portfolio_returns.empty or len(portfolio_returns) < 2:
        raise ValueError("Los datos de retornos de la cartera no tienen suficientes valores.")

    annualized_return = portfolio_returns.mean() * 252
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    cumulative_return = (1 + portfolio_returns).prod() - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    correlation_matrix = returns.corr()

    return returns, annualized_return, annualized_volatility, cumulative_return, volatility, correlation_matrix, market_returns, portfolio_returns

# Resto del código...

