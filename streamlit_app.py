import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Función para obtener datos históricos y la información del stock
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    info = stock.info
    return hist, info

# Función para calcular indicadores técnicos
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

# Configuración de la página
st.set_page_config(page_title="Dashboard Financiero", layout="wide")

def update_layout(fig, title, yaxis_title):
    fig.update_layout(
        title=title,
        title_font=dict(size=18, color='white'),
        xaxis_title='Fecha',
        xaxis_title_font=dict(size=14, color='white'),
        yaxis_title=yaxis_title,
        yaxis_title_font=dict(size=14, color='white'),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(gridcolor='grey', zerolinecolor='grey'),
        yaxis=dict(gridcolor='grey', zerolinecolor='grey')
    )
    return fig

def get_user_input():
    # Identificadores únicos para widgets
    tickers_input = st.text_input("Introduce los tickers de las acciones (separados por comas):", key="tickers")
    weights_input = st.text_input("Introduce los pesos de las acciones (separados por comas, deben sumar 1):", key="weights")
    risk_free_rate_input = st.text_input("Introduce la tasa libre de riesgo actual (como fracción, ej. 0.0234 para 2.34%):", key="risk_free_rate")
    
    try:
        if tickers_input and weights_input and risk_free_rate_input:
            tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
            weights = np.array([float(weight.strip()) for weight in weights_input.split(',')])
            risk_free_rate = float(risk_free_rate_input.strip())

            if not np.isclose(sum(weights), 1.0, atol=1e-5):
                st.error("La suma de los pesos debe ser aproximadamente igual a 1.0.")
                return None, None, None

            return tickers, weights, risk_free_rate
        else:
            st.warning("Por favor, llena todos los campos.")
            return None, None, None
            
    except ValueError as e:
        st.error(f"Error: {e}")
        st.warning("Por favor, ingresa los datos nuevamente.")
        return None, None, None

def download_data(tickers, start_date, end_date):
    retries = 3
    for attempt in range(retries):
        try:
            data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
            if not data.empty:
                return data
        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"Intento {attempt + 1} fallido. Reintentando... ({e})")
            else:
                st.error(f"Error al descargar datos: {e}")
                return None
    return None

def calculate_portfolio_metrics(tickers, weights):
    end_date = datetime.today().strftime('%Y-%m-%d')
    tickers_with_market = tickers + ['^GSPC']
    
    # Intentar descargar los datos con reintentos
    data = download_data(tickers_with_market, start='2020-01-01', end=end_date)
    
    if data is None:
        st.error("No se pudieron descargar los datos. Verifica tus tickers y conexión a Internet.")
        return None, None, None, None, None, None

    returns = data.pct_change().dropna()
    if returns.empty:
        st.error("Los datos descargados no tienen suficientes retornos.")
        return None, None, None, None, None, None

    market_returns = returns['^GSPC']
    portfolio_returns = returns[tickers].dot(weights)
    
    annualized_return = portfolio_returns.mean() * 252
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    
    correlation_matrix = returns.corr()

    return returns, annualized_return, annualized_volatility, correlation_matrix, market_returns, portfolio_returns

def calculate_sharpe_ratio(portfolio_return, portfolio_volatility, risk_free_rate):
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return sharpe_ratio

def calculate_sortino_ratio(portfolio_returns, risk_free_rate):
    downside_risk = np.sqrt(np.mean(np.minimum(0, portfolio_returns - risk_free_rate / 252) ** 2) * 252)
    portfolio_return = portfolio_returns.mean() * 252
    sortino_ratio = (portfolio_return - risk_free_rate) / downside_risk
    return sortino_ratio

def calculate_treynor_ratio(portfolio_returns, market_returns, risk_free_rate):
    portfolio_return = portfolio_returns.mean() * 252
    market_return = market_returns.mean() * 252
    beta = np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)
    treynor_ratio = (portfolio_return - risk_free_rate) / beta
    return treynor_ratio

def optimize_portfolio(returns, risk_free_rate):
    def objective(weights):
        portfolio_returns = returns.dot(weights)
        portfolio_return = portfolio_returns.mean() * 252
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

    num_assets = returns.shape[1]
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, 1) for _ in range(num_assets)]
    initial_weights = np.array(num_assets * [1. / num_assets])

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

def plot_portfolio_data(portfolio_return, portfolio_volatility, correlation_matrix):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Rentabilidad y Volatilidad")
    plt.bar(["Rentabilidad", "Volatilidad"], [portfolio_return * 100, portfolio_volatility * 100], color=['blue', 'orange'])
    plt.ylabel('Porcentaje')

    plt.subplot(1, 2, 2)
    plt.title("Matriz de Correlación")
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    
    plt.tight_layout()
    st.pyplot(plt.gcf())

def plot_technical_indicators(hist):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], mode='lines', name='SMA 50'))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_100'], mode='lines', name='SMA 100'))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], mode='lines', name='SMA 200'))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA_20'], mode='lines', name='EMA 20'))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA_50'], mode='lines', name='EMA 50'))

    fig.update_layout(
        title='Indicadores Técnicos',
        title_font=dict(size=18, color='white'),
        xaxis_title='Fecha',
        xaxis_title_font=dict(size=14, color='white'),
        yaxis_title='Precio',
        yaxis_title_font=dict(size=14, color='white'),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(gridcolor='grey', zerolinecolor='grey'),
        yaxis=dict(gridcolor='grey', zerolinecolor='grey')
    )

    st.plotly_chart(fig, use_container_width=True)

def display_stock_analysis(hist, info):
    st.write(f"### Información de la acción")
    st.write(f"- **Nombre**: {info.get('longName', 'N/A')}")
    st.write(f"- **Sector**: {info.get('sector', 'N/A')}")
    st.write(f"- **Industria**: {info.get('industry', 'N/A')}")
    st.write(f"- **Descripción**: {info.get('longBusinessSummary', 'N/A')}")
    
    plot_technical_indicators(hist)

def main():
    st.title("Dashboard Financiero")
    
    menu = st.sidebar.selectbox("Selecciona una opción", ["Análisis Técnico", "Análisis Fundamental", "Optimización de Carteras"])
    
    if menu == "Análisis Técnico":
        st.subheader("Análisis Técnico")
        ticker = st.text_input("Introduce el ticker de la acción:", key="ticker")
        
        if ticker:
            start_date = st.date_input("Fecha de Inicio", datetime(2020, 1, 1))
            end_date = st.date_input("Fecha de Fin", datetime.today())
            
            if start_date < end_date:
                hist, info = get_stock_data(ticker, start_date, end_date)
                if hist is not None:
                    data_with_indicators = calculate_technical_indicators(hist)
                    display_stock_analysis(data_with_indicators, info)
                else:
                    st.error("No se pudieron obtener datos para el ticker proporcionado.")
            else:
                st.error("La fecha de fin debe ser posterior a la fecha de inicio.")

    elif menu == "Análisis Fundamental":
        st.subheader("Análisis Fundamental")
        ticker = st.text_input("Introduce el ticker de la acción:", key="ticker_fundamental")

        if ticker:
            start_date = st.date_input("Fecha de Inicio", datetime(2020, 1, 1), key="start_date_fundamental")
            end_date = st.date_input("Fecha de Fin", datetime.today(), key="end_date_fundamental")
            
            if start_date < end_date:
                hist, info = get_stock_data(ticker, start_date, end_date)
                if hist is not None:
                    st.write(f"### Información de la acción")
                    st.write(f"- **Nombre**: {info.get('longName', 'N/A')}")
                    st.write(f"- **Sector**: {info.get('sector', 'N/A')}")
                    st.write(f"- **Industria**: {info.get('industry', 'N/A')}")
                    st.write(f"- **Descripción**: {info.get('longBusinessSummary', 'N/A')}")
                    st.write(f"- **Precio Actual**: {hist['Close'].iloc[-1]:.2f}")
                    st.write(f"- **Capitalización de Mercado**: {info.get('marketCap', 'N/A')}")
                    st.write(f"- **PER**: {info.get('trailingPE', 'N/A')}")
                    st.write(f"- **PEG Ratio**: {info.get('pegRatio', 'N/A')}")
                    st.write(f"- **ROE**: {info.get('returnOnEquity', 'N/A')}")
                    st.write(f"- **Dividend Yield**: {info.get('dividendYield', 'N/A') * 100:.2f}%")
                else:
                    st.error("No se pudieron obtener datos para el ticker proporcionado.")
            else:
                st.error("La fecha de fin debe ser posterior a la fecha de inicio.")

    elif menu == "Optimización de Carteras":
        st.subheader("Optimización de Carteras")
        tickers, weights, risk_free_rate = get_user_input()
        
        if tickers and weights and risk_free_rate:
            returns, annualized_return, annualized_volatility, correlation_matrix, market_returns, portfolio_returns = calculate_portfolio_metrics(tickers, weights)
            if returns is not None:
                st.write(f"### Métricas del Portafolio")
                st.write(f"- **Rentabilidad Anualizada**: {annualized_return * 100:.2f}%")
                st.write(f"- **Volatilidad Anualizada**: {annualized_volatility * 100:.2f}%")
                
                sharpe_ratio = calculate_sharpe_ratio(annualized_return, annualized_volatility, risk_free_rate)
                sortino_ratio = calculate_sortino_ratio(portfolio_returns, risk_free_rate)
                treynor_ratio = calculate_treynor_ratio(portfolio_returns, market_returns, risk_free_rate)
                
                st.write(f"- **Sharpe Ratio**: {sharpe_ratio:.2f}")
                st.write(f"- **Sortino Ratio**: {sortino_ratio:.2f}")
                st.write(f"- **Treynor Ratio**: {treynor_ratio:.2f}")
                
                plot_portfolio_data(annualized_return, annualized_volatility, correlation_matrix)
                
                optimized_weights = optimize_portfolio(returns, risk_free_rate)
                st.write(f"### Pesos Óptimos de la Cartera")
                st.write(f"Pesos óptimos: {', '.join([f'{ticker}: {weight:.2f}' for ticker, weight in zip(tickers, optimized_weights)])}")
            else:
                st.error("No se pudieron obtener datos para calcular las métricas del portafolio.")
                
if __name__ == "__main__":
    main()

