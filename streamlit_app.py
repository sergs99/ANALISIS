import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize
from datetime import datetime, timedelta
import ta
import plotly.graph_objs as go
import plotly.graph_objects as go

def get_user_input():
    tickers = st.text_input("Introduce los tickers de las acciones (separados por comas):", value='AAPL, MSFT').split(',')
    weights = st.text_input("Introduce los pesos de las acciones (separados por comas, deben sumar 1):", value='0.5, 0.5').split(',')
    tickers = [ticker.strip().upper() for ticker in tickers]
    weights = np.array([float(weight.strip()) for weight in weights])
    if not np.isclose(sum(weights), 1.0, atol=1e-5):
        st.error("La suma de los pesos debe ser aproximadamente igual a 1.0.")
        return None, None, None
    risk_free_rate = st.number_input("Introduce la tasa libre de riesgo actual (como fracción, ej. 0.0234 para 2.34%):", value=0.0234)
    return tickers, weights, risk_free_rate

def calculate_portfolio_metrics(tickers, weights):
    end_date = datetime.today().strftime('%Y-%m-%d')
    tickers_with_market = tickers + ['^GSPC']
    try:
        data = yf.download(tickers_with_market, start='2020-01-01', end=end_date)['Adj Close']
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        return None, None, None, None, None, None
    if data.empty:
        st.error("No se descargaron datos para los tickers proporcionados.")
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
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.tight_layout()
    st.pyplot(plt)

def plot_cml_sml(portfolio_return, portfolio_volatility, market_returns, risk_free_rate):
    market_return = market_returns.mean() * 252
    market_volatility = market_returns.std() * np.sqrt(252)
    volatilities = np.linspace(0, market_volatility * 2, 100)
    returns_cml = risk_free_rate + (portfolio_return - risk_free_rate) / portfolio_volatility * volatilities
    returns_sml = np.linspace(risk_free_rate, market_return * 1.5, 100)
    volatilities_sml = (returns_sml - risk_free_rate) / market_return * market_volatility
    plt.figure(figsize=(12, 8))
    plt.title("Capital Market Line (CML) y Security Market Line (SML)")
    plt.plot(volatilities, returns_cml, label='Capital Market Line (CML)', color='blue')
    plt.plot(volatilities_sml, returns_sml, label='Security Market Line (SML)', color='red')
    plt.scatter(portfolio_volatility, portfolio_return, color='green', marker='o', s=100, label='Cartera')
    plt.scatter(market_volatility, market_return, color='orange', marker='x', s=100, label='Mercado')
    plt.xlabel('Volatilidad')
    plt.ylabel('Retorno')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def check_normality(returns):
    plt.figure(figsize=(12, 6))
    sns.histplot(returns.mean(axis=1) * 252, kde=True, stat='density', linewidth=0, bins=50)
    mu, std = norm.fit(returns.mean(axis=1) * 252)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title("Distribución de los Retornos Anualizados")
    plt.xlabel('Retorno Anualizado')
    plt.ylabel('Densidad')
    plt.grid(True)
    st.pyplot(plt)

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
    return data

# Configuración de la aplicación Streamlit
st.set_page_config(layout='wide', page_title='Análisis de Portafolios e Indicadores Técnicos')

# Título de la aplicación
st.title('Análisis de Portafolios e Indicadores Técnicos')

# Obtener entradas del usuario para análisis de portafolio
st.header("Análisis de Portafolios")
tickers, weights, risk_free_rate = get_user_input()

if tickers and weights is not None and risk_free_rate is not None:
    returns, portfolio_return, portfolio_volatility, correlation_matrix, market_returns, portfolio_returns = calculate_portfolio_metrics(tickers, weights)
    if returns is not None:
        st.subheader("Resultados del Portafolio Inicial")
        plot_portfolio_data(portfolio_return, portfolio_volatility, correlation_matrix)
        sharpe_ratio = calculate_sharpe_ratio(portfolio_return, portfolio_volatility, risk_free_rate)
        st.write(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
        sortino_ratio = calculate_sortino_ratio(portfolio_returns, risk_free_rate)
        st.write(f"Ratio de Sortino: {sortino_ratio:.2f}")
        treynor_ratio = calculate_treynor_ratio(portfolio_returns, market_returns, risk_free_rate)
        st.write(f"Ratio de Treynor: {treynor_ratio:.2f}")

        st.subheader("Optimización del Portafolio")
        optimal_weights = optimize_portfolio(returns[tickers], risk_free_rate)
        optimal_portfolio_returns = returns[tickers].dot(optimal_weights)
        optimal_return = optimal_portfolio_returns.mean() * 252
        optimal_volatility = optimal_portfolio_returns.std() * np.sqrt(252)
        optimal_sharpe_ratio = calculate_sharpe_ratio(optimal_return, optimal_volatility, risk_free_rate)
        optimal_sortino_ratio = calculate_sortino_ratio(optimal_portfolio_returns, risk_free_rate)
        optimal_treynor_ratio = calculate_treynor_ratio(optimal_portfolio_returns, market_returns, risk_free_rate)
        st.write("\nComposición óptima de la cartera:")
        for ticker, weight in zip(tickers, optimal_weights):
            st.write(f"{ticker}: {weight:.2%}")
        st.write(f"\nRentabilidad media anualizada de la cartera óptima: {optimal_return * 100:.2f}%")
        st.write(f"Volatilidad anualizada de la cartera óptima: {optimal_volatility * 100:.2f}%")
        st.write(f"Ratio de Sharpe de la cartera óptima: {optimal_sharpe_ratio:.2f}")
        st.write(f"Ratio de Sortino de la cartera óptima: {optimal_sortino_ratio:.2f}")
        st.write(f"Ratio de Treynor de la cartera óptima: {optimal_treynor_ratio:.2f}")

        st.subheader("Capital Market Line (CML) y Security Market Line (SML)")
        plot_cml_sml(portfolio_return, portfolio_volatility, market_returns, risk_free_rate)

        st.subheader("Verificación de Normalidad de los Retornos")
        check_normality(returns[tickers])

# Sección de indicadores técnicos
st.header("Análisis de Indicadores Técnicos")
stock_ticker = st.text_input("Introduce el ticker de la acción para análisis técnico:", value='AAPL').upper()
start_date = st.date_input("Fecha de inicio:", value=datetime.today() - timedelta(days=365))
end_date = st.date_input("Fecha de fin:", value=datetime.today())

# Descargar datos y calcular indicadores técnicos
try:
    hist, info = get_stock_data(stock_ticker, start_date, end_date)
    technical_data = calculate_technical_indicators(hist)

    st.subheader(f"Datos Históricos de {stock_ticker}")
    st.write(hist.tail())

    st.subheader(f"Indicadores Técnicos de {stock_ticker}")
    st.write(technical_data.tail())

    st.subheader("Gráficos de Indicadores Técnicos")

    # Gráfico de SMA y EMA
    st.write("SMA y EMA")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['SMA_50'], mode='lines', name='SMA 50'))
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['SMA_100'], mode='lines', name='SMA 100'))
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['SMA_200'], mode='lines', name='SMA 200'))
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['EMA_20'], mode='lines', name='EMA 20'))
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['EMA_50'], mode='lines', name='EMA 50'))
    fig.update_layout(title=f"SMA y EMA de {stock_ticker}", xaxis_title="Fecha", yaxis_title="Precio", template="plotly_dark")
    st.plotly_chart(fig)

    # Gráfico de Bollinger Bands
    st.write("Bollinger Bands")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['BB_High'], mode='lines', name='BB High'))
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['BB_Middle'], mode='lines', name='BB Middle'))
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['BB_Low'], mode='lines', name='BB Low'))
    fig.update_layout(title=f"Bollinger Bands de {stock_ticker}", xaxis_title="Fecha", yaxis_title="Precio", template="plotly_dark")
    st.plotly_chart(fig)

    # Gráfico de MACD
    st.write("MACD")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['MACD_Signal'], mode='lines', name='MACD Signal'))
    fig.add_trace(go.Bar(x=technical_data.index, y=technical_data['MACD_Histogram'], name='MACD Histogram'))
    fig.update_layout(title=f"MACD de {stock_ticker}", xaxis_title="Fecha", yaxis_title="Valor", template="plotly_dark")
    st.plotly_chart(fig)

    # Gráfico de ADX
    st.write("ADX")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['ADX'], mode='lines', name='ADX'))
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['ADX_Pos'], mode='lines', name='ADX Pos'))
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['ADX_Neg'], mode='lines', name='ADX Neg'))
    fig.update_layout(title=f"ADX de {stock_ticker}", xaxis_title="Fecha", yaxis_title="Valor", template="plotly_dark")
    st.plotly_chart(fig)

    # Gráfico de CCI
    st.write("CCI")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['CCI'], mode='lines', name='CCI'))
    fig.update_layout(title=f"CCI de {stock_ticker}", xaxis_title="Fecha", yaxis_title="Valor", template="plotly_dark")
    st.plotly_chart(fig)

    # Gráfico de RSI
    st.write("RSI")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['RSI'], mode='lines', name='RSI'))
    fig.update_layout(title=f"RSI de {stock_ticker}", xaxis_title="Fecha", yaxis_title="RSI", template="plotly_dark")
    st.plotly_chart(fig)

except Exception as e:
    st.error(f"Error: {e}")
