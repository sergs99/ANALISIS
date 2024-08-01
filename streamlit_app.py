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
st.set_page_config(page_title="Dashboard Financiero", layout="centered")

# Título Principal
st.title('Dashboard Financiero')

# Menú de navegación
menu = st.selectbox('Seleccionar sección', ['Selecciona una opción', 'Análisis Técnico', 'Análisis Fundamental', 'Gestión de Carteras'])

if menu != 'Selecciona una opción':
    # Entradas de usuario
    ticker = st.text_input("Símbolo bursátil:", value='AAPL')
    start_date = st.date_input('Fecha de inicio', (datetime.today() - timedelta(days=252)).date())
    end_date = st.date_input('Fecha de fin', datetime.today().date())

    try:
        hist, info = get_stock_data(ticker, start_date, end_date)
        data = calculate_technical_indicators(hist)

        if menu == 'Análisis Técnico':
            st.subheader(f'Análisis Técnico de {ticker}')
            # (Gráficos de análisis técnico aquí...)

        elif menu == 'Análisis Fundamental':
            st.subheader(f'Análisis Fundamental de {ticker}')
            # (Análisis fundamental aquí...)

        elif menu == 'Gestión de Carteras':
            st.subheader('Gestión de Carteras')
            
            def get_user_input():
                st.sidebar.header('Parámetros de la Cartera')
                tickers = st.sidebar.text_input("Introduce los tickers de las acciones (separados por comas):", "AAPL,MSFT,GOOGL").split(',')
                weights = st.sidebar.text_input("Introduce los pesos de las acciones (separados por comas, deben sumar 1):", "0.4,0.3,0.3").split(',')
                tickers = [ticker.strip().upper() for ticker in tickers]
                weights = np.array([float(weight.strip()) for weight in weights])
                
                if not np.isclose(sum(weights), 1.0, atol=1e-5):
                    st.sidebar.error("La suma de los pesos debe ser aproximadamente igual a 1.0.")
                
                risk_free_rate = st.sidebar.number_input("Introduce la tasa libre de riesgo actual (como fracción, ej. 0.0234 para 2.34%):", value=0.0234)
                return tickers, weights, risk_free_rate

            def download_data(tickers_with_market):
                try:
                    data = yf.download(tickers_with_market, start='2020-01-01', end=datetime.today().strftime('%Y-%m-%d'))['Adj Close']
                except Exception as e:
                    st.error(f"Error al descargar datos: {e}")
                    raise
                return data

            def filter_valid_tickers(data):
                if data.empty:
                    raise ValueError("No se descargaron datos para los tickers proporcionados.")
                
                valid_tickers = [ticker for ticker in data.columns if not data[ticker].isnull().all()]
                if '^GSPC' not in valid_tickers:
                    raise ValueError("No se encontraron datos para el índice de mercado (^GSPC).")
                
                data = data[valid_tickers]
                return data, valid_tickers

            def calculate_portfolio_metrics(tickers, weights):
                tickers_with_market = tickers + ['^GSPC']
                data = download_data(tickers_with_market)
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

            def plot_portfolio_data(portfolio_return, portfolio_volatility, cumulative_return, correlation_matrix):
                st.write(f"Retorno Anualizado de la Cartera: {portfolio_return:.2%}")
                st.write(f"Volatilidad Anualizada de la Cartera: {portfolio_volatility:.2%}")
                st.write(f"Retorno Acumulado de la Cartera: {cumulative_return:.2%}")
                st.write("Matriz de Correlación:")
                st.dataframe(correlation_matrix)

            def plot_distribution(portfolio_returns):
                sns.histplot(portfolio_returns, kde=True, stat="density", linewidth=0)
                mu, std = norm.fit(portfolio_returns)
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                plt.plot(x, p, 'k', linewidth=2)
                plt.title("Distribución de los Retornos de la Cartera")
                plt.show()

            tickers, weights, risk_free_rate = get_user_input()

            try:
                returns, portfolio_return, portfolio_volatility, cumulative_return, volatility, correlation_matrix, market_returns, portfolio_returns = calculate_portfolio_metrics(tickers, weights)

                plot_portfolio_data(portfolio_return, portfolio_volatility, cumulative_return, correlation_matrix)

                sharpe_ratio = calculate_sharpe_ratio(portfolio_return, portfolio_volatility, risk_free_rate)
                sortino_ratio = calculate_sortino_ratio(portfolio_returns, risk_free_rate)
                treynor_ratio = calculate_treynor_ratio(portfolio_returns, market_returns, risk_free_rate)

                st.write(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
                st.write(f"Ratio de Sortino: {sortino_ratio:.2f}")
                st.write(f"Ratio de Treynor: {treynor_ratio:.2f}")

                st.subheader("Optimización de la Cartera")
                optimal_weights = optimize_portfolio(returns, risk_free_rate)
                optimal_weights = np.round(optimal_weights, 4)
                optimal_weights = dict(zip(returns.columns, optimal_weights))

                st.write("Pesos Óptimos de la Cartera para Maximizar el Ratio de Sharpe:")
                for ticker, weight in optimal_weights.items():
                    st.write(f"{ticker}: {weight:.2%}")

                st.subheader("Distribución de los Retornos de la Cartera")
                fig, ax = plt.subplots()
                sns.histplot(portfolio_returns, kde=True, stat="density", linewidth=0, ax=ax)
                mu, std = norm.fit(portfolio_returns)
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                ax.plot(x, p, 'k', linewidth=2)
                ax.set_title("Distribución de los Retornos de la Cartera")
                st.pyplot(fig)

            except ValueError as e:
                st.error(f"Error en la gestión de carteras: {e}")

    except Exception as e:
        st.error(f"Error al obtener datos para el ticker {ticker}: {e}")

# Mostrar mensaje de bienvenida si no se ha seleccionado ninguna sección
else:
    st.write("Bienvenido al Dashboard Financiero. Por favor, selecciona una sección en el menú desplegable para empezar.")
