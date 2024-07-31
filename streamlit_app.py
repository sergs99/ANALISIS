import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize
from datetime import datetime, timedelta
import ta

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

st.title('Dashboard Financiero')

# Entradas de usuario
ticker = st.text_input("Símbolo bursátil:", value='AAPL')
start_date = st.date_input('Fecha de inicio', (datetime.today() - timedelta(days=252)).date())
end_date = st.date_input('Fecha de fin', datetime.today().date())

# Mostrar datos y gráficos
try:
    hist, info = get_stock_data(ticker, start_date, end_date)
    data = calculate_technical_indicators(hist)

    # Tabs
    selected_tab = st.selectbox('Seleccionar pestaña', ['Análisis Técnico', 'Fundamental'])

    if selected_tab == 'Análisis Técnico':
        # Gráfico de Velas
        price_fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            name='Candlestick'
        )])
        price_fig.update_layout(
            title=f'Gráfico de Velas de {ticker}',
            xaxis_title='Fecha',
            yaxis_title='Precio',
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )
        st.plotly_chart(price_fig)

        # Gráfico de Volumen
        volume_fig = go.Figure()
        volume_fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volumen de Negociación', marker_color='rgba(255, 87, 34, 0.8)'))
        volume_fig.update_layout(
            title=f'Volumen de Negociación de {ticker}',
            xaxis_title='Fecha',
            yaxis_title='Volumen',
            template='plotly_dark'
        )
        st.plotly_chart(volume_fig)

        # Bandas de Bollinger
        bollinger_fig = go.Figure()
        bollinger_fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            name='Candlestick'
        ))
        bollinger_fig.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'], mode='lines', name='Media Móvil', line=dict(color='cyan')))
        bollinger_fig.add_trace(go.Scatter(x=data.index, y=data['BB_High'], mode='lines', name='Banda Superior', line=dict(color='red')))
        bollinger_fig.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], mode='lines', name='Banda Inferior', line=dict(color='green')))
        bollinger_fig.update_layout(
            title=f'Bandas de Bollinger de {ticker}',
            xaxis_title='Fecha',
            yaxis_title='Precio',
            template='plotly_dark'
        )
        st.plotly_chart(bollinger_fig)

        # RSI
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='magenta')))
        rsi_fig.add_hline(y=70, line_dash='dash', line_color='red')
        rsi_fig.add_hline(y=30, line_dash='dash', line_color='green')
        rsi_fig.update_layout(
            title=f'Índice de Fuerza Relativa (RSI) de {ticker}',
            xaxis_title='Fecha',
            yaxis_title='RSI',
            template='plotly_dark'
        )
        st.plotly_chart(rsi_fig)

        # Oscilador Estocástico
        stochastic_fig = go.Figure()
        stochastic_fig.add_trace(go.Scatter(x=data.index, y=data['Stoch_K'], mode='lines', name='%K', line=dict(color='yellow')))
        stochastic_fig.add_trace(go.Scatter(x=data.index, y=data['Stoch_D'], mode='lines', name='%D', line=dict(color='lightcoral')))
        stochastic_fig.update_layout(
            title=f'Oscilador Estocástico de {ticker}',
            xaxis_title='Fecha',
            yaxis_title='Valor',
            template='plotly_dark'
        )
        st.plotly_chart(stochastic_fig)

        # MACD
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
        macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='MACD Signal', line=dict(color='orange')))
        macd_fig.add_trace(go.Bar(x=data.index, y=data['MACD_Histogram'], name='MACD Histogram', marker_color='rgba(255, 87, 34, 0.8)'))
        macd_fig.update_layout(
            title=f'MACD de {ticker}',
            xaxis_title='Fecha',
            yaxis_title='Valor',
            template='plotly_dark'
        )
        st.plotly_chart(macd_fig)

        # Momentum
        momentum_fig = go.Figure()
        momentum_fig.add_trace(go.Scatter(x=data.index, y=data['Momentum'], mode='lines', name='Momentum', line=dict(color='purple')))
        momentum_fig.update_layout(
            title=f'Momentum de {ticker}',
            xaxis_title='Fecha',
            yaxis_title='Valor',
            template='plotly_dark'
        )
        st.plotly_chart(momentum_fig)

        # ADX
        adx_fig = go.Figure()
        adx_fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], mode='lines', name='ADX', line=dict(color='blue')))
        adx_fig.add_trace(go.Scatter(x=data.index, y=data['ADX_Pos'], mode='lines', name='ADX+', line=dict(color='green')))
        adx_fig.add_trace(go.Scatter(x=data.index, y=data['ADX_Neg'], mode='lines', name='ADX-', line=dict(color='red')))
        adx_fig.update_layout(
            title=f'ADX de {ticker}',
            xaxis_title='Fecha',
            yaxis_title='ADX',
            template='plotly_dark'
        )
        st.plotly_chart(adx_fig)

        # CCI
        cci_fig = go.Figure()
        cci_fig.add_trace(go.Scatter(x=data.index, y=data['CCI'], mode='lines', name='CCI', line=dict(color='orange')))
        cci_fig.add_hline(y=100, line_dash='dash', line_color='red')
        cci_fig.add_hline(y=-100, line_dash='dash', line_color='green')
        cci_fig.update_layout(
            title=f'Índice de Canal de Commodities (CCI) de {ticker}',
            xaxis_title='Fecha',
            yaxis_title='CCI',
            template='plotly_dark'
        )
        st.plotly_chart(cci_fig)

        # OBV
        obv_fig = go.Figure()
        obv_fig.add_trace(go.Scatter(x=data.index, y=data['OBV'], mode='lines', name='OBV', line=dict(color='blue')))
        obv_fig.update_layout(
            title=f'On-Balance Volume (OBV) de {ticker}',
            xaxis_title='Fecha',
            yaxis_title='OBV',
            template='plotly_dark'
        )
        st.plotly_chart(obv_fig)

        # VWAP
        vwap_fig = go.Figure()
        vwap_fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP', line=dict(color='cyan')))
        vwap_fig.update_layout(
            title=f'Precio Promedio Ponderado por Volumen (VWAP) de {ticker}',
            xaxis_title='Fecha',
            yaxis_title='VWAP',
            template='plotly_dark'
        )
        st.plotly_chart(vwap_fig)


# Función para calcular métricas de cartera
def calculate_portfolio_metrics(tickers, weights):
    end_date = datetime.today().strftime('%Y-%m-%d')
    tickers_with_market = tickers + ['^GSPC']
    try:
        data = yf.download(tickers_with_market, start='2020-01-01', end=end_date)['Adj Close']
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        raise

    if data.empty:
        raise ValueError("No se descargaron datos para los tickers proporcionados.")

    returns = data.pct_change().dropna()
    if returns.empty:
        raise ValueError("Los datos descargados no tienen suficientes retornos.")

    market_returns = returns['^GSPC']
    portfolio_returns = returns[tickers].dot(weights)
    
    annualized_return = portfolio_returns.mean() * 252
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    
    correlation_matrix = returns.corr()

    return returns, annualized_return, annualized_volatility, correlation_matrix, market_returns, portfolio_returns

# Función para calcular ratios
def calculate_sharpe_ratio(portfolio_return, portfolio_volatility, risk_free_rate):
    return (portfolio_return - risk_free_rate) / portfolio_volatility

def calculate_sortino_ratio(portfolio_returns, risk_free_rate):
    downside_risk = np.sqrt(np.mean(np.minimum(0, portfolio_returns - risk_free_rate / 252) ** 2) * 252)
    portfolio_return = portfolio_returns.mean() * 252
    return (portfolio_return - risk_free_rate) / downside_risk

def calculate_treynor_ratio(portfolio_returns, market_returns, risk_free_rate):
    portfolio_return = portfolio_returns.mean() * 252
    market_return = market_returns.mean() * 252
    beta = np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)
    return (portfolio_return - risk_free_rate) / beta

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

# Función para graficar datos de la cartera
def plot_portfolio_data(portfolio_return, portfolio_volatility, correlation_matrix):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].bar(["Rentabilidad", "Volatilidad"], [portfolio_return * 100, portfolio_volatility * 100], color=['blue', 'orange'])
    ax[0].set_title("Rentabilidad y Volatilidad")
    ax[0].set_ylabel('Porcentaje')

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', ax=ax[1])
    ax[1].set_title("Matriz de Correlación")
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_cml_sml(portfolio_return, portfolio_volatility, market_returns, risk_free_rate):
    market_return = market_returns.mean() * 252
    market_volatility = market_returns.std() * np.sqrt(252)

    volatilities = np.linspace(0, market_volatility * 2, 100)
    returns_cml = risk_free_rate + (portfolio_return - risk_free_rate) / portfolio_volatility * volatilities

    returns_sml = np.linspace(risk_free_rate, market_return * 1.5, 100)
    volatilities_sml = (returns_sml - risk_free_rate) / market_return * market_volatility

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(volatilities, returns_cml, label='Capital Market Line (CML)', color='blue')
    ax.plot(volatilities_sml, returns_sml, label='Security Market Line (SML)', color='red')
    ax.scatter(portfolio_volatility, portfolio_return, color='green', marker='o', s=100, label='Cartera')
    ax.scatter(market_volatility, market_return, color='orange', marker='x', s=100, label='Mercado')

    ax.set_xlabel('Volatilidad')
    ax.set_ylabel('Retorno')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def check_normality(returns):
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.histplot(returns.mean(axis=1) * 252, kde=True, stat='density', linewidth=0, bins=50, ax=ax)

    mu, std = norm.fit(returns.mean(axis=1) * 252)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)

    ax.set_title("Distribución de los Retornos Anualizados")
    ax.set_xlabel('Retorno Anualizado')
    ax.set_ylabel('Densidad')
    ax.grid(True)
    st.pyplot(fig)

# Función para obtener y mostrar análisis fundamental
@st.cache_data
def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info

# Configuración de la página
st.set_page_config(page_title="Dashboard Financiero", layout="wide")

# Menú de navegación
st.sidebar.title("Menú de Navegación")
option = st.sidebar.radio("Selecciona una sección", ["Análisis Técnico", "Gestión de Carteras", "Análisis Fundamental"])

if option == "Análisis Técnico":
    st.title("Análisis Técnico de Acciones")

    # Selección de ticker y fechas
    ticker = st.text_input("Introduce el ticker de la acción (ejemplo: AAPL)", "AAPL")
    start_date = st.date_input("Fecha de inicio", datetime(2022, 1, 1))
    end_date = st.date_input("Fecha de fin", datetime.today())

    if st.button("Obtener Datos"):
        if ticker:
            try:
                hist, info = get_stock_data(ticker, start_date, end_date)
                st.write(f"**Información del Ticker {ticker}:**")
                st.write(info)
                
                st.write("**Datos Históricos:**")
                st.dataframe(hist)

                st.write("**Datos de Indicadores Técnicos:**")
                data = calculate_technical_indicators(hist)
                st.dataframe(data)

                # Graficar los datos
                st.write("**Gráficas de los Indicadores Técnicos:**")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50'))
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_100'], mode='lines', name='SMA 100'))
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], mode='lines', name='SMA 200'))
                fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], mode='lines', name='EMA 20'))
                fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], mode='lines', name='EMA 50'))
                st.plotly_chart(fig)

                st.write("**Gráfica de Bandas de Bollinger:**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_High'], mode='lines', name='Banda Alta'))
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], mode='lines', name='Banda Baja'))
                st.plotly_chart(fig)
                
                st.write("**Histograma de RSI:**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'))
                st.plotly_chart(fig)
                
                st.write("**Gráfica de MACD:**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'))
                fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='MACD Signal'))
                fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Histogram'], mode='lines', name='MACD Histogram'))
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error al obtener datos: {e}")

elif option == "Gestión de Carteras":
    st.title("Gestión de Carteras")

    tickers_input = st.text_area("Introduce los tickers separados por comas (ejemplo: AAPL, MSFT, GOOG)", "AAPL, MSFT, GOOG")
    tickers = [ticker.strip() for ticker in tickers_input.split(',')]
    weights_input = st.text_area("Introduce los pesos separados por comas (ejemplo: 0.4, 0.3, 0.3)", "0.4, 0.3, 0.3")
    weights = list(map(float, weights_input.split(',')))

    if len(tickers) != len(weights):
        st.error("El número de tickers y pesos debe ser igual.")
    elif sum(weights) != 1:
        st.error("La suma de los pesos debe ser igual a 1.")
    else:
        if st.button("Calcular Métricas"):
            try:
                returns, annualized_return, annualized_volatility, correlation_matrix, market_returns, portfolio_returns = calculate_portfolio_metrics(tickers, weights)
                st.write(f"**Rentabilidad Anualizada de la Cartera:** {annualized_return:.2%}")
                st.write(f"**Volatilidad Anualizada de la Cartera:** {annualized_volatility:.2%}")

                # Calcular y mostrar ratios
                risk_free_rate = 0.04 / 252  # Supongamos una tasa libre de riesgo anual del 4%
                sharpe_ratio = calculate_sharpe_ratio(annualized_return, annualized_volatility, risk_free_rate)
                sortino_ratio = calculate_sortino_ratio(portfolio_returns, risk_free_rate)
                treynor_ratio = calculate_treynor_ratio(portfolio_returns, market_returns, risk_free_rate)
                
                st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
                st.write(f"**Sortino Ratio:** {sortino_ratio:.2f}")
                st.write(f"**Treynor Ratio:** {treynor_ratio:.2f}")

                # Graficar datos de la cartera
                plot_portfolio_data(annualized_return, annualized_volatility, correlation_matrix)
                
                # Graficar CML y SML
                plot_cml_sml(annualized_return, annualized_volatility, market_returns, risk_free_rate)

                # Verificar normalidad
                check_normality(returns)
                
            except Exception as e:
                st.error(f"Error al calcular métricas de la cartera: {e}")

elif option == "Análisis Fundamental":
    st.title("Análisis Fundamental de Acciones")

    # Selección de ticker
    ticker = st.text_input("Introduce el ticker de la acción (ejemplo: AAPL)", "AAPL")

    if st.button("Obtener Datos Fundamentales"):
        if ticker:
            try:
                info = get_fundamental_data(ticker)
                
                st.write(f"**Información Fundamental del Ticker {ticker}:**")
                st.write(f"**Nombre:** {info.get('shortName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industria:** {info.get('industry', 'N/A')}")
                st.write(f"**Descripción:** {info.get('longBusinessSummary', 'N/A')}")
                st.write(f"**Precio Actual:** {info.get('currentPrice', 'N/A')}")
                st.write(f"**PE Ratio (TTM):** {info.get('trailingPE', 'N/A')}")
                st.write(f"**PEG Ratio (TTM):** {info.get('pegRatio', 'N/A')}")
                st.write(f"**Dividendo Yield (TTM):** {info.get('dividendYield', 'N/A')}")
                st.write(f"**Market Cap:** {info.get('marketCap', 'N/A')}")
                st.write(f"**Enterprise Value:** {info.get('enterpriseValue', 'N/A')}")
                st.write(f"**Revenue (TTM):** {info.get('totalRevenue', 'N/A')}")
                st.write(f"**Gross Profit:** {info.get('grossProfits', 'N/A')}")
                st.write(f"**EBITDA:** {info.get('ebitda', 'N/A')}")
                st.write(f"**Net Income (TTM):** {info.get('netIncomeToCommon', 'N/A')}")
                st.write(f"**Total Debt:** {info.get('totalDebt', 'N/A')}")
                st.write(f"**Cash and Cash Equivalents:** {info.get('cash', 'N/A')}")

            except Exception as e:
                st.error(f"Error al obtener datos fundamentales: {e}")
