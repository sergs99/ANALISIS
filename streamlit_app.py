import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
import ta

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

st.title('Análisis Financiero')

# Entradas de usuario
ticker = st.text_input("Símbolo bursátil:", value='AAPL')
start_date = st.date_input('Fecha de inicio', (datetime.today() - timedelta(days=252)).date())
end_date = st.date_input('Fecha de fin', datetime.today().date())

# Función para actualizar el diseño de los gráficos
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
            increasing_line_color='lime',
            decreasing_line_color='red',
            name='Candlestick'
        )])
        price_fig = update_layout(price_fig, f'Gráfico de Velas de {ticker}', 'Precio')
        st.plotly_chart(price_fig)

        # Gráfico de Volumen
        volume_fig = go.Figure()
        volume_fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volumen de Negociación', marker_color='rgba(255, 87, 34, 0.8)'))
        volume_fig = update_layout(volume_fig, f'Volumen de Negociación de {ticker}', 'Volumen')
        st.plotly_chart(volume_fig)

        # Bandas de Bollinger
        bollinger_fig = go.Figure()
        bollinger_fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            increasing_line_color='lime',
            decreasing_line_color='red',
            name='Candlestick'
        ))
        bollinger_fig.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'], mode='lines', name='Media Móvil', line=dict(color='cyan')))
        bollinger_fig.add_trace(go.Scatter(x=data.index, y=data['BB_High'], mode='lines', name='Banda Superior', line=dict(color='red')))
        bollinger_fig.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], mode='lines', name='Banda Inferior', line=dict(color='green')))
        bollinger_fig = update_layout(bollinger_fig, f'Bandas de Bollinger de {ticker}', 'Precio')
        st.plotly_chart(bollinger_fig)

        # MACD
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
        macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='Línea de Señal', line=dict(color='red')))
        macd_fig.add_trace(go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histograma', marker_color='rgba(255, 193, 7, 0.5)'))
        macd_fig = update_layout(macd_fig, f'MACD de {ticker}', 'MACD')
        st.plotly_chart(macd_fig)

        # RSI
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
        rsi_fig.add_hline(y=70, line_dash='dash', line_color='red')
        rsi_fig.add_hline(y=30, line_dash='dash', line_color='green')
        rsi_fig = update_layout(rsi_fig, f'RSI de {ticker}', 'RSI')
        st.plotly_chart(rsi_fig)

        # Momentum
        momentum_fig = go.Figure()
        momentum_fig.add_trace(go.Scatter(x=data.index, y=data['Momentum'], mode='lines', name='Momentum', line=dict(color='magenta')))
        momentum_fig = update_layout(momentum_fig, f'Momentum de {ticker}', 'Valor')
        st.plotly_chart(momentum_fig)

        # ADX
        adx_fig = go.Figure()
        adx_fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], mode='lines', name='ADX', line=dict(color='blue')))
        adx_fig.add_trace(go.Scatter(x=data.index, y=data['ADX_Pos'], mode='lines', name='ADX+', line=dict(color='green')))
        adx_fig.add_trace(go.Scatter(x=data.index, y=data['ADX_Neg'], mode='lines', name='ADX-', line=dict(color='red')))
        adx_fig = update_layout(adx_fig, f'ADX de {ticker}', 'ADX')
        st.plotly_chart(adx_fig)

        # CCI
        cci_fig = go.Figure()
        cci_fig.add_trace(go.Scatter(x=data.index, y=data['CCI'], mode='lines', name='CCI', line=dict(color='orange')))
        cci_fig.add_hline(y=100, line_dash='dash', line_color='red')
        cci_fig.add_hline(y=-100, line_dash='dash', line_color='green')
        cci_fig = update_layout(cci_fig, f'Índice de Canal de Commodities (CCI) de {ticker}', 'CCI')
        st.plotly_chart(cci_fig)

        # OBV
        obv_fig = go.Figure()
        obv_fig.add_trace(go.Scatter(x=data.index, y=data['OBV'], mode='lines', name='OBV', line=dict(color='blue')))
        obv_fig = update_layout(obv_fig, f'On-Balance Volume (OBV) de {ticker}', 'OBV')
        st.plotly_chart(obv_fig)

        # VWAP
        vwap_fig = go.Figure()
        vwap_fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP', line=dict(color='cyan')))
        vwap_fig = update_layout(vwap_fig, f'Precio Promedio Ponderado por Volumen (VWAP) de {ticker}', 'VWAP')
        st.plotly_chart(vwap_fig)

    elif selected_tab == 'Fundamental':
        # Información financiera organizada
        fundamental_data = {
            'Nombre': info.get('shortName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industria': info.get('industry', 'N/A'),
            'Precio Actual': f"${info.get('currentPrice', 'N/A'):.2f}" if 'currentPrice' in info else 'N/A',
            'Ratios de Valoración': {
                'Price Earnings Ratio': info.get('trailingPE', 'N/A'),
                'Dividend Yield': f"{info.get('dividendYield', 'N/A')*100:.2f}%" if info.get('dividendYield') else 'N/A',
                'Price to Book Value': info.get('priceToBook', 'N/A'),
                'PEG Ratio (5yr expected)': info.get('pegRatio', 'N/A'),
                'Price to Cash Flow Ratio': info.get('priceToCashflow', 'N/A'),
                'EV/EBITDA': info.get('enterpriseToEbitda', 'N/A')
            },
            'Ratios de Rentabilidad': {
                'Return on Equity': f"{info.get('returnOnEquity', 'N/A')*100:.2f}%" if info.get('returnOnEquity') else 'N/A',
                'Return on Assets': f"{info.get('returnOnAssets', 'N/A')*100:.2f}%" if info.get('returnOnAssets') else 'N/A',
                'Profit Margin': f"{info.get('profitMargins', 'N/A')*100:.2f}%" if info.get('profitMargins') else 'N/A',
                'Operating Margin (ttm)': f"{info.get('operatingMargins', 'N/A')*100:.2f}%" if info.get('operatingMargins') else 'N/A',
                'Payout Ratio': f"{info.get('payoutRatio', 'N/A')*100:.2f}%" if info.get('payoutRatio') else 'N/A'
            },
            'Ratios de Liquidez y Solvencia': {
                'Current Ratio (mrq)': info.get('currentRatio', 'N/A'),
                'Total Debt/Equity (mrq)': info.get('debtToEquity', 'N/A')
            },
            'Otras Métricas': {
                'Volumen Actual': f"{info.get('volume', 'N/A'):,}" if 'volume' in info else 'N/A',
                'Earnings Per Share (EPS)': info.get('trailingEps', 'N/A'),
                'Capitalización de Mercado': f"${info.get('marketCap', 'N/A') / 1e9:.2f} B" if info.get('marketCap') else 'N/A',
                'Beta': info.get('beta', 'N/A')
            }
        }

        # Mostrar la información en una tabla
        st.subheader(f"Análisis Fundamental de {ticker}")

        for category, metrics in fundamental_data.items():
            st.write(f"**{category}:**")
            if isinstance(metrics, dict):
                st.write(pd.DataFrame(list(metrics.items()), columns=['Métrica', 'Valor']).set_index('Métrica'))
            else:
                st.write(metrics)

except Exception as e:
    st.error(f"Error: {e}")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize
from datetime import datetime
import streamlit as st

def get_user_input():
    while True:
        try:
            tickers = st.text_input("Introduce los tickers de las acciones (separados por comas):").split(',')
            weights = st.text_input("Introduce los pesos de las acciones (separados por comas, deben sumar 1):").split(',')
            
            tickers = [ticker.strip().upper() for ticker in tickers]
            weights = np.array([float(weight.strip()) for weight in weights])

            if not np.isclose(sum(weights), 1.0, atol=1e-5):
                raise ValueError("La suma de los pesos debe ser aproximadamente igual a 1.0.")

            risk_free_rate = float(st.text_input("Introduce la tasa libre de riesgo actual (como fracción, ej. 0.0234 para 2.34%):").strip())

            return tickers, weights, risk_free_rate
        
        except ValueError as e:
            st.error(f"Error: {e}")
            st.warning("Por favor, ingresa los datos nuevamente.")

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
    st.pyplot()

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
    st.pyplot()

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
    st.pyplot()

def main():
    st.sidebar.title("Menú")
    app_mode = st.sidebar.selectbox("Selecciona una opción:", ["Gestión de Carteras", "Análisis Técnico", "Análisis Fundamental"])

    if app_mode == "Gestión de Carteras":
        tickers, weights, risk_free_rate = get_user_input()

        if tickers and weights is not None:
            # Calcular métricas de la cartera inicial
            returns, portfolio_return, portfolio_volatility, correlation_matrix, market_returns, portfolio_returns = calculate_portfolio_metrics(tickers, weights)

            if returns is not None:
                # Mostrar resultados de la cartera inicial
                plot_portfolio_data(portfolio_return, portfolio_volatility, correlation_matrix)

                # Calcular y mostrar el Ratio de Sharpe para la cartera inicial
                sharpe_ratio = calculate_sharpe_ratio(portfolio_return, portfolio_volatility, risk_free_rate)
                st.write(f"Ratio de Sharpe: {sharpe_ratio:.2f}")

                # Calcular y mostrar el Ratio de Sortino para la cartera inicial
                sortino_ratio = calculate_sortino_ratio(portfolio_returns, risk_free_rate)
                st.write(f"Ratio de Sortino: {sortino_ratio:.2f}")

                # Calcular y mostrar el Ratio de Treynor para la cartera inicial
                treynor_ratio = calculate_treynor_ratio(portfolio_returns, market_returns, risk_free_rate)
                st.write(f"Ratio de Treynor: {treynor_ratio:.2f}")

                # Optimizar la cartera
                optimal_weights = optimize_portfolio(returns[tickers], risk_free_rate)

                if optimal_weights is not None:
                    # Calcular métricas de la cartera óptima
                    optimal_portfolio_returns = returns[tickers].dot(optimal_weights)
                    optimal_return = optimal_portfolio_returns.mean() * 252
                    optimal_volatility = optimal_portfolio_returns.std() * np.sqrt(252)

                    # Calcular el Ratio de Sharpe para la cartera óptima
                    optimal_sharpe_ratio = calculate_sharpe_ratio(optimal_return, optimal_volatility, risk_free_rate)

                    # Calcular el Ratio de Sortino para la cartera óptima
                    optimal_sortino_ratio = calculate_sortino_ratio(optimal_portfolio_returns, risk_free_rate)

                    # Calcular el Ratio de Treynor para la cartera óptima
                    optimal_treynor_ratio = calculate_treynor_ratio(optimal_portfolio_returns, market_returns, risk_free_rate)

                    st.write("Composición óptima de la cartera:")
                    for ticker, weight in zip(tickers, optimal_weights):
                        st.write(f"{ticker}: {weight:.2%}")

                    st.write(f"Rentabilidad media anualizada de la cartera óptima: {optimal_return * 100:.2f}%")
                    st.write(f"Volatilidad anualizada de la cartera óptima: {optimal_volatility * 100:.2f}%")
                    st.write(f"Ratio de Sharpe de la cartera óptima: {optimal_sharpe_ratio:.2f}")
                    st.write(f"Ratio de Sortino de la cartera óptima: {optimal_sortino_ratio:.2f}")
                    st.write(f"Ratio de Treynor de la cartera óptima: {optimal_treynor_ratio:.2f}")

                    # Graficar CML y SML
                    plot_cml_sml(optimal_return, optimal_volatility, market_returns, risk_free_rate)

                    # Verificar normalidad de los retornos
                    check_normality(returns[tickers])

    elif app_mode == "Análisis Técnico":
        st.subheader("Análisis Técnico")
        # Aquí puedes incluir funciones y gráficos para análisis técnico

    elif app_mode == "Análisis Fundamental":
        st.subheader("Análisis Fundamental")
        # Aquí puedes incluir funciones y gráficos para análisis fundamental

if __name__ == "__main__":
    main()

