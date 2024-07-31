import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
import ta
import numpy as np
from scipy.optimize import minimize

# Función para obtener datos históricos y la información del stock
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    info = stock.info
    return hist, info

Error: could not convert string to float: ''

# Funciones adicionales para la gestión de carteras
def get_user_input():
    while True:
        try:
            tickers = st.text_input("Introduce los tickers de las acciones (separados por comas)").split(',')
            weights = st.text_input("Introduce los pesos de las acciones (separados por comas, deben sumar 1)").split(',')
            
            tickers = [ticker.strip().upper() for ticker in tickers]
            weights = np.array([float(weight.strip()) for weight in weights])

            if not np.isclose(sum(weights), 1.0, atol=1e-5):
                st.error("La suma de los pesos debe ser aproximadamente igual a 1.0.")
                return None, None, None

            risk_free_rate = float(st.text_input("Introduce la tasa libre de riesgo actual (como fracción, ej. 0.0234 para 2.34%)").strip())

            return tickers, weights, risk_free_rate
        
        except ValueError as e:
            st.error(f"Error: {e}")
            st.write("Por favor, ingresa los datos nuevamente.")
            return None, None, None

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
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].bar(["Rentabilidad", "Volatilidad"], [portfolio_return * 100, portfolio_volatility * 100], color=['blue', 'orange'])
    axes[0].set_title("Rentabilidad y Volatilidad")
    axes[0].set_ylabel('Porcentaje')

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', ax=axes[1])
    axes[1].set_title("Matriz de Correlación")
    
    st.pyplot(fig)

def plot_cml_sml(portfolio_return, portfolio_volatility, market_returns, risk_free_rate):
    market_return = market_returns.mean() * 252
    market_volatility = market_returns.std() * np.sqrt(252)

    volatilities = np.linspace(0, market_volatility * 2, 100)
    returns_cml = risk_free_rate + (portfolio_return - risk_free_rate) / portfolio_volatility * volatilities

    returns_sml = np.linspace(risk_free_rate, market_return * 1.5, 100)
    volatilities_sml = (returns_sml - risk_free_rate) / (market_return - risk_free_rate) * market_volatility

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=volatilities, y=returns_cml, mode='lines', name='CML'))
    fig.add_trace(go.Scatter(x=volatilities_sml, y=returns_sml, mode='lines', name='SML', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=[portfolio_volatility], y=[portfolio_return], mode='markers+text', name='Portafolio Óptimo', text='Óptimo', textposition='top right', marker=dict(color='red', size=10)))
    
    fig.update_layout(title='Capital Market Line y Security Market Line', xaxis_title='Volatilidad', yaxis_title='Rentabilidad')
    st.plotly_chart(fig)

def check_normality(returns):
    from scipy.stats import normaltest
    stat, p_value = normaltest(returns)
    st.write(f'Prueba de normalidad: Estadístico = {stat:.3f}, p-valor = {p_value:.3f}')

# Configuración de la página
st.set_page_config(page_title="Dashboard Financiero", layout="wide")

st.title('Análisis Financiero')

# Menú de navegación
menu = st.sidebar.selectbox('Selecciona una opción', ['Análisis Técnico', 'Análisis Fundamental', 'Gestión de Carteras'])

if menu == 'Análisis Técnico':
    ticker = st.text_input("Símbolo bursátil:", value='AAPL')
    start_date = st.date_input('Fecha de inicio', (datetime.today() - timedelta(days=252)).date())
    end_date = st.date_input('Fecha de fin', datetime.today().date())

    # Mostrar datos y gráficos
    try:
        hist, info = get_stock_data(ticker, start_date, end_date)
        data = calculate_technical_indicators(hist)

        # Gráficos
        # (Los gráficos aquí se mantienen igual que en el código original)

    except Exception as e:
        st.error(f"Error: {e}")

elif menu == 'Análisis Fundamental':
    ticker = st.text_input("Símbolo bursátil:", value='AAPL')
    start_date = st.date_input('Fecha de inicio', (datetime.today() - timedelta(days=252)).date())
    end_date = st.date_input('Fecha de fin', datetime.today().date())

    # Mostrar datos y gráficos
    try:
        hist, info = get_stock_data(ticker, start_date, end_date)

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

elif menu == 'Gestión de Carteras':
    st.subheader('Gestión de Carteras')

    # Entradas del usuario para la gestión de carteras
    tickers, weights, risk_free_rate = get_user_input()
    if tickers and weights:
        returns, annualized_return, annualized_volatility, correlation_matrix, market_returns, portfolio_returns = calculate_portfolio_metrics(tickers, weights)

        if returns is not None:
            # Resultados de la cartera
            st.write(f"Rentabilidad Anualizada: {annualized_return * 100:.2f}%")
            st.write(f"Volatilidad Anualizada: {annualized_volatility * 100:.2f}%")

            # Mostrar gráficos y métricas
            plot_portfolio_data(annualized_return, annualized_volatility, correlation_matrix)
            st.write(f"Ratio de Sharpe: {calculate_sharpe_ratio(annualized_return, annualized_volatility, risk_free_rate):.2f}")
            st.write(f"Ratio de Sortino: {calculate_sortino_ratio(portfolio_returns, risk_free_rate):.2f}")
            st.write(f"Ratio de Treynor: {calculate_treynor_ratio(portfolio_returns, market_returns, risk_free_rate):.2f}")

            # Optimización de la cartera
            optimal_weights = optimize_portfolio(returns, risk_free_rate)
            optimal_portfolio_returns = returns.dot(optimal_weights)
            optimal_return = optimal_portfolio_returns.mean() * 252
            optimal_volatility = optimal_portfolio_returns.std() * np.sqrt(252)
            optimal_sharpe_ratio = calculate_sharpe_ratio(optimal_return, optimal_volatility, risk_free_rate)
            
            # Mostrar resultados de la cartera óptima
            st.write(f"\nCartera Óptima:\n")
            st.write(f"Pesos Óptimos: {optimal_weights}")
            st.write(f"Rentabilidad Óptima: {optimal_return:.2f}")
            st.write(f"Volatilidad Óptima: {optimal_volatility:.2f}")
            st.write(f"Ratio de Sharpe Óptimo: {optimal_sharpe_ratio:.2f}")

            # Graficar CML y SML
            plot_cml_sml(optimal_return, optimal_volatility, market_returns, risk_free_rate)
            
            # Verificar normalidad
            check_normality(portfolio_returns)

            st.write("Análisis completado.")
