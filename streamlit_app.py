import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize
from datetime import datetime

# Funciones para el análisis de carteras
def get_user_input():
    tickers = st.text_input("Introduce los tickers de las acciones (separados por comas):")
    weights_input = st.text_input("Introduce los pesos de las acciones (separados por comas, deben sumar 1):")
    risk_free_rate_input = st.text_input("Introduce la tasa libre de riesgo actual (como fracción, ej. 0.0234 para 2.34%):")

    if tickers and weights_input and risk_free_rate_input:
        try:
            tickers = [ticker.strip().upper() for ticker in tickers.split(',')]
            weights = np.array([float(weight.strip()) for weight in weights_input.split(',')])

            if not np.isclose(sum(weights), 1.0, atol=1e-5):
                st.error("La suma de los pesos debe ser aproximadamente igual a 1.0.")
                return None, None, None

            risk_free_rate = float(risk_free_rate_input.strip())

            return tickers, weights, risk_free_rate
        
        except ValueError as e:
            st.error(f"Error: {e}")
            return None, None, None

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

# Función de visualización
def plot_portfolio_data(portfolio_return, portfolio_volatility, correlation_matrix):
    st.subheader("Rentabilidad y Volatilidad")
    st.bar_chart({"Rentabilidad": portfolio_return * 100, "Volatilidad": portfolio_volatility * 100})

    st.subheader("Matriz de Correlación")
    st.write(correlation_matrix)

def plot_cml_sml(portfolio_return, portfolio_volatility, market_returns, risk_free_rate):
    market_return = market_returns.mean() * 252
    market_volatility = market_returns.std() * np.sqrt(252)

    volatilities = np.linspace(0, market_volatility * 2, 100)
    returns_cml = risk_free_rate + (portfolio_return - risk_free_rate) / portfolio_volatility * volatilities

    returns_sml = np.linspace(risk_free_rate, market_return * 1.5, 100)
    volatilities_sml = (returns_sml - risk_free_rate) / market_return * market_volatility

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=volatilities, y=returns_cml, mode='lines', name='Capital Market Line (CML)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=volatilities_sml, y=returns_sml, mode='lines', name='Security Market Line (SML)', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=[portfolio_volatility], y=[portfolio_return], mode='markers', marker=dict(color='green', size=10), name='Cartera'))
    fig.add_trace(go.Scatter(x=[market_volatility], y=[market_return], mode='markers', marker=dict(color='orange', size=10), name='Mercado'))

    fig.update_layout(title='Capital Market Line (CML) y Security Market Line (SML)', xaxis_title='Volatilidad', yaxis_title='Retorno')
    st.plotly_chart(fig)

# App Streamlit
st.title('Análisis Financiero y Gestión de Carteras')

menu = st.sidebar.selectbox("Selecciona una opción", ['Análisis Técnico', 'Análisis Fundamental', 'Gestión de Carteras'])

if menu == 'Gestión de Carteras':
    st.subheader('Gestión de Carteras')

    tickers, weights, risk_free_rate = get_user_input()

    if tickers is not None and weights is not None and risk_free_rate is not None:
        # Calcular métricas de la cartera
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

            st.write("\nComposición óptima de la cartera:")
            for ticker, weight in zip(tickers, optimal_weights):
                st.write(f"{ticker}: {weight:.2%}")

            st.write(f"\nRentabilidad media anualizada de la cartera óptima: {optimal_return * 100:.2f}%")
            st.write(f"Volatilidad anualizada de la cartera óptima: {optimal_volatility * 100:.2f}%")
            st.write(f"Ratio de Sharpe de la cartera óptima: {optimal_sharpe_ratio:.2f}")
            st.write(f"Ratio de Sortino de la cartera óptima: {optimal_sortino_ratio:.2f}")
            st.write(f"Ratio de Treynor de la cartera óptima: {optimal_treynor_ratio:.2f}")

            # Graficar CML y SML
            plot_cml_sml(portfolio_return, portfolio_volatility, market_returns, risk_free_rate)

else:
    st.write('Selecciona una opción del menú para mostrar el contenido.')
