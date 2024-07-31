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
    try:
        tickers_input = st.text_input("Introduce los tickers de las acciones (separados por comas):")
        weights_input = st.text_input("Introduce los pesos de las acciones (separados por comas, deben sumar 1):")
        risk_free_rate_input = st.text_input("Introduce la tasa libre de riesgo actual (como fracción, ej. 0.0234 para 2.34%):")
        
        if not tickers_input or not weights_input or not risk_free_rate_input:
            st.warning("Por favor, completa todos los campos.")
            return None, None, None
        
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
        weights = np.array([float(weight.strip()) for weight in weights_input.split(',')])
        risk_free_rate = float(risk_free_rate_input.strip())

        if not np.isclose(sum(weights), 1.0, atol=1e-5):
            st.error("La suma de los pesos debe ser aproximadamente igual a 1.0.")
            return None, None, None

        return tickers, weights, risk_free_rate
    
    except ValueError as e:
        st.error(f"Error en los datos proporcionados: {e}")
        return None, None, None

def calculate_portfolio_metrics(tickers, weights):
    end_date = datetime.today().strftime('%Y-%m-%d')
    tickers_with_market = tickers + ['^GSPC']
    try:
        data = yf.download(tickers_with_market, start='2022-01-01', end=end_date)['Adj Close']
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        st.error("Asegúrate de que los tickers son válidos y que tienes una conexión a Internet estable.")
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
    st.pyplot(plt.gcf())

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
    st.pyplot(plt.gcf())

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
    st.pyplot(plt.gcf())

# Solicitar entrada del usuario
tickers, weights, risk_free_rate = get_user_input()

if tickers is not None and weights is not None and risk_free_rate is not None:
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

            # Verificar normalidad de los retornos
            check_normality(returns[tickers])
