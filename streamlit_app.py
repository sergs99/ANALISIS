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
    selected_tab = st.selectbox('Seleccionar pestaña', ['Análisis Técnico', 'Análisis Fundamental', 'Gestión de Carteras'])

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
        cci_fig = update_layout(cci_fig, f'CCI de {ticker}', 'CCI')
        st.plotly_chart(cci_fig)

        # OBV
        obv_fig = go.Figure()
        obv_fig.add_trace(go.Scatter(x=data.index, y=data['OBV'], mode='lines', name='OBV', line=dict(color='cyan')))
        obv_fig = update_layout(obv_fig, f'OBV de {ticker}', 'OBV')
        st.plotly_chart(obv_fig)

        # VWAP
        vwap_fig = go.Figure()
        vwap_fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP', line=dict(color='yellow')))
        vwap_fig = update_layout(vwap_fig, f'VWAP de {ticker}', 'VWAP')
        st.plotly_chart(vwap_fig)

    elif selected_tab == 'Análisis Fundamental':
        # Datos fundamentales
        st.subheader(f'Información Fundamental de {ticker}')
        info_df = pd.DataFrame.from_dict(info, orient='index', columns=['Valor'])
        st.write(info_df)

    elif selected_tab == 'Gestión de Carteras':
        # Selección de acciones
        st.subheader('Gestión de Carteras')
        tickers = st.text_area('Ingrese los símbolos bursátiles separados por comas (por ejemplo: AAPL, MSFT, TSLA)').split(',')

        if tickers:
            tickers = [ticker.strip().upper() for ticker in tickers]
            start_date = st.date_input('Fecha de inicio de cartera', (datetime.today() - timedelta(days=365)).date())
            end_date = st.date_input('Fecha de fin de cartera', datetime.today().date())
            portfolio_data = {}

            for ticker in tickers:
                try:
                    hist, _ = get_stock_data(ticker, start_date, end_date)
                    portfolio_data[ticker] = hist['Close']
                except Exception as e:
                    st.error(f'Error al obtener datos para {ticker}: {e}')

            if portfolio_data:
                # DataFrame de cartera
                portfolio_df = pd.DataFrame(portfolio_data)
                portfolio_returns = portfolio_df.pct_change().dropna()
                st.subheader('Rendimiento de la Cartera')
                st.line_chart(portfolio_df)

                # Gráfico de correlación
                st.subheader('Correlación de Rendimientos')
                corr_matrix = portfolio_returns.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

                # Optimización de la cartera
                def portfolio_performance(weights, returns):
                    port_return = np.sum(returns.mean() * weights) * 252
                    port_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
                    return port_volatility, port_return

                def negative_sharpe_ratio(weights, returns):
                    port_volatility, port_return = portfolio_performance(weights, returns)
                    return -(port_return / port_volatility)

                def optimize_portfolio(returns):
                    num_assets = len(returns.columns)
                    args = (returns)
                    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                    bounds = tuple((0, 1) for _ in range(num_assets))
                    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
                    return result

                optimal_result = optimize_portfolio(portfolio_returns)
                st.subheader('Optimización de la Cartera')
                st.write('Pesos óptimos:')
                st.write(pd.Series(optimal_result.x, index=portfolio_returns.columns))
                optimal_volatility, optimal_return = portfolio_performance(optimal_result.x, portfolio_returns)
                st.write(f'Rendimiento esperado: {optimal_return:.2f}')
                st.write(f'Volatilidad esperada: {optimal_volatility:.2f}')

except Exception as e:
    st.error(f'Ocurrió un error: {e}')
