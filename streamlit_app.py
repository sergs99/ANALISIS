import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
import ta

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Financiero",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Función para mostrar gráficos con estilo
def plot_graph(fig, title, yaxis_title):
    fig.update_layout(
        title=title,
        xaxis_title='Fecha',
        yaxis_title=yaxis_title,
        template='plotly_dark',
        title_font_size=24,
        title_font_color='white',
        xaxis_title_font_size=18,
        yaxis_title_font_size=18
    )
    return fig

# Encabezado
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #00BFAE;
    }
    .subheader {
        font-size: 28px;
        font-weight: bold;
        color: #009688;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Dashboard Financiero</div>', unsafe_allow_html=True)

# Entradas de usuario en la barra lateral
st.sidebar.header('Configuración')
ticker = st.sidebar.text_input("Símbolo bursátil:", value='AAPL')
start_date = st.sidebar.date_input('Fecha de inicio', (datetime.today() - timedelta(days=252)).date())
end_date = st.sidebar.date_input('Fecha de fin', datetime.today().date())

# Mostrar datos y gráficos
try:
    hist, info = get_stock_data(ticker, start_date, end_date)
    data = calculate_technical_indicators(hist)

    # Tabs
    selected_tab = st.sidebar.selectbox('Seleccionar pestaña', ['Análisis Técnico', 'Fundamental'])

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
        price_fig = plot_graph(price_fig, f'Gráfico de Velas de {ticker}', 'Precio')
        st.plotly_chart(price_fig)

        # Gráfico de Volumen
        volume_fig = go.Figure()
        volume_fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volumen de Negociación', marker_color='rgba(255, 87, 34, 0.8)'))
        volume_fig = plot_graph(volume_fig, f'Volumen de Negociación de {ticker}', 'Volumen')
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
        bollinger_fig = plot_graph(bollinger_fig, f'Bandas de Bollinger de {ticker}', 'Precio')
        st.plotly_chart(bollinger_fig)

        # RSI
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='magenta')))
        rsi_fig.add_hline(y=70, line_dash='dash', line_color='red')
        rsi_fig.add_hline(y=30, line_dash='dash', line_color='green')
        rsi_fig = plot_graph(rsi_fig, f'Índice de Fuerza Relativa (RSI) de {ticker}', 'RSI')
        st.plotly_chart(rsi_fig)

        # Oscilador Estocástico
        stochastic_fig = go.Figure()
        stochastic_fig.add_trace(go.Scatter(x=data.index, y=data['Stoch_K'], mode='lines', name='%K', line=dict(color='yellow')))
        stochastic_fig.add_trace(go.Scatter(x=data.index, y=data['Stoch_D'], mode='lines', name='%D', line=dict(color='lightcoral')))
        stochastic_fig = plot_graph(stochastic_fig, f'Oscilador Estocástico de {ticker}', 'Valor')
        st.plotly_chart(stochastic_fig)

        # MACD
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
        macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='MACD Signal', line=dict(color='orange')))
        macd_fig.add_trace(go.Bar(x=data.index, y=data['MACD_Histogram'], name='MACD Histogram', marker_color='rgba(255, 87, 34, 0.8)'))
        macd_fig = plot_graph(macd_fig, f'MACD de {ticker}', 'Valor')
        st.plotly_chart(macd_fig)

        # Momentum
        momentum_fig = go.Figure()
        momentum_fig.add_trace(go.Scatter(x=data.index, y=data['Momentum'], mode='lines', name='Momentum', line=dict(color='purple')))
        momentum_fig = plot_graph(momentum_fig, f'Momentum de {ticker}', 'Valor')
        st.plotly_chart(momentum_fig)

        # ADX
        adx_fig = go.Figure()
        adx_fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], mode='lines', name='ADX', line=dict(color='blue')))
        adx_fig.add_trace(go.Scatter(x=data.index, y=data['ADX_Pos'], mode='lines', name='ADX+', line=dict(color='green')))
        adx_fig.add_trace(go.Scatter(x=data.index, y=data['ADX_Neg'], mode='lines', name='ADX-', line=dict(color='red')))
        adx_fig = plot_graph(adx_fig, f'ADX de {ticker}', 'ADX')
        st.plotly_chart(adx_fig)

        # CCI
        cci_fig = go.Figure()
        cci_fig.add_trace(go.Scatter(x=data.index, y=data['CCI'], mode='lines', name='CCI', line=dict(color='orange')))
        cci_fig.add_hline(y=100, line_dash='dash', line_color='red')
        cci_fig.add_hline(y=-100, line_dash='dash', line_color='green')
        cci_fig = plot_graph(cci_fig, f'Índice de Canal de Commodities (CCI) de {ticker}', 'CCI')
        st.plotly_chart(cci_fig)

        # OBV
        obv_fig = go.Figure()
        obv_fig.add_trace(go.Scatter(x=data.index, y=data['OBV'], mode='lines', name='OBV', line=dict(color='blue')))
        obv_fig = plot_graph(obv_fig, f'On-Balance Volume (OBV) de {ticker}', 'OBV')
        st.plotly_chart(obv_fig)

        # VWAP
        vwap_fig = go.Figure()
        vwap_fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP', line=dict(color='cyan')))
        vwap_fig = plot_graph(vwap_fig, f'Precio Promedio Ponderado por Volumen (VWAP) de {ticker}', 'VWAP')
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
        st.write("Aquí tienes un resumen del análisis fundamental.")

        for category, metrics in fundamental_data.items():
            st.write(f"**{category}:**")
            if isinstance(metrics, dict):
                st.write(pd.DataFrame(list(metrics.items()), columns=['Métrica', 'Valor']).set_index('Métrica'))
            else:
                st.write(metrics)

except Exception as e:
    st.error(f"Error: {e}")
