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

# Función para obtener el precio actual de una acción
def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info.get('currentPrice', None)

# Función para calcular el valor total de la cartera
def calculate_portfolio_value(portfolio):
    total_value = 0
    for ticker, quantity in portfolio.items():
        price = get_current_price(ticker)
        if price is not None:
            total_value += price * quantity
    return total_value

# Configuración de la página
st.set_page_config(page_title="Dashboard Financiero", layout="wide")

# Selección de pestaña
selected_option = st.sidebar.selectbox('Seleccionar Sección', ['Análisis Técnico', 'Análisis Fundamental', 'Gestión de Carteras'])

# Control de flujo según la opción seleccionada
if selected_option == 'Análisis Técnico' or selected_option == 'Análisis Fundamental':
    # Entradas de usuario para análisis
    ticker = st.text_input("Símbolo bursátil:", value='AAPL')
    start_date = st.date_input('Fecha de inicio', (datetime.today() - timedelta(days=252)).date())
    end_date = st.date_input('Fecha de fin', datetime.today().date())

    try:
        hist, info = get_stock_data(ticker, start_date, end_date)
        data = calculate_technical_indicators(hist)

        if selected_option == 'Análisis Técnico':
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
            macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='MACD Señal', line=dict(color='red')))
            macd_fig.add_trace(go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histograma MACD', marker_color='rgba(255, 87, 34, 0.8)'))
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
                yaxis_title='Momentum',
                template='plotly_dark'
            )
            st.plotly_chart(momentum_fig)

            # ADX
            adx_fig = go.Figure()
            adx_fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], mode='lines', name='ADX', line=dict(color='blue')))
            adx_fig.add_trace(go.Scatter(x=data.index, y=data['ADX_Pos'], mode='lines', name='ADX Positivo', line=dict(color='green')))
            adx_fig.add_trace(go.Scatter(x=data.index, y=data['ADX_Neg'], mode='lines', name='ADX Negativo', line=dict(color='red')))
            adx_fig.update_layout(
                title=f'ADX de {ticker}',
                xaxis_title='Fecha',
                yaxis_title='Valor',
                template='plotly_dark'
            )
            st.plotly_chart(adx_fig)

            # CCI
            cci_fig = go.Figure()
            cci_fig.add_trace(go.Scatter(x=data.index, y=data['CCI'], mode='lines', name='CCI', line=dict(color='orange')))
            cci_fig.update_layout(
                title=f'CCI de {ticker}',
                xaxis_title='Fecha',
                yaxis_title='CCI',
                template='plotly_dark'
            )
            st.plotly_chart(cci_fig)

            # OBV
            obv_fig = go.Figure()
            obv_fig.add_trace(go.Scatter(x=data.index, y=data['OBV'], mode='lines', name='OBV', line=dict(color='cyan')))
            obv_fig.update_layout(
                title=f'OBV de {ticker}',
                xaxis_title='Fecha',
                yaxis_title='OBV',
                template='plotly_dark'
            )
            st.plotly_chart(obv_fig)

            # VWAP
            vwap_fig = go.Figure()
            vwap_fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP', line=dict(color='green')))
            vwap_fig.update_layout(
                title=f'VWAP de {ticker}',
                xaxis_title='Fecha',
                yaxis_title='VWAP',
                template='plotly_dark'
            )
            st.plotly_chart(vwap_fig)

        elif selected_option == 'Análisis Fundamental':
            # Mostrar la información fundamental
            st.subheader(f'Información Fundamental de {ticker}')
            st.write(f"**Nombre:** {info.get('shortName', 'No disponible')}")
            st.write(f"**Sector:** {info.get('sector', 'No disponible')}")
            st.write(f"**Industria:** {info.get('industry', 'No disponible')}")
            st.write(f"**Descripción:** {info.get('longBusinessSummary', 'No disponible')}")
            st.write(f"**Precio Actual:** ${get_current_price(ticker)}")

            # Mostrar la información financiera
            st.write(f"**P/E Ratio:** {info.get('forwardEps', 'No disponible')}")
            st.write(f"**Dividend Yield:** {info.get('dividendYield', 'No disponible')}")
            st.write(f"**Market Cap:** {info.get('marketCap', 'No disponible')}")
            st.write(f"**52-Week High:** {info.get('fiftyTwoWeekHigh', 'No disponible')}")
            st.write(f"**52-Week Low:** {info.get('fiftyTwoWeekLow', 'No disponible')}")

elif selected_option == 'Gestión de Carteras':
    # Inicializar la cartera
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = {}

    # Formulario para agregar o actualizar activos
    st.sidebar.header('Añadir/Actualizar Activo')
    ticker = st.sidebar.text_input('Símbolo bursátil:', '')
    quantity = st.sidebar.number_input('Cantidad:', min_value=0, value=0)

    if st.sidebar.button('Añadir/Actualizar'):
        if ticker and quantity > 0:
            st.session_state['portfolio'][ticker] = quantity
            st.sidebar.success(f'{ticker} añadido/actualizado en la cartera.')
        else:
            st.sidebar.error('Por favor, introduzca un símbolo bursátil válido y una cantidad mayor a 0.')

    # Mostrar la cartera
    st.subheader('Cartera Actual')
    if len(st.session_state['portfolio']) > 0:
        portfolio_df = pd.DataFrame.from_dict(st.session_state['portfolio'], orient='index', columns=['Cantidad'])
        portfolio_df.index.name = 'Símbolo Bursátil'
        st.write(portfolio_df)

        # Calcular y mostrar el valor total de la cartera
        total_value = calculate_portfolio_value(st.session_state['portfolio'])
        st.write(f'**Valor Total de la Cartera:** ${total_value:,.2f}')

        # Gráfico de la composición de la cartera
        import plotly.graph_objects as go

        portfolio_values = [get_current_price(ticker) * quantity for ticker, quantity in st.session_state['portfolio'].items()]
        portfolio_names = list(st.session_state['portfolio'].keys())

        pie_fig = go.Figure(data=[go.Pie(labels=portfolio_names, values=portfolio_values)])
        pie_fig.update_layout(title='Composición de la Cartera')
        st.plotly_chart(pie_fig)
    else:
        st.write('La cartera está vacía.')
