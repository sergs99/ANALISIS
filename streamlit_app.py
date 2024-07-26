import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import ta

# Función para obtener datos fundamentales
def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
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
    return fundamental_data

# Función para obtener datos históricos
def get_historical_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    if hist.empty:
        return None
    return hist

# Configuración de Streamlit
st.title('Dashboard Financiero')

ticker = st.text_input('Símbolo bursátil:', 'AAPL')
start_date = st.date_input('Fecha de inicio', value=datetime.today() - timedelta(days=252))
end_date = st.date_input('Fecha de fin', value=datetime.today())

if ticker:
    # Obtener datos históricos
    hist = get_historical_data(ticker, start_date, end_date)
    if hist is not None:
        st.subheader('Datos Históricos')
        st.line_chart(hist['Close'])
        st.write(hist)

    # Obtener y mostrar datos fundamentales
    fundamental_data = get_fundamental_data(ticker)

    st.subheader('Análisis Fundamental')

    st.write(f"**Nombre:** {fundamental_data['Nombre']}")
    st.write(f"**Sector:** {fundamental_data['Sector']}")
    st.write(f"**Industria:** {fundamental_data['Industria']}")
    st.write(f"**Precio Actual:** {fundamental_data['Precio Actual']}")

    st.write("**Ratios de Valoración:**")
    for key, value in fundamental_data['Ratios de Valoración'].items():
        st.write(f"{key}: {value}")

    st.write("**Ratios de Rentabilidad:**")
    for key, value in fundamental_data['Ratios de Rentabilidad'].items():
        st.write(f"{key}: {value}")

    st.write("**Ratios de Liquidez y Solvencia:**")
    for key, value in fundamental_data['Ratios de Liquidez y Solvencia'].items():
        st.write(f"{key}: {value}")

    st.write("**Otras Métricas:**")
    for key, value in fundamental_data['Otras Métricas'].items():
        st.write(f"{key}: {value}")

    st.write('Fin de los Datos Fundamentales')
