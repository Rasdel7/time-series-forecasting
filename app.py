import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Time Series Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Time Series Forecasting App")
st.markdown("Forecast any time series data "
            "using Facebook Prophet — "
            "used by Meta and Airbnb.")
st.markdown("---")

# Generate sample datasets
def generate_sales_data(periods=365):
    dates = pd.date_range(
        start='2022-01-01',
        periods=periods,
        freq='D'
    )
    trend     = np.linspace(100, 200, periods)
    seasonal  = 30 * np.sin(
        2 * np.pi * np.arange(periods) / 365)
    weekly    = 15 * np.sin(
        2 * np.pi * np.arange(periods) / 7)
    noise     = np.random.normal(0, 10, periods)
    values    = trend + seasonal + weekly + noise
    return pd.DataFrame({
        'ds': dates,
        'y':  np.maximum(values, 10)
    })

def generate_stock_data(periods=365):
    dates  = pd.date_range(
        start='2022-01-01',
        periods=periods,
        freq='D'
    )
    prices = [100]
    for _ in range(periods - 1):
        change = np.random.normal(0.001, 0.02)
        prices.append(
            prices[-1] * (1 + change))
    return pd.DataFrame({
        'ds': dates,
        'y':  prices
    })

def generate_temperature_data(periods=365):
    dates = pd.date_range(
        start='2022-01-01',
        periods=periods,
        freq='D'
    )
    base_temp = 25
    seasonal  = 15 * np.sin(
        2 * np.pi * np.arange(periods) / 365
        - np.pi / 2)
    noise     = np.random.normal(0, 3, periods)
    temps     = base_temp + seasonal + noise
    return pd.DataFrame({
        'ds': dates,
        'y':  temps
    })

def generate_website_traffic(periods=365):
    dates   = pd.date_range(
        start='2022-01-01',
        periods=periods,
        freq='D'
    )
    trend   = np.linspace(1000, 5000, periods)
    weekly  = 500 * np.sin(
        2 * np.pi * np.arange(periods) / 7)
    monthly = 300 * np.sin(
        2 * np.pi * np.arange(periods) / 30)
    noise   = np.random.normal(0, 150, periods)
    traffic = trend + weekly + monthly + noise
    return pd.DataFrame({
        'ds': dates,
        'y':  np.maximum(traffic, 100)
    })

DATASETS = {
    "Sales Data":        generate_sales_data,
    "Stock Prices":      generate_stock_data,
    "Temperature":       generate_temperature_data,
    "Website Traffic":   generate_website_traffic
}

# Sidebar
st.sidebar.header("⚙️ Forecast Settings")

data_source = st.sidebar.radio(
    "Data source:",
    ["Use sample data", "Upload CSV"]
)

if data_source == "Use sample data":
    dataset_name = st.sidebar.selectbox(
        "Select dataset:",
        list(DATASETS.keys())
    )
    np.random.seed(42)
    df = DATASETS[dataset_name]()
else:
    uploaded = st.sidebar.file_uploader(
        "Upload CSV (needs 'date' and "
        "'value' columns):",
        type=['csv']
    )
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.sidebar.success(
            f"Loaded {len(df_raw)} rows!")

        date_col = st.sidebar.selectbox(
            "Date column:",
            df_raw.columns.tolist()
        )
        value_col = st.sidebar.selectbox(
            "Value column:",
            df_raw.columns.tolist()
        )
        df = df_raw[[date_col, value_col]]\
            .copy()
        df.columns = ['ds', 'y']
        df['ds']   = pd.to_datetime(df['ds'])
        df['y']    = pd.to_numeric(
            df['y'], errors='coerce')
        df.dropna(inplace=True)
        dataset_name = "Uploaded Data"
    else:
        st.sidebar.info(
            "Upload a CSV or use "
            "sample data.")
        np.random.seed(42)
        dataset_name = "Sales Data"
        df = DATASETS[dataset_name]()

forecast_days = st.sidebar.slider(
    "Forecast period (days):",
    7, 365, 90, 7
)
include_uncertainty = st.sidebar.checkbox(
    "Show uncertainty intervals",
    value=True
)
yearly_seasonality  = st.sidebar.checkbox(
    "Yearly seasonality", value=True)
weekly_seasonality  = st.sidebar.checkbox(
    "Weekly seasonality", value=True)
daily_seasonality   = st.sidebar.checkbox(
    "Daily seasonality", value=False)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview",
    "🔮 Forecast",
    "📈 Components",
    "📋 Forecast Table"
])

# Tab 1 — Data Overview
with tab1:
    st.markdown(
        f"### 📊 {dataset_name} — Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Points",  len(df))
    col2.metric("Start Date",
                df['ds'].min().strftime(
                    '%d %b %Y'))
    col3.metric("End Date",
                df['ds'].max().strftime(
                    '%d %b %Y'))
    col4.metric("Average Value",
                f"{df['y'].mean():.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Historical Data")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['ds'], y=df['y'],
            mode='lines',
            name=dataset_name,
            line=dict(color='#3498db',
                      width=1.5)
        ))
        fig.update_layout(
            title=f'{dataset_name} Over Time',
            xaxis_title='Date',
            yaxis_title='Value',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig,
                        use_container_width=True)

    with col2:
        st.markdown("#### 📊 Distribution")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=df['y'],
            nbinsx=30,
            marker_color='#2ecc71',
            opacity=0.8
        ))
        fig2.update_layout(
            title='Value Distribution',
            xaxis_title='Value',
            yaxis_title='Count',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig2,
                        use_container_width=True)

    # Stats
    st.markdown("#### 📋 Statistical Summary")
    stats = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std Dev',
                   'Min', 'Max', 'Range'],
        'Value':  [
            f"{df['y'].mean():.2f}",
            f"{df['y'].median():.2f}",
            f"{df['y'].std():.2f}",
            f"{df['y'].min():.2f}",
            f"{df['y'].max():.2f}",
            f"{df['y'].max()-df['y'].min():.2f}"
        ]
    })
    st.dataframe(stats,
                 use_container_width=True,
                 hide_index=True)

# Tab 2 — Forecast
with tab2:
    st.markdown("### 🔮 Prophet Forecast")

    if st.button("🚀 Generate Forecast",
                 type="primary"):
        with st.spinner(
            "Training Prophet model and "
            "generating forecast..."
        ):
            try:
                from prophet import Prophet

                m = Prophet(
                    yearly_seasonality=
                        yearly_seasonality,
                    weekly_seasonality=
                        weekly_seasonality,
                    daily_seasonality=
                        daily_seasonality,
                    interval_width=0.95
                )
                m.fit(df)

                future = m.make_future_dataframe(
                    periods=forecast_days)
                forecast = m.predict(future)

                # Store in session
                st.session_state['forecast'] = \
                    forecast
                st.session_state['model']    = m
                st.session_state['df']       = df

                st.success(
                    f"✅ Forecast generated "
                    f"for {forecast_days} days!")

                # Forecast chart
                fig3 = go.Figure()

                # Historical
                fig3.add_trace(go.Scatter(
                    x=df['ds'], y=df['y'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='#3498db',
                              width=1.5)
                ))

                # Forecast
                future_fc = forecast[
                    forecast['ds'] > df['ds'].max()
                ]
                fig3.add_trace(go.Scatter(
                    x=future_fc['ds'],
                    y=future_fc['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#e74c3c',
                              width=2,
                              dash='dash')
                ))

                # Uncertainty
                if include_uncertainty:
                    fig3.add_trace(go.Scatter(
                        x=pd.concat([
                            future_fc['ds'],
                            future_fc['ds'][::-1]
                        ]),
                        y=pd.concat([
                            future_fc['yhat_upper'],
                            future_fc['yhat_lower'][::-1]
                        ]),
                        fill='toself',
                        fillcolor=
                            'rgba(231,76,60,0.15)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% Confidence'
                    ))

                fig3.update_layout(
                    title=f'{dataset_name} — '
                          f'{forecast_days}-Day '
                          f'Forecast',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    height=500,
                    template='plotly_white',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                st.plotly_chart(
                    fig3,
                    use_container_width=True)

                # Forecast metrics
                last_actual = df['y'].iloc[-1]
                last_forecast = future_fc[
                    'yhat'].iloc[-1]
                change = (
                    (last_forecast - last_actual)
                    / last_actual * 100)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Last Actual",
                    f"{last_actual:.2f}")
                c2.metric(
                    f"Forecast in "
                    f"{forecast_days} days",
                    f"{last_forecast:.2f}",
                    f"{change:+.1f}%")
                c3.metric(
                    "Forecast Max",
                    f"{future_fc['yhat'].max():.2f}")
                c4.metric(
                    "Forecast Min",
                    f"{future_fc['yhat'].min():.2f}")

            except Exception as e:
                st.error(
                    f"Forecast error: {str(e)}\n"
                    f"Make sure prophet is "
                    f"installed: "
                    f"pip install prophet")
    else:
        st.info(
            "Click 'Generate Forecast' "
            "to run Prophet!")

# Tab 3 — Components
with tab3:
    st.markdown(
        "### 📈 Forecast Components")

    if 'forecast' in st.session_state \
            and 'model' in st.session_state:
        forecast = st.session_state['forecast']
        m        = st.session_state['model']

        # Trend
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['trend'],
            mode='lines',
            line=dict(color='#3498db', width=2),
            name='Trend'
        ))
        fig_trend.update_layout(
            title='Trend Component',
            xaxis_title='Date',
            yaxis_title='Value',
            height=300,
            template='plotly_white'
        )
        st.plotly_chart(fig_trend,
                        use_container_width=True)

        # Yearly seasonality
        if yearly_seasonality and \
                'yearly' in forecast.columns:
            fig_year = go.Figure()
            year_data = forecast[
                forecast['ds'].dt.year ==
                forecast['ds'].dt.year.iloc[0]
            ].copy()
            fig_year.add_trace(go.Scatter(
                x=year_data['ds'],
                y=year_data['yearly'],
                mode='lines',
                line=dict(color='#2ecc71',
                          width=2),
                name='Yearly Seasonality'
            ))
            fig_year.update_layout(
                title='Yearly Seasonality',
                xaxis_title='Date',
                yaxis_title='Effect',
                height=300,
                template='plotly_white'
            )
            st.plotly_chart(
                fig_year,
                use_container_width=True)

        # Weekly seasonality
        if weekly_seasonality and \
                'weekly' in forecast.columns:
            week_data = forecast.head(7).copy()
            week_data['day'] = \
                week_data['ds'].dt.day_name()

            fig_week = go.Figure()
            fig_week.add_trace(go.Bar(
                x=week_data['day'],
                y=week_data['weekly'],
                marker_color='#f39c12',
                name='Weekly Effect'
            ))
            fig_week.update_layout(
                title='Weekly Seasonality Effect',
                xaxis_title='Day of Week',
                yaxis_title='Effect',
                height=300,
                template='plotly_white'
            )
            st.plotly_chart(
                fig_week,
                use_container_width=True)
    else:
        st.info(
            "Generate a forecast first "
            "to see components!")

# Tab 4 — Forecast Table
with tab4:
    st.markdown("### 📋 Forecast Data Table")

    if 'forecast' in st.session_state and \
            'df' in st.session_state:
        forecast = st.session_state['forecast']
        df_orig  = st.session_state['df']

        future_only = forecast[
            forecast['ds'] > df_orig['ds'].max()
        ][['ds', 'yhat',
           'yhat_lower', 'yhat_upper']]\
            .copy()

        future_only.columns = [
            'Date', 'Forecast',
            'Lower Bound', 'Upper Bound'
        ]
        future_only['Date'] = \
            future_only['Date'].dt.strftime(
                '%Y-%m-%d')
        for col in ['Forecast',
                    'Lower Bound',
                    'Upper Bound']:
            future_only[col] = \
                future_only[col].round(2)

        st.dataframe(
            future_only,
            use_container_width=True,
            hide_index=True)

        st.download_button(
            "⬇️ Download Forecast CSV",
            future_only.to_csv(index=False),
            "forecast.csv",
            "text/csv"
        )

        # Summary
        s1, s2, s3 = st.columns(3)
        s1.metric("Forecast Days",
                  len(future_only))
        s2.metric("Avg Forecast",
                  f"{future_only['Forecast'].mean():.2f}")
        s3.metric("Forecast Range",
                  f"{future_only['Forecast'].min():.0f}"
                  f" – "
                  f"{future_only['Forecast'].max():.0f}")
    else:
        st.info(
            "Generate a forecast first!")

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Time Series Forecasting | "
    "Powered by Facebook Prophet"
)