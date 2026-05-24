# Time Series Forecasting App 📈

Forecasts any time series data using
Facebook Prophet with interactive Plotly charts.

## Live Demo
[Click here](https://time-series-forecasting-3uefskisxqgttukusmvnxs.streamlit.app)

## Features
- 4 sample datasets: Sales, Stock, Temperature, Traffic
- Upload your own CSV for custom forecasting
- Adjustable forecast horizon (7-365 days)
- 95% confidence interval visualization
- Trend, yearly and weekly component breakdown
- Download forecast as CSV

## How It Works
Facebook Prophet decomposes time series into:
- Trend component
- Yearly seasonality
- Weekly seasonality
- Holiday effects

## Tools Used
- Python, Prophet, Plotly, Streamlit, Pandas

## How to Run Locally
pip install streamlit prophet plotly pandas numpy matplotlib
streamlit run app.py
