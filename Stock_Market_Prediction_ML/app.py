import os
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pandas_market_calendars as mcal

# Path to the model
model_path = 'Stock_Predictions_Open_Close_Model.keras'

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please ensure the file exists.")
else:
    # Load your trained model
    model = load_model(model_path)

    # Streamlit header
    st.header('Stock Open/Close Price Prediction')

    # User input for stock symbol
    stock = st.text_input('Enter Stock Symbol', 'GOOG')

    # Set start date and use the current date as the end date
    start = '2012-01-01'
    end = datetime.now()

    # Download stock data
    data = yf.download(stock, start=start, end=end.strftime('%Y-%m-%d'))

    # Display the downloaded stock data
    st.subheader('Stock Data')
    st.write(data)

    # Prepare the data for open and close prices
    data = data[['Open', 'Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Prepare the last sequence of data (100 days)
    last_100_days = data_scaled[-100:, :]  # Make sure to keep the shape (100, 2)
    last_100_days = last_100_days.reshape((1, 100, 2))  # Correctly reshape to (1, 100, 2)

    # Predict the next 5 trading days
    future_predictions = []
    for _ in range(5):
        prediction = model.predict(last_100_days)
        future_predictions.append(prediction[0])
        # Append the prediction to last_100_days and continue
        last_100_days = np.append(last_100_days[:, 1:, :], prediction.reshape(1, 1, 2), axis=1)

    future_predictions = np.array(future_predictions)
    future_predictions = scaler.inverse_transform(future_predictions)

    # Calculate the next 5 trading days
    nyse = mcal.get_calendar('NYSE')  # Use NYSE calendar for trading days
    future_dates = nyse.valid_days(start_date=data.index[-1], end_date=(end + timedelta(days=14)).strftime('%Y-%m-%d'))[:5]

    # Plot the combined historical and predicted prices
    st.subheader('Stock Prices: 5 Days Prior and 5 Days Future (Open/Close)')
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot historical data for open and close prices
    historical_dates = data.index[-5:]
    last_5_days_open = data['Open'][-5:].values
    last_5_days_close = data['Close'][-5:].values

    ax.plot(historical_dates, last_5_days_open, marker='o', linestyle='-', color='green', linewidth=2, markersize=8, label='Actual Open Prices (Last 5 Days)')
    ax.plot(historical_dates, last_5_days_close, marker='o', linestyle='-', color='red', linewidth=2, markersize=8, label='Actual Close Prices (Last 5 Days)')

    # Add data labels with $ sign to historical data points
    for i, txt in enumerate(last_5_days_open):
        ax.annotate(f'${txt:.2f}', (historical_dates[i], last_5_days_open[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12, color='black')
    for i, txt in enumerate(last_5_days_close):
        ax.annotate(f'${txt:.2f}', (historical_dates[i], last_5_days_close[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12, color='black')

    # Plot future predictions for open and close prices
    ax.plot(future_dates, future_predictions[:, 0], marker='o', linestyle='-', color='blue', linewidth=2, markersize=8, label='Predicted Open Prices (Next 5 Days)')
    ax.plot(future_dates, future_predictions[:, 1], marker='o', linestyle='-', color='orange', linewidth=2, markersize=8, label='Predicted Close Prices (Next 5 Days)')

    # Add data labels with $ sign to future prediction points
    for i, txt in enumerate(future_predictions[:, 0]):
        ax.annotate(f'${txt:.2f}', (future_dates[i], future_predictions[i, 0]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12, color='black')
    for i, txt in enumerate(future_predictions[:, 1]):
        ax.annotate(f'${txt:.2f}', (future_dates[i], future_predictions[i, 1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12, color='black')

    # Improve the date format on the x-axis
    date_form = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Add a grid
    ax.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='gray')

    # Enhance axis labels
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price (USD)', fontsize=14)

    # Improve legend appearance
    ax.legend(loc='upper left', fontsize=14, frameon=True, fancybox=True, shadow=True)

    # Add a title
    ax.set_title('Stock Prices: 5 Days Prior and 5 Days Future (Open/Close)', fontsize=20, pad=20)

    # Show the plot
    plt.tight_layout()
    st.pyplot(fig)
