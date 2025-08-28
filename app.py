import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import io
import datetime

st.set_page_config(page_title="Stock Price Prediction Dashboard", layout="wide")

st.title("📈 Stock Price Prediction Dashboard")
st.markdown("""
Predict next day’s closing price based on previous day's Volume, Open, High, and Low.  
Use Linear Regression or Random Forest model.
""")

with st.sidebar:
    st.header("Enter your inputs")
    ticker = st.text_input("Stock ticker", value="AAPL", help="Enter stock symbol, e.g. AAPL")
    today = datetime.date.today()
    start_date = st.date_input("Start date", pd.to_datetime("2010-01-01"), max_value=today)
    end_date = st.date_input("End date", today, min_value=start_date, max_value=today)
    model_option = st.selectbox("Choose model", ["Linear Regression", "Random Forest Regressor"])
    run = st.button("Run Prediction")

if run:
    if not ticker:
        st.error("⚠️ Please enter a stock ticker symbol.")
    elif start_date >= end_date:
        st.error("⚠️ Start date must be before end date.")
    else:
        data = yf.download(ticker.upper(), start=start_date, end=end_date)

        # Flatten multiindex columns from yfinance if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        if data.empty:
            st.error(f"⚠️ No data found for ticker '{ticker.upper()}' in the date range {start_date} to {end_date}. Please check your inputs.")
        else:
            st.write(f"**Data rows after download:** {len(data)}")
            st.write("Last 5 rows of downloaded data:")
            st.dataframe(data.tail())

            # Create target variable: next day's close price
            data['Next_Close'] = data['Close'].shift(-1)
            data.dropna(inplace=True)

            st.write(f"**Data rows after dropping NaNs (shifting):** {len(data)}")
            st.write("Last 5 rows after processing:")
            st.dataframe(data.tail())

            feature_cols = ['Volume', 'Open', 'High', 'Low']
            X = data[feature_cols]
            y = data['Next_Close']

            if len(X) < 5:
                st.error("⚠️ Not enough data after processing to train the model. Try a larger date range or a different ticker.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if model_option == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.header(f"📊 {ticker.upper()} Overview")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Latest Close Price", f"${float(data['Close'].iloc[-1]):.2f}")
                col2.metric("Latest Volume", f"{int(data['Volume'].iloc[-1]):,}")
                col3.metric("Test MSE", f"{mse:.2f}")
                col4.metric("Test R² Score", f"{r2:.2f}")

                tab1, tab2, tab3, tab4 = st.tabs(["Historical Data", "Feature Distributions", "Model Performance", "Prediction & Download"])

                with tab1:
                    fig_price = px.line(data, y='Close', title=f"{ticker.upper()} Closing Price")
                    st.plotly_chart(fig_price, use_container_width=True)

                    fig_volume = px.bar(data, y='Volume', title=f"{ticker.upper()} Trading Volume")
                    st.plotly_chart(fig_volume, use_container_width=True)

                with tab2:
                    fig_features = px.histogram(data[feature_cols], marginal="box", barmode="overlay", title="Feature Distributions")
                    st.plotly_chart(fig_features, use_container_width=True)

                with tab3:
                    fig_perf = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual Close', 'y':'Predicted Close'}, 
                                          title="Actual vs Predicted Closing Prices",
                                          trendline="ols", trendline_color_override="red")
                    st.plotly_chart(fig_perf, use_container_width=True)

                    st.markdown(f"""
                    **Model:** {model_option}  
                    **Mean Squared Error (MSE):** {mse:.2f}  
                    **R² Score:** {r2:.2f}  
                    """)

                with tab4:
                    last_features = data[feature_cols].iloc[-1].values.reshape(1, -1)
                    predicted_next_close = model.predict(last_features)[0]
                    st.write(f"Based on last day's features:\n\n"
                             f"- Volume: {int(last_features[0][0]):,}\n"
                             f"- Open: ${last_features[0][1]:.2f}\n"
                             f"- High: ${last_features[0][2]:.2f}\n"
                             f"- Low: ${last_features[0][3]:.2f}\n")
                    st.markdown(f"### Predicted next day’s closing price: **${predicted_next_close:.2f}**")

                    pred_df = pd.DataFrame({
                        'Date': y_test.index,
                        'Actual_Close': y_test.values,
                        'Predicted_Close': y_pred
                    }).reset_index(drop=True)

                    csv_buffer = io.StringIO()
                    pred_df.to_csv(csv_buffer, index=False)
                    csv_str = csv_buffer.getvalue()

                    st.download_button(
                        label="Download Test Set Predictions as CSV",
                        data=csv_str,
                        file_name=f"{ticker.upper()}_predictions.csv",
                        mime="text/csv"
                    )
else:
    st.info("Enter inputs in the sidebar and click 'Run Prediction' to start.")
