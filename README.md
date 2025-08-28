# 📈 Stock Price Prediction Dashboard

A sophisticated machine learning web application for predicting next-day stock closing prices using historical market data. The application leverages advanced regression models to analyze stock patterns and provide accurate price forecasts.

## 🎯 Overview

This application uses machine learning algorithms to predict the next day's closing price of any publicly traded stock based on the previous day's trading data. It provides an intuitive web interface for users to analyze stock performance and make informed investment decisions.

## ✨ Key Features

### 📊 Data Analysis
- **Real-time Stock Data**: Fetches live market data from Yahoo Finance
- **Historical Analysis**: Supports custom date ranges for comprehensive analysis
- **Interactive Visualizations**: Dynamic charts and graphs for data exploration

### 🤖 Machine Learning Models
- **Linear Regression**: Fast and interpretable baseline model
- **Random Forest Regressor**: Advanced ensemble method for improved accuracy
- **Model Performance Metrics**: MSE and R² scores for evaluation

### 📈 Visualization Suite
- **Price Trends**: Historical closing price line charts
- **Volume Analysis**: Trading volume bar charts
- **Feature Distributions**: Statistical distribution plots with box plots
- **Model Performance**: Actual vs. predicted scatter plots with trend lines

### 🔧 Interactive Features
- **Stock Symbol Search**: Support for any valid ticker symbol
- **Date Range Selection**: Flexible time period configuration
- **Model Comparison**: Switch between different prediction algorithms
- **Data Export**: Download predictions as CSV files

## 🏗️ Architecture

### Core Components

#### Data Pipeline
- **Data Source**: Yahoo Finance API via `yfinance` library
- **Data Processing**: Pandas for data manipulation and cleaning
- **Feature Engineering**: Creates next-day target variables

#### Machine Learning Pipeline
- **Feature Selection**: Volume, Open, High, Low prices
- **Target Variable**: Next day's closing price
- **Model Training**: 80/20 train-test split
- **Validation**: Cross-validation with performance metrics

#### User Interface
- **Framework**: Streamlit for rapid web app development
- **Layout**: Responsive sidebar and multi-tab interface
- **Charts**: Plotly for interactive visualizations

### Data Flow

```
Stock Ticker Input → Yahoo Finance API → Data Processing → Feature Engineering → Model Training → Prediction → Visualization
```

## 📋 Features Breakdown

### Input Parameters
| Parameter | Description | Example |
|-----------|-------------|---------|
| Stock Ticker | Company symbol | AAPL, MSFT, GOOGL |
| Start Date | Beginning of analysis period | 2010-01-01 |
| End Date | End of analysis period | Current date |
| Model Type | Prediction algorithm | Linear Regression or Random Forest |

### Output Metrics
| Metric | Description |
|--------|-------------|
| Latest Close Price | Most recent closing price |
| Latest Volume | Most recent trading volume |
| Test MSE | Mean Squared Error on test set |
| Test R² Score | Coefficient of determination |

### Dashboard Tabs

#### 1. Historical Data
- **Closing Price Chart**: Time series visualization of stock prices
- **Trading Volume Chart**: Bar chart showing daily trading volumes

#### 2. Feature Distributions
- **Statistical Overview**: Histograms with box plots for all features
- **Data Quality Check**: Visual inspection of data distributions

#### 3. Model Performance
- **Prediction Accuracy**: Scatter plot of actual vs. predicted values
- **Trend Analysis**: Linear regression line showing model fit
- **Performance Metrics**: Detailed MSE and R² score display

#### 4. Prediction & Download
- **Next Day Forecast**: Predicted closing price for the next trading day
- **Feature Summary**: Input values used for prediction
- **Data Export**: CSV download of test set predictions

## 🔍 Technical Implementation

### Data Processing Pipeline

1. **Data Retrieval**
   - Connects to Yahoo Finance API
   - Downloads OHLCV (Open, High, Low, Close, Volume) data
   - Handles multi-index column flattening

2. **Feature Engineering**
   - Creates target variable by shifting close prices
   - Removes NaN values from dataset
   - Validates data quality and completeness

3. **Model Training**
   - Splits data into training and testing sets
   - Trains selected machine learning model
   - Evaluates performance using standard metrics

### Error Handling

- **Invalid Ticker**: Validates stock symbol existence
- **Date Range Validation**: Ensures logical date ordering
- **Data Availability**: Checks for sufficient data points
- **Model Training**: Validates minimum data requirements

### Performance Optimization

- **Efficient Data Loading**: Streamlined API calls
- **Caching**: Streamlit's built-in caching for improved performance
- **Responsive UI**: Fast rendering with optimized chart libraries

## 📁 Project Structure

```
market-closing-price-predictor/
├── app.py                 # Main Streamlit application
└── README.md             # Project documentation
```

## 🎮 Usage Guide

### Quick Start

1. **Launch Application**: Run the Streamlit app
2. **Enter Stock Symbol**: Input desired ticker (e.g., "AAPL")
3. **Set Date Range**: Choose analysis period
4. **Select Model**: Pick prediction algorithm
5. **Run Analysis**: Click "Run Prediction" button

### Best Practices

#### Data Selection
- **Date Range**: Use at least 1 year of data for reliable predictions
- **Stock Selection**: Choose actively traded stocks for better data quality
- **Market Hours**: Consider market closures and holidays

#### Model Selection
- **Linear Regression**: Best for quick analysis and interpretability
- **Random Forest**: Use for higher accuracy and complex patterns
- **Performance**: Compare R² scores to select optimal model

### Interpretation Guidelines

#### Performance Metrics
- **R² Score > 0.8**: Excellent model performance
- **R² Score 0.6-0.8**: Good predictive capability
- **R² Score < 0.6**: Consider different features or models

#### Prediction Reliability
- **Stable Markets**: Higher prediction accuracy
- **Volatile Periods**: Lower reliability, use with caution
- **News Events**: External factors may affect predictions

## 🚀 Advanced Features

### Model Customization
The application supports easy extension with additional models:
- Support Vector Regression (SVR)
- Neural Networks
- LSTM for time series prediction

### Feature Enhancement
Potential additional features:
- Technical indicators (RSI, MACD, Moving Averages)
- Market sentiment data
- Economic indicators
- Options data

### Export Capabilities
- **CSV Format**: Structured data export
- **Prediction History**: Track model performance over time
- **Model Comparison**: Side-by-side algorithm evaluation

## ⚠️ Important Considerations

### Financial Disclaimer
- **Not Financial Advice**: This tool is for educational and analytical purposes only
- **Market Risk**: Past performance does not guarantee future results
- **Due Diligence**: Always conduct thorough research before making investment decisions

### Technical Limitations
- **Data Dependency**: Predictions rely on historical patterns
- **Market Conditions**: Sudden market changes may reduce accuracy
- **Model Assumptions**: Linear and tree-based models have inherent limitations

### Data Quality
- **Source Reliability**: Yahoo Finance data quality may vary
- **Missing Data**: Weekends and holidays result in gaps
- **Corporate Actions**: Stock splits and dividends may affect historical data

## 🔮 Future Enhancements

### Planned Features
- **Multi-stock Analysis**: Portfolio-level predictions
- **Real-time Updates**: Live data streaming
- **Advanced Models**: Deep learning implementations
- **Alert System**: Price target notifications

### Technical Improvements
- **Database Integration**: Historical data storage
- **API Development**: RESTful prediction endpoints
- **Mobile Optimization**: Responsive design improvements
- **Performance Monitoring**: Model drift detection

## 📊 Sample Output

The application generates comprehensive analysis including:
- Interactive price charts with zoom and pan functionality
- Statistical summaries with key performance indicators
- Model evaluation metrics with confidence intervals
- Downloadable prediction datasets for further analysis

## 🤝 Contributing

This project welcomes contributions for:
- Additional machine learning models
- Enhanced visualization features
- Performance optimizations
- Documentation improvements

## 📝 License

This project is open source and available for educational and research purposes.

---

**Built with**: Python, Streamlit, Scikit-learn, Plotly, Pandas, NumPy, yfinance