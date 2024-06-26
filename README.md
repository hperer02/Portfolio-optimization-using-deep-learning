# Stock Portfolio Optimization Using Deep Learning for FTSE 100 Listed Stocks
## Overview
This project implements a deep learning framework applied to stock portfolio management. Using the top 20 stocks of FTSE (Financial Times Stock Exchange) top 100 by market share, 10 portfolios are constructed. At first, historical stock data is downloaded, cleaned, transformed and then, using RFE key features are chosen. Then 2 models are used for predicting the stock returns, an existing LSTM architecture and a designed and developed time series transformer. Finally, using Sharpe ratio based portfolio optimisation, portfolio returns are calculated and the performance of each model is evaluated. The model was tested against an LSTM architecture proposed by Sen et al. for the same task and yielded much better performance. Repository contains all the files related to the project. 

## Table of Contents
- [Overview](#overview)
- [Data Loading & Preprocessing](#data-loading--preprocessing)
- [Data Augmentation](#data-augmentation)
- [Feature Engineering](#feature-engineering)
- [Model Building & Training](#model-building--training)
- [Inference](#inference)
- [Results](#results)
- [Conclusion](#conclusion)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Acknowledgments](#acknowledgments)

## Data Loading & Preprocessing
For data acquisition the widely used libraries are pandas_datareader and yfinance. Pandas_datareader is an extension of pandas library to access and retrieve available financial data from Yahoo, Tiingo, IEX, FRED, etc. But, Yahoo has been used as the main data source in most recent research due to ease of use, free and accessible, integration with pandas and active maintenance compared to libraries such as pandas_datareader (Lynn, 2021).
Using yfinance, ‘Closing Price’ data for all the assets in the portfolio were downloaded for the period ‘2010-01-01’ to ‘2023-12-31’. Even though ‘Closing Price’ is the most prominent feature in predicting the stock returns, other features such as RSI (Relative Strength Index), EMA (Exponential Moving Average), MoM (Momentum Indicator) and simple moving averages (5day rolling average, 10day rolling average, 20day rolling average) could contribute towards predicting better returns for the model. Therefore, the above attributes were calculated for each asset. The attributes are explained in detail below,
### Attributes
- RSI
It is a momentum indicator that is used in technical analysis which measures the magnitude and speed of an asset’s recent price to understand whether it is over or undervalued. The calculation has 2 steps as shown below,
	〖RSI〗_(step one)=100-[100/(1+(Average gain)/(Average loss))]	

The standard number of periods used to calculate the initial RSI value is 14. Therefore, a period of 14 days will be considered for the RSI calculation. The second will be calculated as below (Relative Strength Index (RSI) Indicator Explained With Formula, 2021),
	〖RSI〗_(step two)=100-[100/(1+((Previous average gain×13)+Current gain)/((Previous average loss×13)+Current loss))]

- EMA
  It is a weighted moving average which gives more priority to recent price data of the asset. It can be calculated using the below equation,
	EMA=(〖Price〗_current×2/(N+1))+〖EMA〗_previous×(1-2/(N+1))	5.3


Here, the N indicates the number of days chosen for EMA and the weighting given to the most recent price is greater for a short period of EMA than for a longer period of EMA (What is EMA? How to Use Exponential Moving Average With Formula, 2023).


