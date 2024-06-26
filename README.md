# Stock Portfolio Optimization Using Deep Learning for FTSE 100 Listed Stocks
## Overview
This project implements a deep learning framework applied to stock portfolio management. Using the top 20 stocks of FTSE (Financial Times Stock Exchange) top 100 by market share, 10 portfolios are constructed. At first, historical stock data is downloaded, cleaned, transformed and then, using RFE key features are chosen. Then 2 models are used for predicting the stock returns, an existing LSTM architecture and a designed and developed time series transformer. Finally, using Sharpe ratio based portfolio optimisation, portfolio returns are calculated and the performance of each model is evaluated. The model was tested against an LSTM architecture proposed by Sen et al. for the same task and yielded much better performance. Repository contains all the files related to the project. 

## Table of Contents
- [Overview](#overview)
- [Data Loading & Preprocessing](#data-loading--preprocessing)
- [Feature Engineering](#feature-engineering)
- [Feature selection](#feature-selection)
- [Data transformation for supervised learning](#Data-transformation-for-supervised-learning)
- [Model architecture](#model-architecture)
- [Inference](#inference)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

## Data Loading & Preprocessing
For data acquisition the widely used libraries are pandas_datareader and yfinance. Pandas_datareader is an extension of pandas library to access and retrieve available financial data from Yahoo, Tiingo, IEX, FRED, etc. But, Yahoo has been used as the main data source in most recent research due to ease of use, free and accessible, integration with pandas and active maintenance compared to libraries such as pandas_datareader (Lynn, 2021).
Using yfinance, ‘Closing Price’ data for all the assets in the portfolio were downloaded for the period ‘2010-01-01’ to ‘2023-12-31’. 

### Sentiment analysis scores of financial news
Incorporating sentiment analysis scores of financial news relating to each stock as an input feature and checking its impact on the model performance could be interesting. There are many off the shelf models available today for sentiment analysis. To perform sentiment analysis on historical financial news, a model that has been trained on financial data/text is vital since, the accuracy of these readily available models could change based on the data they have been trained on. FinBERT is the most widely used model for sentiment analysis on financial texts. It is based on BERT language model, and it has been further developed and trained in the finance domain (Araci, 2019). It uses a transformer architecture with only an encoder similar to the time series transformer architecture that has been proposed for this project. The output of the model has 2 values, a label and a score. The label could be either negative, positive or neutral and the score is a softmax value. 
In order to extract financial news for each asset, an open-source API, NewsAPI was used. Then, daily top 25 news headlines of each asset were saved in json files, for the queried time period. Then, using FinBERT a sentiment score for each news headline is calculated. If the label is ‘neutral’ the softmax score is multiplied by 0, if it’s ‘negative’ then it is multiplied by -1 and it is ‘positive’ it is multiplied by 1. Then a daily mean score is calculated per each stock and this value is saved in the database along with the other features. Technical implementation was done using Hugging Face implementation of FinBERT (ProsusAI/finbert · Hugging Face, 2023)
However, ultimately sentiment score could not be incorporated into the input since, NewsAPI free subscription only provides 2 weeks of data. After making a request I was able to get access for Perigon API and query historical financial news data for a period of 2 years but, the data was inconsistent and sparse. The technical implementation was completed even though it could not be incorporated.

### Data cleaning
It is vital that data is cleaned properly for time series prediction tasks. When a database is created by downloading stock prices for a long period of time, there could be inconsistencies in the data. Therefore, data needs to be cleaned by dropping the missing values, removing outliers, standardizing the data, transforming data etc. Before dropping the missing values and removing outliers, it is important to know the impact it could have on the overall quality and quantity of the data. First the number of missing values and outliers were checked and in this case with respect to the number of entries in the database, the number of missing values and outliers were negligible. Therefore, the missing values and outliers could be easily removed from the database. 

## Feature Engineering
Even though ‘Closing Price’ is the most prominent feature in predicting the stock returns, other features such as RSI (Relative Strength Index), EMA (Exponential Moving Average), MoM (Momentum Indicator) and simple moving averages (5day rolling average, 10day rolling average, 20day rolling average) could contribute towards predicting better returns for the model. Therefore, the above attributes were calculated for each asset. The attributes are explained in detail below,

### Attributes
- RSI
It is a momentum indicator that is used in technical analysis which measures the magnitude and speed of an asset’s recent price to understand whether it is over or undervalued. The calculation has 2 steps as shown below,
$$\text{RSI}_{\text{step one}} = 100 - \left[ \frac{100}{1 + \left( \frac{\text{Average gain}}{\text{Average loss}} \right)} \right]$$


The standard number of periods used to calculate the initial RSI value is 14. Therefore, a period of 14 days will be considered for the RSI calculation. The second will be calculated as below (Relative Strength Index (RSI) Indicator Explained With Formula, 2021),
	$$\text{RSI}_{\text{step two}} = 100 - \left[ \frac{100}{1 + \left( \frac{(\text{Previous average gain} \times 13) + \text{Current gain}}{(\text{Previous average loss} \times 13) + \text{Current loss}} \right)} \right]$$


- EMA
  It is a weighted moving average which gives more priority to recent price data of the asset. It can be calculated using the below equation,
	$$\text{EMA} = \left( \frac{\text{Price}_{\text{current}} \times 2}{N+1} \right) + \text{EMA}_{\text{previous}} \times \left( 1 - \frac{2}{N+1} \right)$$

Here, the N indicates the number of days chosen for EMA and the weighting given to the most recent price is greater for a short period of EMA than for a longer period of EMA (What is EMA? How to Use Exponential Moving Average With Formula, 2023).

- MOM
This attribute is used to determine the momentum of an asset when it is gaining or falling in price in the market. It simply compares the current price of an asset with the previous price of it from a given number of periods ago. It can be calculated as below,
	$$\text{MOM} = \text{Closing Price}_{\text{current}} - \text{Closing Price}_{\text{n periods ago}}$$
	
- Simple Moving Averages (SMA)
It is a moving average calculated by adding the recent prices of an asset and dividing it by the number of time periods to get an average. For an example, to calculate the 5-day simple moving average of a stock, the last 5 closing prices of that particular stock will be added together and divided by 5. As input features, 5-day, 10-day and 20-day moving averages will be chosen and they will be calculated as below,
	$$\text{5-day SMA} = \frac{1}{5} \sum_{t=1}^{5} \text{Closing Price}_t$$
	
	$$\text{10-day SMA} = \frac{1}{10} \sum_{t=1}^{10} \text{Closing Price}_t$$

	$$\text{20-day SMA} = \frac{1}{20} \sum_{t=1}^{20} \text{Closing Price}_t$$
The above attributes were calculated for each asset and saved into a pandas data frame along with the closing prices and stock ticker value. The stock ticker value represents an integer value between 1-10. This is used for converting stock names into numerical values and identifying different assets for training purposes. For example, if a portfolio consists of [AAPL, MSFT, TSLA, …], their stock ticker values would be [1, 2, 3, …]

## Feature selection
Initially, 8 features were chosen for the model. The nature of the application is time series prediction and there are numerous feature selection techniques that could be used for selecting the vital features. Namely, RFE (Recursive Feature Elimination), correlation with the target variable or other predictors, Principal Component Analysis (PCA) etc. however, RFE is the most common feature selection method that is used for time series prediction tasks. For this task RFE is used with RF (Random Forest) to select the 5 most important features by setting the threshold to 0.3. This value was previously used by Wang et al. for a time series feature selection in their research (Wang et al., 2020). sklearn package was used for implementing feature selection. Results of RFE with RF proved that Closing price, stock ticker value, RSI, 5-day SMA, 10-day SMA and 20-day SMA are the key features for the model.

## Data transformation for supervised learning
In order to prepare the dataset for a time series prediction task, it has to be transformed. At first, based on the results of the feature selection, the less important features will be dropped from the database (pandas data frame) and then the whole database is converted to a NumPy array. This resulted in an array which has number of rows equal to the number of trading days and columns equal to the number of features considered. Then the array was reshaped in such a way that each row consists of data pertaining to 5 trading days. This arrangement was made because the model predicts the returns of assets for the following week, given the current weeks’ data. 
Then the data will be split into training and test sets. A split of 0.6 is used for training, 0.2 for validation and 0.2 is used for testing.

## Model architecture
There are several models that have been used in past research for stock return predictions. Recurrent Neural Networks is a popular method which has been used in multiple occasions and the LSTM (Long Short Term Memory) is the most widely used architecture (Fischer and Krauss, 2018; Wang et al., 2020). 
However, due to the introduction of attention mechanism in 2017, Transformers became wildly popular. Multiheaded attention mechanism became the base of any transformers model. Even though, initially it was introduced for Natural Language Processing tasks, now it has been adapted to time series prediction tasks as well (Vaswani et al., 2017). 
For this project, I will be using the LSTM architecture proposed by Sen et al. due to its frequent usage in literature and a time series transformer to compare the return predictions.

### Design & framework
All models have been designed using TensorFlow. Pytorch and TensorFlow are widely used and supported deep learning packages for python. Contrary to Pytorch, TensorFlow provides better visualization, therefore debugging is more straightforward, and the implementation is easier. The program has been written using Object Oriented Programming (OOP) concepts. Separate classes have been created for time embeddings, ETL, predictions, evaluations, etc. 

### LSTM architecture
LSTMs are made of an input layer, one/multiple hidden layers, and an output layer. The number of neurons in the input layer is equal to the number of input features.  
![image](https://github.com/hperer02/Portfolio-optimization-using-deep-learning/assets/124153856/c4cee307-488b-4fc9-9bc2-cd495ca8cc34)

The special characteristic of LSTM is the capability of maintaining memory cells. To enable this, 3 gates are used in a LSTM memory cell, 
	- Forget gate: Defines the information which needs to be removed from the memory
 	- Input gate: Defines the information which needs to be added to the memory.
  	- Output gate: Defines which information to use as the output

Sen et al. designed an LSTM architecture to predict the future stock prices of top five assets from nine different sectors of the Indian Stock Market. The model uses only the ‘Daily close prices’ of the past 50 days of each stock. Therefore, the input layer takes the shape (50,1). Then, the model receives this data and forwards it to the first LSTM layer which contains 256 nodes. The LSTM layer has a shape of (50,256) at the output. The first LSTM layer is followed by a dropout layer which switches off 30 percent of the nodes to avoid overfitting and then another LSTM layer (50,256) receives the output coming from the dropout layer and which is followed by another dropout layer of the same dropout rate. Then finally, a dense layer with 256 nodes receives the output of the second LSTM layer and outputs a single node which yields the predicted value of the close price. The architecture diagram of the network is shown in figure 3.
![image](https://github.com/hperer02/Portfolio-optimization-using-deep-learning/assets/124153856/da9f20a8-7b48-4c5f-848b-8c9e1d44e6fb)

However, the layout of the of the above architecture has to be adjusted to fulfill the purpose of the task. Therefore, instead of taking the previous close prices of the past 50 days, last weeks’ data is considered and instead of predicting the close price of the next day, the closing prices for the next are predicted using the model. The model was implemented using TensorFlow (refer Appendix 2)

![image](https://github.com/hperer02/Portfolio-optimization-using-deep-learning/assets/124153856/98cc48c3-2d84-444e-9018-a5235e66a86d)

### Time series transformer model
Attention concept was first introduced by Vaswani et al. to improve the performance of neural machine translation applications. Transformers use attention to enhance the training speed of the application. The transformers have 2 main components, the encoder part, and the decoder part. However, transformer models do not necessarily need to have both.  Figure 5 shows the Transformer model architecture proposed by Vaswani et al. (Vaswani et al., 2023)
