# Portfolio-optimization-using-deep-learning
Thesis of my masters in Data Science. 

Designed and developed a time series transformer model to predict stock returns. In order to enhance the performance of the model, time2vec embeddings and a learning rate scheduler were used in addition to the traditional techniques in data processing, validation, hyper parameter tuning etc. FinBERT was used to perform sentimental analysis on financial news related to each stock and NEWS API was used to download the historical news data. Sentiment score was taken as an input feature for the model along with other features such as Simple Moving Averages, Relative Strength Index (RSI), Momentum Index (MoM), Exponentail Moving Average (EMA) etc. 

The model was tested against an LSTM architecture proposed by Sen et al. for the same task and yielded much better performance. Repository contains all the files related to the project. 
