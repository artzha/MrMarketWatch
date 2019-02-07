# Mr.MarketWatch

![network structure](https://github.com/KingArthurZ3/StockDisplaySite/blob/master/assets/MrMarketWatch.png "Stock Display Site")

Mr. Stock Market is an intelligent trading advisor that analyzes numeric financial data and market sentiment using a gradient boosting machine learning model and predicts weekly stock prices. The frontend and backend parts of this project continuously update and display current financial predictions and graph data. The machine learning models used are built using Tensorflow and Sci-kit learn and the graphical display gui is built on top of the Vue.js framework. 

![network structure](https://github.com/KingArthurZ3/StockDisplaySite/blob/master/assets/10OBV.png "Stock Display Site")
![network structure](https://github.com/KingArthurZ3/StockDisplaySite/blob/master/assets/SMA.png "Stock Display Site")
![network structure](https://github.com/KingArthurZ3/StockDisplaySite/blob/master/assets/VIX.png "Stock Display Site")

Sample numerical data used as features in machine learning models

![network structure](https://github.com/KingArthurZ3/StockDisplaySite/blob/master/assets/GB_CLASSIFIER_1.png "Stock Display Site")
![network structure](https://github.com/KingArthurZ3/StockDisplaySite/blob/master/assets/GBM_Boost_1.png "Stock Display Site")

I used a gradient boosting classifier model (GBM) to predict future stock prices and cross-validated my predictions against a separate dataset to ensure rigidness against overfitting. Despite the high accuracy of 89.86%, the overall cross validation score was low, indicating that my model overfitted on the training dataset. To address this, I isolated the most influential financial features and only analyzed those instead. This ended up reducing the overall noise in the dataset.

![network structure](https://github.com/KingArthurZ3/StockDisplaySite/blob/master/assets/Hyperparameter_tuning.png "Stock Display Site")

Automated hyperparameter tuning, which iterates through a range of possible hyperparameters and identifies the optimal combination that produces the highest model accuracy

![network structure](https://github.com/KingArthurZ3/StockDisplaySite/blob/master/assets/GB_CLASSIFIER_2.png "Stock Display Site")
![network structure](https://github.com/KingArthurZ3/StockDisplaySite/blob/master/assets/AAPL_RETURNS.png "Stock Display Site")

After tuning the GBM model and cleaning the dataset for the most important numerical features, the overall returns on predictions dramatically increased, with some returns as high as 40% over seven months! Although the overall model accuracy decreased, the cross validation accuracy rose to nearly 70%, which increases the probability of accurate predictions on future stock prices.
