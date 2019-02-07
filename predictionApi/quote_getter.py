import zipline as zp
import quandl as qd
import pandas as pd
from pandas import DataFrame
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pytz
import os.path
import string
import time
import datetime
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

# import functions needed for GBM model
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV  # Performing grid search


def init_data(companies):
    #     pd.DataFrame(),pd.DataFrame(), pd.DataFrame(),[],pd.DataFrame(),
    for comp in companies:
        # main_data[comp] = pd.read_csv(f'./rsc/{comp}_data.csv', sep = ',')
        # main_data[comp].dropna()

        # Input all the technical indicators and the stock times series data

        ti = TechIndicators(key='L2O2TYTG382ETN0N', output_format='pandas')

        # Fetch the simple moving average (SMA) values
        main_data[comp], meta_data = ti.get_sma(symbol=comp, interval='1min', time_period=100, series_type='close')

        # Fetch the exponential moving average (EMA) values
        ema_data, meta_data = ti.get_ema(symbol=comp, interval='1min', time_period=100, series_type='close')
        main_data[comp]['EMA'] = ema_data

        """
        # Fetch the weighted moving average (WMA) values
        wma_data, meta_data = ti.get_wma(symbol=comp,interval='1min', time_period=10, series_type='close')
        main_data[comp]['WMA'] = wma_data
        
        # Fetch the double exponential moving agerage (DEMA) values
        dema_data, meta_data = ti.get_dema(symbol=comp,interval='1min', time_period=10, series_type='close')
        main_data[comp]['DEMA'] = dema_data
        
        # Fetch the triple exponential moving average (TEMA) values
        tema_data, meta_data = ti.get_tema(symbol=comp,interval='1min', time_period=10, series_type='close')
        main_data[comp]['TEMA'] = tema_data
        
        # Fetch the triangular moving average (TRIMA) values 
        trima_data, meta_data = ti.get_trima(symbol=comp,interval='1min', time_period=10, series_type='close')
        main_data[comp]['TRIMA'] = trima_data

        # Fetch the Kaufman adaptive moving average (KAMA) values
        kama_data, meta_data = ti.get_kama(symbol=comp,interval='1min', time_period=10, series_type='close')
        main_data[comp]['KAMA'] = kama_data		

        # Fetch the MESA adaptive moving average (MAMA) values
        mama_data, meta_data = ti.get_mama(symbol=comp,interval='1min', time_period=10, series_type='close')
        main_data[comp]['MAMA'] = mama_data['MAMA']
        main_data[comp]['FAMA'] = mama_data['FAMA']

        # Fetch the triple exponential moving average (T3) values
        t3_data, meta_data = ti.get_t3(symbol=comp,interval='1min', time_period=10, series_type='close')
        main_data[comp]['T3'] = t3_data	
        """

        # Fetch the moving average convergence / divergence (MACD) values
        macd_data, meta_data = ti.get_macd(symbol=comp, interval='1min', series_type='close')
        main_data[comp]['MACD'] = macd_data['MACD']
        main_data[comp]['MACD_Hist'] = macd_data['MACD_Hist']
        main_data[comp]['MACD_Signal'] = macd_data['MACD_Signal']

        """		
        # Fetch the moving average convergence / divergence values with controllable moving average type
        macdext_data, meta_data = ti.get_macdext(symbol=comp,interval='1min', series_type='close')
        main_data[comp]['MACDEXT'] = macdext_data['MACD']
        main_data[comp]['MACDEXT_Hist'] = macdext_data['MACD_Hist']
        main_data[comp]['MACDEXT_Signal'] = macdext_data['MACD_Signal']
        """

        # Fetch the stochastic oscillator (STOCH) values
        stoch_data, meta_data = ti.get_stoch(symbol=comp, interval='1min')
        main_data[comp]['SlowK'] = stoch_data['SlowK']
        main_data[comp]['SlowD'] = stoch_data['SlowD']

        """
        # Fetch the stochastic fast (STOCHF) values
        stochf_data, meta_data = ti.get_stochf(symbol=comp,interval='1min')
        main_data[comp]['FastK'] = stochf_data['FastK']
        main_data[comp]['FastD'] = stochf_data['FastD']
        """

        # Fetch the relative strength index (RSI) values
        rsi_data, meta_data = ti.get_rsi(symbol=comp, interval='1min', time_period=10, series_type='close')
        main_data[comp]['RSI'] = rsi_data

        """
        # Fetch the stochastic relative strength index (STOCHRSI) values
        stochrsi_data, meta_data = ti.get_stochrsi(symbol=comp,interval='1min', time_period=10, series_type='close')
        main_data[comp]['STOCHRSI_FastK'] = stochrsi_data['FastK']
        main_data[comp]['STOCHRSI_FastD'] = stochrsi_data['FastD']

        # Fetch the Williams' %R (WILLR) values
        willr_data, meta_data = ti.get_willr(symbol=comp,interval='1min', time_period=10)
        main_data[comp]['WILLR'] = willr_data
        """

        # Fetch the average directional movement index (ADX) values
        adx_data, meta_data = ti.get_adx(symbol=comp, interval='1min', time_period=100)
        main_data[comp]['ADX'] = adx_data

        """
        # Fetch the average directional movement index rating (ADXR) values
        adxr_data, meta_data = ti.get_adxr(symbol=comp,interval='1min', time_period=10)
        main_data[comp]['ADXR'] = adxr_data

        # Fetch the absolute price oscillator (APO) values
        apo_data, meta_data = ti.get_apo(symbol=comp,interval='1min', series_type='close')
        main_data[comp]['APO'] = apo_data

        # Fetch the percentage price oscillator (PPO) values
        ppo_data, meta_data = ti.get_ppo(symbol=comp,interval='1min', series_type='close')
        main_data[comp]['PPO'] = ppo_data

        # Fetch the momentum (MOM) values
        mom_data, meta_data = ti.get_mom(symbol=comp,interval='1min', time_period=10, series_type='close')
        main_data[comp]['MOM'] = mom_data

        # Fetch the balance of power (BOP) values
        bop_data, meta_data = ti.get_bop(symbol=comp,interval='1min')
        main_data[comp]['BOP'] = bop_data
        """
        time.sleep(5)

        # Fetch the commodity channel index (CCI) values
        cci_data, meta_data = ti.get_cci(symbol=comp, interval='1min', time_period=100)
        main_data[comp]['CCI'] = cci_data

        """
        # Fetch the Chande momentum oscillator (CMO) values
        cmo_data, meta_data = ti.get_cmo(symbol=comp,interval='1min', time_period=10, series_type='close')
        main_data[comp]['CMO'] = cmo_data

        # Fetch the rate of change (ROC) values
        roc_data, meta_data = ti.get_roc(symbol=comp,interval='1min', time_period=10, series_type='close')
        main_data[comp]['ROC'] = roc_data


        # Fetch the rate of change ratio (ROCR) values
        rocr_data, meta_data = ti.get_rocr(symbol=comp,interval='1min', time_period=10, series_type='close')
        main_data[comp]['ROCR'] = rocr_data

        time.sleep(5)
        """
        # Fetch the Aroon (AROON) values
        aroon_data, meta_data = ti.get_aroon(symbol=comp, interval='1min', time_period=100)
        main_data[comp]['Aroon Down'] = aroon_data['Aroon Down']
        main_data[comp]['Aroon Up'] = aroon_data['Aroon Up']

        """
        # Fetch the Aroon oscillator (AROONOSC) values
        aroonosc_data, meta_data = ti.get_aroonosc(symbol=comp,interval='1min', time_period=10)
        main_data[comp]['AROONOSC'] = aroonosc_data

        # Fetch the money flow index (MFI) values
        mfi_data, meta_data = ti.get_mfi(symbol=comp,interval='1min', time_period=10)
        main_data[comp]['MFI'] = mfi_data

        # Fetch the 1-day rate of change of a triple smooth exponential moving average (TRIX) values
        triX_train_data['AAPL'], meta_data = ti.get_trix(symbol=comp,interval='1min', time_period=10, series_type='close')
        main_data[comp]['TRIX'] = triX_train_data['AAPL']

        # Fetch the ultimate oscillator (ULTOSC) values
        ultosc_data, meta_data = ti.get_ultsoc(symbol=comp,interval='1min', time_period=10)
        main_data[comp]['ULTOSC'] = ultosc_data

        # Fetch the directional movement index (DX) values
        dX_train_data['AAPL'], meta_data = ti.get_dx(symbol=comp,interval='1min', time_period=10)
        main_data[comp]['DX'] = dX_train_data['AAPL']
        """

        # Fetch the Chaikin A/D line (AD) value
        ad_data, meta_data = ti.get_trix(symbol=comp, interval='1min')
        main_data[comp]['AD'] = ad_data

        # Fetch the on balance volume (OBV) values
        obv_data, meta_data = ti.get_obv(symbol=comp, interval='1min')
        main_data[comp]['OBV'] = obv_data

        # print(main_data[comp].head())

        ts = TimeSeries(key='L2O2TYTG382ETN0N', output_format='pandas')
        intraday_data, meta_data = ts.get_intraday(symbol=comp, interval='1min', outputsize='full')

        # intraday_data = intraday_data.iloc[9:]
        # intraday_data = intraday_data.reset_index(drop=True)
        # intraday_data.index = main_data[comp].index
        # intraday_data.set_index('date')
        intraday_data.index = pd.Index([index[:-3] for index in intraday_data.index], name='date')
        # intraday_data.set_index('date')

        """
        for index in intraday_data.index:
            print(index)
        print(type(intraday_data.index))
        """

        main_data[comp] = pd.concat([main_data[comp], intraday_data], axis=1)
        print(main_data[comp].index)

        print(main_data[comp].head())

        main_data[comp] = main_data[comp].dropna()
        main_data[comp].index.name = 'date'

        y = np.where(main_data[comp]['4. close'].shift(-1) > main_data[comp]['4. close'], 1, -1)
        main_data[comp]['Open-Close'] = main_data[comp]['1. open'] - main_data[comp]['4. close']
        main_data[comp]['High-Low'] = main_data[comp]['2. high'] - main_data[comp]['3. low']
        X = main_data[comp][main_data[comp].columns[0:]]
        split = int(split_percentage * len(main_data[comp]['1. open']))
        X_split_data[comp] = main_data[comp][split:]
        X_train_data[comp] = X[:split]
        y_train_data[comp] = y[:split]
        X_test_data[comp] = X[split:]
        y_test_data[comp] = y[split:]

    return main_data


"""
# first two dataframes are the X_train and y_train
# second two dataframes are the X_test and y_test
def model_fit(alg, X_train_data['AAPL'], X_train, y_train, X_test, y_test, predictors, companies, performCV=True, printFeatureImportance=True, cv_folds=5):
    for comp in companies:
        # model_fit(gbm0, all_data[comp][0], predictors, all_data[comp][1], company_name = comp)
        # Fit algorithm on the data
        alg.fit(X_train[comp], y_train[comp])

        # compare accuracy by comparing actual values of trading signal with predicted values of trading signal
        accuracy_train = accuracy_score(y_train[comp], gbm0.predict(X_train[comp]))
        accuracy_test = accuracy_score(y_test[comp], gbm0.predict(X_test[comp]))
        print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))
        print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))
        
        # Predict testing set
        X_train_predictions = alg.predict(X_test[comp])
        X_train_predprob = alg.predict_proba(X_test[comp])[:,1]

        #graph_data(alg, X_train_data['AAPL'][comp], X_test[comp], y_test[comp], comp)
"""


# Graph the data
def graph_data(alg, X_train_data, X_test, y_test, comp):
    if X_train_data['AAPL'].index.name != "date":
        X_train_data['AAPL'].set_index('date', inplace=True)

    X_train_data.loc[:, 'Predicted_Signal'] = alg.predict(X_test)
    X_train_data.loc[:, 'Return'] = y_test
    X_train_data.loc[:, 'Strategy_Return'] = X_train_data.loc[:, 'Return'] * X_train_data.loc[:, 'Predicted_Signal']
    X_train_data.loc[:, 'Strategy_Return'].cumsum().plot(figsize=(10, 4))
    plt.ylabel(f"\n{comp} Strategy Returns (%)")
    plt.show()


def model_fit(alg, dtrain, predictors, outcome, performCV=True, printFeatureImportance=True, cv_folds=5):
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], outcome)

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, dtrain[predictors], outcome, cv=cv_folds, scoring='roc_auc')

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(outcome[0:len(outcome)], dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(outcome, dtrain_predprob))

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (
        np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')


def classification_model(model, data, predictors, outcome):
    # fit the model
    model.fit(data[predictors], outcome)

    # Make predictions on the training set
    predictions = model.predict(data[predictors])

    # Print accuracy
    accuracy = accuracy_score(predictions, outcome)
    print("Accuracy : %s" % "{0:.2%}".format(accuracy))

    # Perform k-fold cross-validation with 5 folds
    kf = KFold(n_splits=5)
    error = []
    for train, test in kf.split(outcome):
        # Filter training data
        train_predictors = (data[predictors].iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = outcome[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test, :], outcome[test]))

        print("Cross-Validation Score : %s" % "{0:.2%}".format(np.mean(error)))

    # Fit the model again so that it can be referred outside the function:
    model.fit(data[predictors], outcome)


if __name__ == '__main__':
    # companies = ['AAPL', 'MSFT', 'AMZN', 'INTC', 'TSLA']
    companies = ['AAPL']

    # first two dataframes are the X_train and y_train
    main_data = dict.fromkeys(companies, pd.DataFrame)
    meta_data = dict.fromkeys(companies, pd.DataFrame)
    X_split_data = dict.fromkeys(companies, pd.DataFrame)
    X_train_data = dict.fromkeys(companies, pd.DataFrame)
    y_train_data = dict.fromkeys(companies, [])
    X_test_data = dict.fromkeys(companies, pd.DataFrame)
    y_test_data = dict.fromkeys(companies, [])

    split_percentage = 0.8

    # Loop every minute to update model

    # while True:

    start = time.time()
    main_data = init_data(companies)

    # optimum parameters found from ML_Model_Training
    parameter_dict = {
        'max_depth': 5,
        'min_samples_split': 500,
        'min_samples_leaf': 10,
        'subsample': 0.8,
        'random_state': 10,
        'max_features': 6
    }

    predictors = main_data['AAPL'].columns

    print(predictors)
    print(X_test_data['AAPL'].columns)

    y = np.where(main_data['AAPL']['4. close'].shift(-1) > main_data['AAPL']['4. close'], 1, -1)
    predictor_var = main_data['AAPL'].columns[1:len(main_data['AAPL'])]
    outcome_var = y
    model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=2)
    classification_model(model, main_data['AAPL'], predictor_var, outcome_var)

    featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
    print(featimp)

    gbm0 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=50, max_depth=parameter_dict['max_depth'],
                                      min_samples_split=parameter_dict['min_samples_split'],
                                      min_samples_leaf=parameter_dict['min_samples_leaf'],
                                      subsample=parameter_dict['subsample'],
                                      random_state=parameter_dict['random_state'],
                                      max_features=parameter_dict['max_features'])

    # Choose predictors test model accuracy
    # hide pandas warnings
    import warnings

    warnings.filterwarnings('ignore')
    # model_fit(alg = gbm0, main_data['AAPL'] = X_split_data, X_train = main_data['AAPL'], y_train = y_train_data, X_test = X_test_data, y_test = y_test_data, predictors = predictors, companies = companies )

    model_fit(alg=gbm0, dtrain=main_data['AAPL'], predictors=predictors, outcome=outcome_var)

    param_test1 = {'n_estimators': list(range(20, 81, 10))}

    gsearch1 = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=parameter_dict['min_samples_split'],
                                             min_samples_leaf=parameter_dict['min_samples_leaf'],
                                             subsample=parameter_dict['subsample'],
                                             random_state=parameter_dict['random_state']), param_grid=param_test1,
        scoring='roc_auc', n_jobs=4, iid=False, cv=5)

    print(main_data['AAPL'].shape)
    print(np.shape(y))
    gsearch1.fit(main_data['AAPL'][main_data['AAPL'].columns[1:len(X_train_data['AAPL'])]], y)

    print(gsearch1.grid_scores_)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)

    param_test2 = {'max_depth': list(range(5, 16, 2)), 'min_samples_split': list(range(200, 1001, 200))}
    gsearch2 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,
                                                                 max_features=parameter_dict['max_features'],
                                                                 subsample=parameter_dict['subsample'],
                                                                 random_state=10), param_grid=param_test2,
                            scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch2.fit(main_data['AAPL'][main_data['AAPL'].columns[1:len(main_data['AAPL'])]], y)
    gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

    X_split_data['AAPL'].head()
    end = time.time()

    # Sleep until the next cycle begins at the next minute
    # time.sleep(60 - (end - start))

    """
    while True:
        ts = TimeSeries(key='L2O2TYTG382ETN0N', output_format='pandas')
        data, meta_data = ts.get_intraday(symbol='AAPL',interval='1min', outputsize='compact')
        print(meta_data)
        print(data['4. close'][-1])
        #plt.title('Intraday Times Series for the AAPT stock (1 min)')
        #plt.show()
        #plt.close('all')
    """
