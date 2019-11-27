
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


def create_model(df,y ,X ,X_train, X_test, y_train, y_test, degree, random_state, test_size, alpha):
    
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    
    
    ss = StandardScaler()
    ss.fit(X_train)

    X_train_scaled = ss.transform(X_train)
    X_test_scaled = ss.transform(X_test)
    
    linreg_norm = LinearRegression()
    linreg_norm.fit(X_train_scaled, y_train)

  
    
    
   
    
    X_cat = df[['Month', 'Origin', 'Dest']]
    X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y,test_size=test_size, random_state=random_state)
    # OneHotEncode Categorical variables
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(X_train_cat)

    X_train_ohe = ohe.transform(X_train_cat)
    X_test_ohe = ohe.transform(X_test_cat)

    columns = ohe.get_feature_names(input_features=X_train_cat.columns)
    cat_train_df = pd.DataFrame(X_train_ohe.todense(), columns=columns)
    cat_test_df = pd.DataFrame(X_test_ohe.todense(), columns=columns)
    X_train_all = pd.concat([pd.DataFrame(X_train_scaled), cat_train_df], axis = 1)
    X_test_all = pd.concat([pd.DataFrame(X_test_scaled), cat_test_df], axis = 1)
    linreg_all = LinearRegression()
    linreg_all.fit(X_train_all, y_train)

    print('Baseline model Continuous and Categorical')
    print('Training r^2:', linreg_all.score(X_train_all, y_train))
    print('Testing r^2:', linreg_all.score(X_test_all, y_test))
    print('Training MSE:', mean_squared_error(y_train, linreg_all.predict(X_train_all)))
    print('Testing MSE:', mean_squared_error(y_test, linreg_all.predict(X_test_all)))
    
   
    
    print("\n")
    
    lasso = Lasso(alpha=alpha) #Lasso is also known as the L1 norm.
    lasso.fit(X_train_all, y_train)
    print( 'Lasso')
    print('Training r^2:', lasso.score(X_train_all, y_train))
    print('Testing r^2:', lasso.score(X_test_all, y_test))
    print('Training MSE:', mean_squared_error(y_train, lasso.predict(X_train_all)))
    print('Testing MSE:', mean_squared_error(y_test, lasso.predict(X_test_all)))
    
    print("\n")
    
    ridge = Ridge(alpha = alpha) #Ridge is also known as the L2 norm.
    ridge.fit(X_train_all, y_train)
    print('Ridge')
    print('Training r^2:', ridge.score(X_train_all, y_train))
    print('Testing r^2:', ridge.score(X_test_all, y_test))
    print('Training MSE:', mean_squared_error(y_train, ridge.predict(X_train_all)))
    print('Testing MSE:', mean_squared_error(y_test, ridge.predict(X_test_all)))
    
    print("\n")
    
    poly_features = PolynomialFeatures(degree)
  
      # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)
  
  # fit the transformed features to Linear Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
  
  # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)
  
  # predicting on test data-set
    y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))
  
  # evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
    r2_train = r2_score(y_train, y_train_predicted)
  
  # evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
    r2_test = r2_score(y_test, y_test_predict)
  
    
    print("\n")
    
    print(" Polynomial training set")
 
    print("MSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}".format(r2_train))
  
    print("\n")
  
    print("Polynomial test set")

    print("MSE of test set is {}".format(rmse_test))
    print("R2 score of test set is {}".format(r2_test))
    
    print("\n")
    
    print('Cross Validation for Polynomial model')
 
    lm = LinearRegression()

    # store scores in scores object
    # we can't use accuracy as our evaluation metric since that's only relevant for classification problems
    # RMSE is not directly available so we will use MSE
    scores = cross_val_score(lm, X_train_poly, y_train, cv=10, scoring='r2')
    mse_scores = cross_val_score(lm, X_train_poly, y_train, cv=10, scoring='neg_mean_squared_error')
    print('Cross Validation Mean r2:',np.mean(scores))
    print('Cross Validation Mean MSE:',np.mean(mse_scores))
    print('Cross Validation 10 Fold Score:',scores)
    print ('Cross Validation 10 Fold mean squared error',-(mse_scores) )
    
    
from sklearn.feature_selection import RFE
def feature_ranking(X,y):
    linreg = LinearRegression()
    selector = RFE(linreg, n_features_to_select = 5)
    selector = selector.fit(X, y.values.ravel()) # convert y to 1d np array to prevent DataConversionWarning
    selector.support_
    
    selected_columns = X.columns[selector.support_ ]
    linreg.fit(X[selected_columns],y)
    yhat = linreg.predict(X[selected_columns])
    SS_Residual = np.sum((y-yhat)**2)
    SS_Total = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X[selected_columns].shape[1]-1)
    print("\n")
    
    print("r_squared is {}".format(r_squared))
    
    print("\n")
    
    print("adjusted_r_sqaured is {}".format(adjusted_r_squared))

    
def stepwise_selection(X, y, 
                       initial_list= [], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
    result = stepwise_selection(X, y, verbose = True)
    print('resulting features:')
    print(result)
