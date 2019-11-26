
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
# def category_features(df,y_test,test_size,random_state):
#     X_cat = df_reg[['Month', 'Origin', 'Dest']]
#     X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y,test_size=test_size, random_state=random_state)
#     # OneHotEncode Categorical variables
#     ohe = OneHotEncoder(handle_unknown='ignore')
#     ohe.fit(X_train_cat)

#     X_train_ohe = ohe.transform(X_train_cat)
#     X_test_ohe = ohe.transform(X_test_cat)

#     columns = ohe.get_feature_names(input_features=X_train_cat.columns)
#     cat_train_df = pd.DataFrame(X_train_ohe.todense(), columns=columns)
#     cat_test_df = pd.DataFrame(X_test_ohe.todense(), columns=columns)
#     X_train_all = pd.concat([pd.DataFrame(X_train_scaled), cat_train_df], axis = 1)
#     X_test_all = pd.concat([pd.DataFrame(X_test_scaled), cat_test_df], axis = 1)
#     linreg_all = LinearRegression()
#     linreg_all.fit(X_train_all, y_train)

#     print('Continuous and Categorical')
#     print('Training r^2:', linreg_all.score(X_train_all, y_train))
#     print('Testing r^2:', linreg_all.score(X_test_all, y_test))
#     print('Training MSE:', mean_squared_error(y_train, linreg_all.predict(X_train_all)))
#     print('Testing MSE:', mean_squared_error(y_test, linreg_all.predict(X_test_all)))
    


def create_model(df,y ,X ,X_train, X_test, y_train, y_test, degree, random_state, test_size, alpha):
    
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    print('Baseline Training r^2:', linreg.score(X_train, y_train))
    print('Baseline Testing r^2:', linreg.score(X_test, y_test))
    print('Baseline Training MSE:', mean_squared_error(y_train, linreg.predict(X_train)))
    print('Baseline Testing MSE:', mean_squared_error(y_test, linreg.predict(X_test)))
    
    print("\n")
    
    ss = StandardScaler()
    ss.fit(X_train)

    X_train_scaled = ss.transform(X_train)
    X_test_scaled = ss.transform(X_test)
    
    linreg_norm = LinearRegression()
    linreg_norm.fit(X_train_scaled, y_train)

    print('Scaled Training r^2:', linreg_norm.score(X_train_scaled, y_train))
    print('Scaled Testing r^2:', linreg_norm.score(X_test_scaled, y_test))
    print('Scaled Training MSE:', mean_squared_error(y_train, linreg_norm.predict(X_train_scaled)))
    print('Scaled Testing MSE:', mean_squared_error(y_test, linreg_norm.predict(X_test_scaled)))
    
    
    print("\n")
    
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

    print('Continuous and Categorical')
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

# def category_features(df,y_test,test_size,random_state):
#     X_cat = df_reg[['Month', 'Origin', 'Dest']]
#     X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y,test_size=test_size, random_state=random_state)
#     # OneHotEncode Categorical variables
#     ohe = OneHotEncoder(handle_unknown='ignore')
#     ohe.fit(X_train_cat)

#     X_train_ohe = ohe.transform(X_train_cat)
#     X_test_ohe = ohe.transform(X_test_cat)

#     columns = ohe.get_feature_names(input_features=X_train_cat.columns)
#     cat_train_df = pd.DataFrame(X_train_ohe.todense(), columns=columns)
#     cat_test_df = pd.DataFrame(X_test_ohe.todense(), columns=columns)
#     X_train_all = pd.concat([pd.DataFrame(X_train_scaled), cat_train_df], axis = 1)
#     X_test_all = pd.concat([pd.DataFrame(X_test_scaled), cat_test_df], axis = 1)
#     linreg_all = LinearRegression()
#     linreg_all.fit(X_train_all, y_train)

#     print('Continuous and Categorical')
#     print('Training r^2:', linreg_all.score(X_train_all, y_train))
#     print('Testing r^2:', linreg_all.score(X_test_all, y_test))
#     print('Training MSE:', mean_squared_error(y_train, linreg_all.predict(X_train_all)))
#     print('Testing MSE:', mean_squared_error(y_test, linreg_all.predict(X_test_all)))
    
# def lasso_reg(alpha):
#     lasso = Lasso(alpha=alpha) #Lasso is also known as the L1 norm.
#     lasso.fit(X_train_all, y_train)
#     print( 'Lasso')
#     print('Training r^2:', lasso.score(X_train_all, y_train))
#     print('Testing r^2:', lasso.score(X_test_all, y_test))
#     print('Training MSE:', mean_squared_error(y_train, lasso.predict(X_train_all)))
#     print('Testing MSE:', mean_squared_error(y_test, lasso.predict(X_test_all)))
    
# def Ridge(alpha):
#     ridge = Ridge(alpha = alpha) #Ridge is also known as the L2 norm.
#     ridge.fit(X_train_all, y_train)
#     print('Ridge')
#     print('Training r^2:', ridge.score(X_train_all, y_train))
#     print('Testing r^2:', ridge.score(X_test_all, y_test))
#     print('Training MSE:', mean_squared_error(y_train, ridge.predict(X_train_all)))
#     print('Testing MSE:', mean_squared_error(y_test, ridge.predict(X_test_all)))
    
# def Backward_Elimination(X):
#     cols = list(X.columns)
#     pmax = 1
#     while (len(cols)>0):
#         p= []
#         X_1 = X[cols]
#         X_1 = sm.add_constant(X_1)
#         model = sm.OLS(y,X_1).fit()
#         p = pd.Series(model.pvalues.values[1:],index = cols)      
#         pmax = max(p)
#         feature_with_p_max = p.idxmax()
#     if(pmax>0.05):
#         cols.remove(feature_with_p_max)
#     else:
#         break
#     selected_features_BE = cols
#     print(selected_features_BE)

# def RFE_reg(X,y,X_train, X_test, y_train, y_test):
#     from sklearn.feature_selection import RFE
#     model = LinearRegression(X,y)
#     #Initializing RFE model
#     rfe = RFE(model, 5)
#     #Transforming data using RFE
#     X_rfe = rfe.fit_transform(X,y)  
#     #Fitting the data to model
#     model.fit(X_rfe,y)
#     print(rfe.support_)
#     print(rfe.ranking_)

#     #no of features
#     nof_list=np.arange(1,13)            
#     high_score=0
#     #Variable to store the optimum features
#     nof=0           
#     score_list =[]
#     for n in range(len(nof_list)):
#         X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 100)
#         model = LinearRegression()
#         rfe = RFE(model,nof_list[n])
#         X_train_rfe = rfe.fit_transform(X_train,y_train)
#         X_test_rfe = rfe.transform(X_test)
#         model.fit(X_train_rfe,y_train)
#         score = model.score(X_test_rfe,y_test)
#         score_list.append(score)
#         if(score>high_score):
#             high_score = score
#             nof = nof_list[n]
#     print("Optimum number of features: %d" %nof)
#     print("Score with %d features: %f" % (nof, high_score))
    
# def cross_validation(X_train, y_train,X,y):
#     lm = LinearRegression()

#     # store scores in scores object
#     # we can't use accuracy as our evaluation metric since that's only relevant for classification problems
#     # RMSE is not directly available so we will use MSE
#     scores = cross_val_score(lm, X_train, y_train, cv=10, scoring='r2')
#     mse_scores = cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')
#     print('Mean r2:',np.mean(scores))
#     print('Mean r2:',np.mean(scores))
#     print('10 Fold Score:',scores)
#     print ('10 Fold mean squared error',-(mse_scores) )
