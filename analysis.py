import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def explore_data(data):
    fig = plt.figure(figsize =(20,10))
    plt.ylabel('Frequency', fontsize =14)
    plt.xlabel('Delay_minutes', fontsize =14)
    sns.kdeplot(data['Delay_min'],shade = True );
    plt.title('KDE plot for the Delay_minutes(target variable)', fontsize = 14)
    

def questions_viz(data):
    plt.figure(figsize =(16,6))
    sns.boxplot(x="Origin", y="Delay_min", data=data, showfliers = False);
    plt.title('Q1.Categorical plots of delay minutes across all the airports');
    plt.figure(figsize =(16,6))
    sns.boxplot(x="Origin", y="Delay_min", data=data, showfliers = True);
    plt.title('Q1.Categorical plots of delay minutes across all the airports with outliers');
    plt.figure(figsize =(16,6))
    sns.boxplot(x="Month", y="Delay_min", data=data, showfliers = False);
    plt.title('Q2.Categorical plots of delay minutes across all the 10 months of travel recorded');
    plt.figure(figsize=(16, 6))
    sns.boxplot(x="DayofMonth", y="Delay_min" ,data=data, showfliers = False);
    plt.title('Q3.Categorical plots of delay minutes across all the 28-31 days/dates of travel recorded');
    plt.figure(figsize =(16,6))
    sns.boxplot(x="Origin", y="Wind", data=data, showfliers = False);
    plt.title('Q4.Categorical plots of Windspeed across all the airports');
    plt.figure(figsize =(16,6))
    sns.boxplot(x="Hour", y="Delay_min", data=data, showfliers = False);
    plt.title('Q5.Categorical plots of Windspeed across all the airports');
    plt.figure(figsize =(16,6))
    sns.boxplot(x="Dest", y="ArrDelay", data=data, showfliers = False);
    plt.title('Q6.Categorical plots of Arrival delay minutes across all the airports');
    
    
def assumptions_check(data):
    print(data.corr() *100)
    fig = plt.figure(figsize = (20,10))

    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)


    sns.scatterplot(x = data['Temp'],y = data['ArrDelay'] ,color = 'red', ax = ax1)
    sns.scatterplot(x= data['Delay_min'], y = data['ArrDelay'], color = 'blue', ax = ax2)
    sns.scatterplot(x = data['Wind'], y = data['ArrDelay'], color = 'green', ax = ax3)
    sns.scatterplot(x= data['Pressure'], y = data['ArrDelay'], color = 'yellow', ax = ax4);
    
    fig = plt.figure(figsize = (20,10))

    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)


    sns.scatterplot(x = data['AirTime'],y = data['ArrDelay'] ,color = 'red', ax = ax1)
    sns.scatterplot(x= data['Hour'], y = data['ArrDelay'], color = 'blue', ax = ax2)
    sns.scatterplot(x = data['Humidity'], y = data['ArrDelay'], color = 'green', ax = ax3)
    sns.scatterplot(x= data['Origin'], y = data['ArrDelay'], color = 'yellow', ax = ax4);

    
    
def normality_check(data):
    fig = plt.figure(figsize = (20,10))

    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    sns.distplot(data['Delay_min'], color = 'red', ax = ax1)
    sns.distplot(data['Pressure'], color = 'blue', ax = ax2)
    sns.distplot(data['ArrDelay'], color = 'green', ax = ax3)
    sns.distplot(data['Wind'], color = 'yellow', ax = ax4);
    
    fig = plt.figure(figsize = (20,10))

    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    sns.distplot(data['Temp'], color = 'red', ax = ax1)
    sns.distplot(data['Hour'], color = 'blue', ax = ax2)
    sns.distplot(data['Humidity'], color = 'green', ax = ax3)
    sns.distplot(data['AirTime'], color = 'yellow', ax = ax4);

def homoscedasticity_check(data):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # build the formula 
    f = 'ArrDelay~Delay_min'
    # create a fitted model in one line
    model = smf.ols(formula=f, data=data).fit()
    print(model.summary())
    #visulaizing the error term for variance and heteroscedasticity
    fig = plt.figure(figsize=(15,8))
    fig = sm.graphics.plot_regress_exog(model, "Delay_min", fig=fig)
    plt.show()
    import scipy.stats as stats
    residuals = model.resid
    fig = sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
    fig.show()
