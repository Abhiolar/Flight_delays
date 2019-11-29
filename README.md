# FlightDelays
 

Mod4 Project

Team Members: Itua Etiobhio, Yusuf Olodo

Business Problem

A US airline has approached our data science company and they are interested in knowing how they can optimise the arrival delays of their flights. We have found a dataset that solves this problem and answered 6 questions the airline was interested in finding out about for the EDA. Our mission is to find what variables that affect Arrival Delays by creating a robust model that can predict Arrival Delays. This mission is essential to the business of our client in terms of identifying what causes arrival delyas so they can minimising the amounts of delay claims that comes through their websites and the amount they have to pay out each time thereby increasing annual profit. Being able to predict arrival delay informs the business on how they can better treat their customers and plan their schedule wisely and if there will be a delay, they can make sure their customers still have a satisfactory experience on their flights.

    
# Methodology

The methodology centered around how we can answer the 6 questions posed to us by the client(the US airline). First step was to clean the data which included dropping null values and converting some certain datatypes as evidenced in the technical notebook.
We used Exploratory Data Analysis to answer the six questions we were most interested and tools such Matplotlib and Seaborn:
1. Which airports have the highest and lowest average delay minutes amongst LAX(LOS ANGELES INTERNATIONAL), MIA(MIAMI INTERNATIONAL), ORD(O'HARE INETERNATIONAL(CHICAGO)), DFW(DALLAS/FORT WORTH INTERNATIONAL), JFK( JOHN F KENNEDY INTERNATIONAL (NEW YORK))?
We found out that Miami International Aiport had the lowest average delay in minutes and JFK airport had the highest average delay in minutes.

2.Which months have the highest and lowest average delay minutes amongst the months of travel?
We found out that the month of June had the highest average delay minutes and the month of September had the lowest average delay minutes.

3..Which days have the highest and lowest average delay minutes amongst the days of travel?
The beginning of the moth had the highest average delay which subsided towards the middle of the month but picked back up again at the end of the month.

4.Which airports have the highest and lowest average windspeed(in terms of how terrible the weather tends to be)?
The airport with the highest average wind speed is Dallas Forth worth Texas and the airport with the lowest average windspeed is Los Angeles international airport

5.Which hours of the day have the highest and lowest average delay minutes across all the international airports ?
The hours with the highest average delay minutes are 02:00 and 03:00 and the hours with the lowest average delay minutes are 23:00 and 00:00 hours.

6.Which international airports have the highest and lowest average Arrival delay minutes?
The Miami International airport, has the highest average delay minutes in arrival followed by Los Angeles International and the lowest average for arrival delay is O'Hare International Airport

# Regression Model Output 

To be able to predict Arrival Delays, a regression model was built. A polynomial fit was also used on the model to try and improve the performance of the model. R-squared for the test data is 0.807 and R-squared for the training data is 0.827. The MSE is 19.3 for the test data and the MSE is 18.8 for the training data. This suggest the model fits the data well and is able to predict Arrival Delays fairly accurately. and out of the variabls used in the model, Delay_min ,Pressure ,Wind ,Temp ,Humidity influence the Arrival Delays the most.   

# Summary of files

This repo contains:

data_cleaning.py: Includes functions that perform the following tasks;
Deleting rows with null values and duplicated rows
Changing column names 
Changing columns into DateTime types

Final Project4 Notebook.ipynb: Technical notebook that includes 
Asuumption checks
visualisation of EDA
Feature Selection 
Regression model

regress.py: Includes functions that perform the following tasks;
creates regression models
cross validation 
Feature Selction
