![](https://res-1.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_120,w_120,f_auto,b_white,q_auto:eco/ajracsdqu5gmyfl6nai0)

# Prediction of wind and solar power generation

## Content
- [Project Description](#project-description)
- [Introduction](#introduction)
- [Goal](#goal)
- [Data](#data)
- [Method](#method)
- [Modelling and Results](#modelling-results)
- [Conclusions](#conclusions)
- [Dataset](#dataset)
- [Literature](#literature)

## Project Description  
This project focuses on the prediction of wind and solar power generation using machine learning techniques and different training datasets (i.e., different combination of weather variables and wind and solar power production data).

## Introduction
The share of renewable energy in the German energy mix has been rapidly growing in the recent years. In 2019, renewable energy amounted for ca. 46% of the total net electricity generation in Germany. Wind energy alone generated ca. 25% of the electricity produced, thus being the #1 energy source before brown coal (~20%) and nuclear energy (~14%) in the country that year. Solar (photovoltaic, PV) energy generated ca. 9% of the net German electricity in 2019. 

Since wind and solar energy are by nature intermittent and since both sources have a significant share in the electricity mix (~ 33%), their integration into the system poses a major challenge, namely to maintain balance in the system between electricity supply and demand. The intermittence and the non-controllable characteristics of the wind and solar production bring a number of other problems such as voltage fluctuations, local power quality and stability issues.

Thus, forecasting the output power of wind and solar systems is required for the effective operation of the power grid or for the optimal management of the energy fluxes occurring into the system. It is also necessary for estimating the reserves, for scheduling the power system (i.e. starting a power plant needs between 5 min for a hydraulic unit, 10min to 1h for fossil-fired power plants to 40 hours for a nuclear reactor), for congestion management, for the optimal management of the power storage and for trading the produced power in the electricity market and finally to achieve a reduction of the costs of electricity production. 

As a result, the prediction of wind and solar yields becomes of paramount importance and machine learning techniques have gained in popularity in recent years for wind and solar power forecasts. The relevant horizons of forecast usually range from 5 minutes to several days.

In regards with wind and solar power generation, forecasting models can be used in three different ways: 
- structural models, which are based on meteorological parameters like numerical weather prediction (NWP) data; 
- time-series models, which only consider the historically observed data of power generation as input features (endogenous forecasting); 
- hybrid models, which consider both, power generation data and other variables (e.g. NWP data) as exogenous variables (exogenous forecasting).

## Goal
The goal of the project is two-fold:
1. predict wind and solar power generation from weather conditions, i.e. data, at time t;
2. forecast wind and solar power generation at different time horizons, i.e. t+1h to t+6h.

## Data
### Sources
The following data were collected and used for the project:
- time-series data on wind and solar power production (MWh) and capacity (MW) for Germany as a whole, at hourly resolution (see [Literature](#literature));
- weather data relevant for power system modelling, at hourly resolution, for Germany, aggregated by Renewables.ninja from the NASA MERRA-2 reanalysis. It covers Germany using a population-weighted mean across all MERRA-2 grid cells within Germany (see [Literature](#literature)).

### Data pre-processing
The project focused on data for three consecutive years (2014-2016), at hourly resolution and comprised the following variables:
- solar power generation, in MWh;
- wind power generation, in MWh;
- solar installed capacity, in MW;
- wind installed capacity, in MW;
- windspeed at 10 meters above ground, in m/s;
- direct horizontal radiation, in W/m2;
- diffuse horizontal radiation, in W/m2;
- top-of-the-atmosphere solar irradiance, in W/m2;
- surface solar irradiance, in W/m2;
- air temperature 2 meters above ground, in °C;
- air density at ground level, in kg/m3;
- precipitation, in mm/hour;
- snowfall, in mm/hour;
- snow mass, in kg/m2;
- cloud cover fraction, a [0, 1] scale.


During the data pre-processing, the direct correlation between wind and solar power generation and the other variables was examined, as well as possible multicollinearity between weather data features. We checked for multicollinearity between features by computing the Pearson's correlation coefficients in a correlation matrix. In general, correlation coefficients of >0.7 indicates the presence of multicollinearity, which leads to model overfitting and in turns to higher modeling computational cost and possibly lower model performance. We could deduce that:
- there was a strong correlation between windspeed and wind power generation.
- there was a strong correlation between solar radiation/irradiance features and solar power generation.
- there was a strong multicollinearity between all radiation-related features, e.g. direct horizontal radiation, diffuse horizontal radiation, top-of-the-atmosphere solar irradiance and surface solar irradiance having the strongest correlation with solar power radiation.
- snowfall and snow mass are strongly correlated.

Since multicollinearity can reduce further models' performance, we limited the dataset to the following features:
- solar power generation, in MWh;
- wind power generation, in MWh;
- solar installed capacity, in MW;
- wind installed capacity, in MW;
- windspeed at 10 meters above ground, in m/s;
- surface solar irradiance, in W/m2;
- air temperature 2 meters above ground, in °C;
- air density at ground level, in kg/m3;
- precipitation, in mm/hour;
- snowfall, in mm/hour;
- cloud cover fraction, [0, 1] scale.

## Method
For the present project, we used three machine learning techniques.

### Linear Regression model
The output of a linear regression algorithm is a linear function of the input:
f:\mathbb{R}^{n}\rightarrow \mathbb{R}, \, y\hat{}\equiv f(\textup{x})= \beta ^{\textup{T}}\textup{x}+\beta _{\textup{0}}
[]!(https://latex.codecogs.com/gif.latex?f%3A%5Cmathbb%7BR%7D%5E%7Bn%7D%5Crightarrow%20%5Cmathbb%7BR%7D%2C%20%5C%2C%20y%5Chat%7B%7D%5Cequiv%20f%28%5Ctextup%7Bx%7D%29%3D%20%5Cbeta%20%5E%7B%5Ctextup%7BT%7D%7D%5Ctextup%7Bx%7D&plus;%5Cbeta%20_%7B%5Ctextup%7B0%7D%7D)
where 
\beta =\left ( \beta_{1}, ...,\, \beta_{n} \right )\in  \mathbb{R}^{n}
[]!(https://latex.codecogs.com/gif.latex?%5Cbeta%20%3D%5Cleft%20%28%20%5Cbeta_%7B1%7D%2C%20...%2C%5C%2C%20%5Cbeta_%7Bn%7D%20%5Cright%20%29%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D)
is a vector of parameters.The objective is to find the parameters which minimize the mean squared error:
\textup{argmin}_{\beta ,\beta _{0}}\frac{1}{N}\sum_{i=1}^{N}\left \( y_{i}-\beta ^{\textup{T}}\textup{x}_{i}-\beta _{0} \right )^{2}
[]!(https://latex.codecogs.com/gif.latex?%5Ctextup%7Bargmin%7D_%7B%5Cbeta%20%2C%5Cbeta%20_%7B0%7D%7D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cleft%20%5C%28%20y_%7Bi%7D-%5Cbeta%20%5E%7B%5Ctextup%7BT%7D%7D%5Ctextup%7Bx%7D_%7Bi%7D-%5Cbeta%20_%7B0%7D%20%5Cright%20%29%5E%7B2%7D)
This can be achieved using LinearRegression from the scikit-learn library.

### Decision Tree Regression model
The idea behind the decision tree technique is that a response or class Y from inputs X1, X2,…., Xp is required to be predicted. This is done by growing a binary tree. At each node in the tree, a test to one of the inputs, say Xi is applied. Depending on the outcome of the test, either the left or the right sub-branch of the tree is selected. Eventually a leaf node is reached, where a prediction is made. This prediction aggregates or averages all the training data points which reach that leaf. A model is obtained by using each of the independent variables. For each of the individual variables, mean squared error is used to determine the best split. The maximum number of features to be considered at each split is set to the total number of features.

Decision trees are sensitive to the specific data on which they are trained. If the training data is changed the resulting decision tree can be quite different and in turn the predictions can be quite different. Also Decision trees are computationally expensive to train, carry a big risk of overfitting, and tend to find local optima because they can’t go back after they have made a split. To address these weaknesses, we turn to Random Forest, which illustrates the power of combining many decision trees into one model.

### Random Forest Regression model
Random forest is an ensemble method, i.e. a technique that combines the predictions from multiple machine learning algorithms (in this case decision trees) together to make more accurate predictions than any individual model. Random forest is a bagging technique, meaning that it operates random sampling with replacement. It allows us to better understand for bias and variance with the dataset. Bagging makes each model run independently and then aggregates the outputs at the end without preference to any model. It operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. (https://towardsdatascience.com/random-forest-and-its-implementation-71824ced454f)

### Performance metrics
- Coefficient of determination
The performance measure that LinearRegression gives by default is the coefficient of determination R² of the prediction. It measures how well the predictions approximate the true values. A value close to 1 means that the regression makes predictions which are close to the true values. It is formally computed using the formula:

[]!(https://latex.codecogs.com/gif.latex?%5Cmathit%7BR%7D%5E%7B2%7D%3D1-%5Cfrac%7B%5Csum_%7Bi%7D%5E%7B%7D%5Cleft%20%28%20y_%7Bi%7D%20-y%5Cwidehat%7B%7D_%7Bi%7D%5Cright%20%29%5E%7B2%7D%7D%7B%5Csum_%7Bi%7D%5E%7B%7D%5Cleft%20%28%20y_%7Bi%7D%20-y%5Cbar%7B%7D%5Cright%20%29%5E%7B2%7D%7D)
\mathit{R}^{2}=1-\frac{\sum_{i}^{}\left ( y_{i} -y\widehat{}_{i}\right )^{2}}{\sum_{i}^{}\left ( y_{i} -y\bar{}\right )^{2}}

- Root mean squared error
A common performance metric is the Root Mean Squared Error (RMSE), given by:
RMSE=\sqrt{\frac{\sum_{i=1}^{n}\left ( yi-y\hat{}_{i} \right )^{2}}{n}}
[]!(https://latex.codecogs.com/gif.latex?RMSE%3D%5Csqrt%7B%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cleft%20%28%20yi-y%5Chat%7B%7D_%7Bi%7D%20%5Cright%20%29%5E%7B2%7D%7D%7Bn%7D%7D)

## Modelling and results
### Prediction of wind and solar power generation from weather data at time t
The goal here is to accurately predict with ML techniques wind and solar power generation based on weather data features only, at time t.

#### Model training
The models were trained over the data for the years 2014 and 2015.

#### Cross-validation
In order to evaluate the performance of the algorithm, a common practice is to perform cross-validation (cv). For the k-fold cv, the dataset is randomly split into k folds, the model is trained in k-1 of those folds, and the resulting model is validated on the remaining part of the data. The performance measure provided by the cv is then the average of the performance measure computed in each experiment. In the present work, we used as performance metrics cross_val_score from sklearn.model_selection, with a number of folds k = 5.

#### Model testing
The models were tested for the 2016 data.

#### Model tuning
- Feature selection
Feature selection is the process of reducing the number of input variables when developing a predictive model. It is desirable to reduce the number of input variables to both reduce the computational cost of modelling and ideally to improve the performance of the model. We selected the most predictive features based on different approaches such as the results of the correlation matrix or permutation features importances for the random forest regression algorithm.

- Permutation feature importance
The permutation feature importance is defined to be the decrease in a model score when a single feature value is randomly shuffled. This procedure breaks the relationship between the feature and the target, thus the drop in the model score is indicative of how much the model depends on the feature. We found that the solar irradiance and windspeed features were by far the most important features for predicting solar and wind energy power generation respectively.

- Hyperparameter tuning (random forest)
A hyperparameter is a parameter of the model that is set prior to the start of the learning process. Different models have different hyperparameters that can be set. In the present project, we adjusted the n_estimators parameters to optimize the models. The n_estimators parameter specifies the number of trees in the forest of the model. The default value for this parameter is 100, which means that 100 different decision trees will be constructed in the random forest. We found that n_estimators = 20 gave significantly similar results that with 100 decision trees for a reduced computational cost.

#### Results
- Prediction of solar power generation from weather data at time t
We created very accurate predicting models for solar power generation. 
A random forest regression algorithm using solar irradiance, windspeed, precipitation, cloud cover and air density as selected features and 20 decision trees gave a R² value of 0.961. 
Worth to mention is that the linear regression algorithm using solely the solar irradiance as input variable gave similar results with a a R² value of 0.953 for a reduced computational  cost.

- Prediction of wind power generation from weather data at time t
The predicting models for wind power generation were somewhat accurate. The best performance was obtained with the linear regression model (R²=0.784) using wind capacity, windspeed, solar irradiance, precipitation, snowfall, cloud cover and air density as input variables. The random forest regression model with 100 decision trees gave similar results (R²=0.770) for a higher computational cost.

### Forecasting wind and solar power generation at different time horizons
The goal here is to accurately forecast with ML techniques wind and solar power generation at different time horizons, from t+1h to t+6h. To do so, we used as input variables: 1. the historical data for power generation, 2. a combination of historical and weather data.

#### Model training and testing
- The models were trained over 80% of the dataset, i.e. the data for the three consecutive years 2014, 2015 and 2016. Also, a 5-fold cross-validation was performed during the model training.
- The models were tested on the remaining 20% of the dataset.

#### Results
- Forecasting wind power generation at different time horizons
Both linear regression and random forest algorithms predicted accurately wind power generation, with R² values ~ 0.990 at t+1h (with historical data only, and hybrid models with historical and weather data) down to R² ~ 0.738 (historical data) and 0.896 (hybrid models) at t+6h. Worth to mention is that predictions models based on the linear regression algorithms gave similar results for reduced computational costs.

- Forecasting solar power generation at different time horizons
Depending on the models (historical/hybrid data, linear regression/random forest regression), accurate forecasts could be made for t+1h (R²: 0.975-0.881). A hybrid linear regression model based on historical and all available weather data could forecast solar power generation with  R² values of 0.913 (t+2h), 0.832 (t+3h), 0.744 (t+4h). Other models (historical/hybrid data) gave very poor forecasts for solar power generation.

## Conclusions
Thanks to machine learning techniques, we could build models that predicted efficiently wind and solar power generation, whether the models focused on the prediction at time t from weather data or on forecasts at different time horizons from historical data and hybrid models. This work constitutes a first step in the development of more accurate prediction models, that would explore further algorithm such as artifitial neural networks and k-nearest neighbors. Also, we could feed additional features to the dataset, such as seasonality (winter, spring, summer and fall) and diurnality (day/night) in order to gain accuracy in the prediction models. Lastly, it would be beneficial to work on a dataset from a single production site (solar/wind park) where the specific installed capacity and weather data would help build more accurate prediction models.

## Dataset  
- [Open Power System Data Platform](https://data.open-power-system-data.org/weather_data/)
- [Renewables.ninja Platform](https://www.renewables.ninja/)
- [NASA-MERRA2](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/)

## Literature
- [Machine Learning methods for solar radiation forecasting: a review](https://www.sciencedirect.com/science/article/abs/pii/S0960148116311648)
- [Solar power forecasting with machine learning techniques](https://www.math.kth.se/matstat/seminarier/reports/M-exjobb18/180601f.pdf)
- [Predicting wind and solar generation from weather data using Machine Learning](https://medium.com/hugo-ferreiras-blog/predicting-wind-and-solar-generation-from-weather-data-using-machine-learning-998d7db8415e)

