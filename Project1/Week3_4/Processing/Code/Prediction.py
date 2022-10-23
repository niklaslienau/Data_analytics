import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
from sklearn.linear_model import LinearRegression
import random

#Define Data and Figure Path
data_path = "C:/Users/nikil/OneDrive/Desktop/Uni/M.Sc/Data Analytics/Project1/Week3_4/Processing/Output/"
fig_path = "C:/Users/nikil/OneDrive/Desktop/Uni/M.Sc/Data Analytics/Project1/Week3_4/Processing/Maps_Figures/"

#Import the data
data=pd.read_csv(data_path + "data_for_pred.csv")
data.drop(columns=["Unnamed: 0"], inplace=True)
#drop observations with NAs
data=data.dropna()

#Drop outliers -> everything not within 3 standard deviations
data = data[(np.abs(stats.zscore(data["Population"])) <3 )]
data= data[(np.abs(stats.zscore(data["appointments"])) <3 )]
#set random seed to make results reproducable
random.seed(1)
#####

#feature matrix
X= data.drop(columns=["Population"])
#outcome vector
y=data["Population"]
feature_list= X.columns
#Randomly split data in into test (25%) and train (75%)
train_X, test_X, train_y, test_y =train_test_split(X, y, test_size = 0.25)

#MODEL 1:
#Random Forest baseline model
#n_estimators is the number of individual regression trees used to build the forest
rf_model = RandomForestRegressor(n_estimators = 1000)
# Train the model on training data
rf_model.fit(train_X, train_y)
#use trained model to predict population for test set
pred=rf_model.predict(test_X)

#Which variables were important
for i in range(len(feature_list)):
    print(feature_list[i])
    print(round(rf_model.feature_importances_[i],3))
    print("")

#Now do some hyperparameter tuning

#n_estimators = number of trees in the forest
# max_depth = max number of levels in the tree
# bootstrap = bootsrapping with or without replacement
# max_sample = %of observations sample used for bootstrap samples

# Number of trees
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 5, num = 4)]

#selecting samples with or without replacement when bootstrapping
bootstrap = [True, False]

# %of observations sample used for bootstrap samples
max_samples=[float(x) for x in np.linspace(0, 1, num = 10)]
#Define grid with parameters we want to tune
grid= {"n_estimators":n_estimators, "max_depth":max_depth, "bootstrap":bootstrap, "max_samples":max_samples}
# Use the random grid to search for best hyperparameters
# Find best hyperparameters with 3 fold CV
rf_hyper = RandomizedSearchCV(estimator = rf_model, param_distributions = grid, n_iter = 100, cv = 3, verbose=2, n_jobs = -1)
# Fit the finetuned model
rf_hyper.fit(train_X, train_y)

#What are the best hyper parameter values
print(rf_hyper.best_params_)

#MODEL 2
#Now use  model with best hyperparameters to predict pop values for test set
pred_hyper=rf_hyper.predict(test_X)

#MODEL 3
#Lets also fit a simple linear regression on the training set and use it to predict population values on the test set
#The Linear Model can serve as a baseline to compare the more complicated RF model against
ln_total=LinearRegression().fit(train_X,train_y)
pred_lm=ln_total.predict(test_X)


#Evaluate Performance for the 3 Models: Linear Reg, RF default, RF after tuning hyperparameters
#Model Performance Metrics: How much percent lie within range of (100,250,500) and MSE

#1. Linear Regression:
print("Model Performance Linear Reg")
print("100")
print(sum(abs(pred_lm-test_y)<100)/len(test_y))
print("250")
print(sum(abs(pred_lm-test_y)<250)/len(test_y))
print("500")
print(sum(abs(pred_lm-test_y)<500)/len(test_y))
print("Mean Squared Error")
print(mean_squared_error(pred_lm, test_y))

#2. Random Forest
print("Model Performance Random Forest")
print("100")
print(sum(abs(pred-test_y)<100)/len(test_y))
print("250")
print(sum(abs(pred-test_y)<250)/len(test_y))
print("500")
print(sum(abs(pred-test_y)<500)/len(test_y))
print("Mean Squared Error")
print(mean_squared_error(pred, test_y))

#3. Random Forest Hyper Tuned
print("Model Performance Hypertuned Random Forest")
print("100")
print(sum(abs(pred_hyper-test_y)<100)/len(test_y))
print("250")
print(sum(abs(pred_hyper-test_y)<250)/len(test_y))
print("500")
print(sum(abs(pred_hyper-test_y)<500)/len(test_y))
print("Mean Squared Error")
print(mean_squared_error(pred_hyper, test_y))


