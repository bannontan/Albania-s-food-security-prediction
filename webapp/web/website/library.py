#importing the neccessary library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats #for statistical analysis
from scipy.stats import norm #for statistical analysis
from datetime import datetime #for time-series plots
# import statsmodels #for integration with pandas and analysis
# import statsmodels.api as sm # for regression modules
# from statsmodels.formula.api import ols # for regression modules
import matplotlib.pyplot as plt 
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
# from sklearn.model_selection import train_test_split
# # from sklearn.metrics import r2_score, mean_squared_error
# import warnings
# warnings.filterwarnings("ignore")
import io
import base64

def normalize_z(dfin, columns_means=None, columns_stds=None):
    if columns_means is None:
        columns_means= dfin.mean(axis=0)
    if columns_stds is None:
        columns_stds= dfin.std(axis=0)
    dfout= (dfin- columns_means)/ columns_stds
        
    return dfout, columns_means, columns_stds

def get_features_targets(df, feature_names, target_names):
    df_feature = pd.DataFrame(df.loc[:, feature_names])
    df_target = pd.DataFrame(df.loc[:, target_names]) 

    return df_feature, df_target

def prepare_feature(df_feature):
    cols= df_feature.shape[1]
    if type(df_feature)==pd.DataFrame:
        np_feature= df_feature.to_numpy()
        
    else:
        np_feature= df_feature
    if np_feature.ndim == 1:
        np_feature= np_feature.reshape(-1,cols)
     # Add a column of ones for the intercept term
    ones = np.ones((np_feature.shape[0], 1)) ##X.shape[0] is to find the number of rows in matrix X,
    #and create a numpy array column vector of 1s 
    X = np.concatenate((ones, np_feature), axis=1)
    return X

def prepare_target(df_target):
    cols=df_target.shape[1] ##3 columns
    
    if type(df_target)==pd.DataFrame:
        np_target= df_target.to_numpy()
    else:
        np_target= df_target
        
    target= np_target.reshape(-1,cols)
    
    return target

def predict_linreg(df_feature, beta, means=None, stds=None):
    # Normalize the features using z normalization
    df_feature_normalized, means, stds = normalize_z(df_feature, means, stds)
    
    # Prepare the feature for prediction (add a column of ones for the intercept)
    X = prepare_feature(df_feature_normalized)

    # Calculate the predicted y values
    y_pred = calc_linreg(X, beta)

    return y_pred

def calc_linreg(X, beta):
    return np.matmul(X,beta)

def split_data(df_feature, df_target, random_state=None, test_size=0.5):
    indexes = df_feature.index
    if random_state != None:
        np.random.seed(random_state)
    test_index = np.random.choice(indexes, int(len(indexes) * test_size), replace = False)
    indexes = set(indexes)
    test_index = set(test_index)
    train_index = indexes - test_index
    
    df_feature_train = df_feature.loc[list(train_index), :]
    df_feature_test = df_feature.loc[list(test_index), :]
    df_target_train = df_target.loc[list(train_index), :]
    df_target_test = df_target.loc[list(test_index), :]
    
    return df_feature_train, df_feature_test, df_target_train, df_target_test
  
def r2_score(y, ypred):
    y_mean = np.mean(y)
    ss_res = np.sum(((y-ypred)**2))
    ss_tot = np.sum(((y-y_mean)**2))

    return 1 - (ss_res / ss_tot)
def mean_squared_error(target, pred):
    summation = np.sum(((target - pred)**2))
    n = target.size
    
    return (1/n) * summation

def compute_cost_linreg(X, y, beta):
    J=0
    m = y.shape[0]  # Number of training examples
    predictions = calc_linreg(X, beta)
    errors = predictions - y
    squared_errors = np.matmul(errors.T, errors)
    J = (1 / (2 * m)) * np.sum(squared_errors)
    return J

def gradient_descent_linreg(X, y, beta, alpha, num_iters):
    J_storage = []
    m = y.shape[0]  ## number of training examples
    
    for i in range(num_iters):
        h = calc_linreg(X, beta)  ## calculate predicted value
        loss = h - y  ## calculate the errors
        gradient = np.matmul(X.T, loss) / m  ## caculate X_transposed * errors(loss) / m
        beta = beta - alpha * gradient
        J_storage.append(compute_cost_linreg(X, y, beta))
    
    return beta, J_storage

def run(future_year, future_gdp, future_import,future_crop): ##the different user input
    df = pd.read_csv('Albania_2d copy.csv')
    # Impute missing values using mean
    df_median = df.fillna(df.median())  
    ##dropping multicollinearity
    df_median= df_median.drop(["crop_pro_f", "pes", "crop_area_f"," pop","temp","forest","exp",'pol','flood'], axis=1)
    # Extract the features after dropping and the targets
    df_features, df_target = get_features_targets(df_median, ['year', 'inf',
       'crop_area_t', 'crop_pro_t', 'crop_y', 'prep', 'gdp', 'import'], ['undern'])
    # Split the data set into training and test
    df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features, df_target, random_state=100, test_size=0.3)



   # Normalize training data
    df_features_train_z, columns_means, columns_stds = normalize_z(df_features_train,columns_means=None, columns_stds=None)


   # Store the means and standard deviations
    training_mean = columns_means
    training_std = columns_stds

   # Prepare features and targets
    X_train = prepare_feature(df_features_train_z)
    target_train = prepare_target(df_target_train)


   # Gradient Descent
    iterations = 3000
    alpha = 0.001
    beta = np.zeros((9, 1))
    beta, J_storage = gradient_descent_linreg(X_train, target_train, beta, alpha, iterations)
    beta[1]=0


   # Update last known values to make prediction
    last_known_values = df_features.iloc[-1].copy()
    last_known_values['year'] = float(future_year)
    last_known_values['gdp'] = float(future_gdp)
    last_known_values['import'] = float(future_import)
    last_known_values['crop_y']= float(future_crop)


   # Convert to DataFrame
    X_future = pd.DataFrame([last_known_values])
    df_features_test = pd.concat([df_features_test, X_future], ignore_index=True)
   # display(df_features_test)


   # Make prediction
    pred = predict_linreg(df_features_test, beta,training_mean,training_std)
    
    return df_features, df_target, last_known_values, pred, future_year


    
   
def get_pred_string(pred, future_year):
    pred= pred.flatten()[-1]
    return f"The predicted percentage of undernourishment in Albania (% of its population) in {future_year} is {pred.round(1)}"

def generate_prediction_graph(df_features, df_target, last_known_values, pred):
    x_historical = df_features['year']
    y_historical = df_target 

    # Plot historical data
    plt.figure(figsize=(10, 6))
    plt.scatter(x_historical, y_historical, label='Historical Data')
    plt.plot(x_historical, y_historical, linestyle='--', alpha=0.5)

    # Add predicted point
    # 'pred' is your predicted value, and 'future_year' is the year of prediction
    x_future = last_known_values["year"]
    y_future = pred.flatten()[-1]
    plt.scatter(x_future, y_future, color='red', label='Predicted Point')

    # Connect the last known point to the predicted point
    last_known_year = x_historical.iloc[-1]
    last_known_value = y_historical.iloc[-1]

    # Ensure these are scalar values for plotting
    last_known_year = float(last_known_year)
    last_known_value = float(last_known_value)
    x_future = float(x_future)
    y_future = float(y_future)

    plt.plot([last_known_year, x_future], [last_known_value, y_future], color='red', linestyle='-', marker='o',alpha=0.5)

    # Labeling the plot
    plt.xlabel('Year')
    plt.ylabel('Prevelence of Undernourishment')  # Replace with your target variable name
    plt.title('Historical Data and Prediction')
    plt.legend()
    
    # Convert plot to a format that can be displayed in HTML
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return 'data:image/png;base64,{}'.format(plot_url)