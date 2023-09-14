# pip install pandas typing plotly
import os
import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
# import plotly.express as px
# pip install -U ydata-profiling
# from ydata_profiling import ProfileReport

# pip install scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

import warnings
warnings.filterwarnings("ignore")


"""---------------------------------------------------------------------------------------
------------------------------ GLOBAL VARIABLES ---------------------------------------"""
r_st = 42
"""######################################################################################
# DATA BY LOCATION ######################################################################
# |-> BBC
# |--> Data by date
# |---> 1 - September 18th 2020
#####################################################################################"""
bbc_sep_2020_1 = "Logs/BBC/1 - Biscayne Bay - September 18th 2020/20200918_105724_Mission1_09_11_IVER2-218.csv"
bbc_sep_2020_2 = "Logs/BBC/1 - Biscayne Bay - September 18th 2020/20200918_105852_TrianglePathMission_IVER2-218.csv"
bbc_sep_2020_3 = "Logs/BBC/1 - Biscayne Bay - September 18th 2020/20200918_110002_TrianglePathMission_IVER2-218.csv"
bbc_sep_2020_4 = "Logs/BBC/1 - Biscayne Bay - September 18th 2020/20200918_115659_PleaseWork_IVER2-218.csv"
bbc_sep_2020_5 = "Logs/BBC/1 - Biscayne Bay - September 18th 2020/20200918_120605_PleaseWorkSRP_IVER2-218.csv"
bbc_sep_2020_6 = "Logs/BBC/1 - Biscayne Bay - September 18th 2020/ManualLog_11_05_09_92.csv"
bbc_sep_2020_7 = "Logs/BBC/1 - Biscayne Bay - September 18th 2020/ManualLog_12_19_47_14.csv"
# |---> 2 - October 4th 2020
bbc_oct_2020_1 = "Logs/BBC/2 - Biscayne Bay - October 4th 2020/20201004_171844_TestForLittleRiver_IVER2-218.csv"
bbc_oct_2020_2 = "Logs/BBC/2 - Biscayne Bay - October 4th 2020/20201004_180324_LittleRiverOct4_IVER2-218.csv"
bbc_oct_2020_3 = "Logs/BBC/2 - Biscayne Bay - October 4th 2020/ManualLog_17_00_45_64.csv"
# |---> 3 - January 30th 2021
bbc_jan_2021_1 = "Logs/BBC/3 - Biscayne Bay - January 30th 2021/20210130_124101_jan302021_bbc_IVER2-218.csv"
bbc_jan_2021_2 = "Logs/BBC/3 - Biscayne Bay - January 30th 2021/20210130_130339_jan302021_bbc_IVER2-218.csv"
bbc_jan_2021_3 = "Logs/BBC/3 - Biscayne Bay - January 30th 2021/20210130_131335_jan302021_bbc_IVER2-218.csv"
bbc_jan_2021_4 = "Logs/BBC/3 - Biscayne Bay - January 30th 2021/20210130_135713_jan302021_bbc_IVER2-218.csv"
bbc_jan_2021_5 = "Logs/BBC/3 - Biscayne Bay - January 30th 2021/20210130_140634_jan302021_bbc_IVER2-218.csv"
bbc_jan_2021_6 = "Logs/BBC/3 - Biscayne Bay - January 30th 2021/20210130_143547_jan302021_bbc_IVER2-218.csv"
# |---> 4 - February 13th 2021
bbc_feb_2021_1 = "Logs/BBC/4 - Biscayne Bay - February 13th 2021/20210213_150135_feb132021_bbc_IVER2-218.csv"
bbc_feb_2021_2 = "Logs/BBC/4 - Biscayne Bay - February 13th 2021/20210213_155452_feb132021_b_bbc_IVER2-218.csv"
# |---> 5 - March 16th 2021
bbc_mar_2021_1 = "Logs/BBC/5 - Biscayne Bay - March 16th 2021/20210316_112824_onthespot_mar16_IVER2-218.csv"
# |---> 6 - June 9th 2021
bbc_jun_2021_1 = "Logs/BBC/6 - Biscayne Bay - June 9th 2021/20210609_105413_mongabay2_IVER2-218.csv"
bbc_jun_2021_2 = "Logs/BBC/6 - Biscayne Bay - June 9th 2021/20210609_110304_mongabay2_IVER2-218.csv"
bbc_jun_2021_3 = "Logs/BBC/6 - Biscayne Bay - June 9th 2021/20210609_112348_mongabay_IVER2-218.csv"
# |---> 7 - September 2nd 2021
bbc_sep_2021_1 = "Logs/BBC/7 - Biscayne Bay - September 2nd 2021/sep2_2021_bbc_mission_1.csv"
bbc_sep_2021_2 = "Logs/BBC/7 - Biscayne Bay - September 2nd 2021/sep2_2021_bbc_mission_2.csv"
bbc_sep_2021_test = "Logs/BBC/7 - Biscayne Bay - September 2nd 2021/sep2_2021_bbc_test.csv"
# |---> 9 - December 16th 2021
bbc_dec_2021 = "Logs/BBC/9 - Biscayne Bay - December 16th 2021/20211216_dataset.csv"
# |---> 10 - February 4th 2022
bbc_feb_2022 = "Logs/BBC/10 - Biscayne Bay - February 4th 2022/20220204_dataset.csv"
# |---> 11 - August 28th 2022
bbc_aug_2022_1 = "Logs/BBC/11 - Biscayne Bay - August 28th 2022/08-28-22-mission1.csv"
bbc_aug_2022_2 = "Logs/BBC/11 - Biscayne Bay - August 28th 2022/08-28-22-mission2.csv"
# |---> 12 - October 7th 2022
bbc_oct_2022_1 = "Logs/BBC/12 - Biscayne Bay - October 7th 2022/10-7-22-mission1.csv"
bbc_oct_2022_2 = "Logs/BBC/12 - Biscayne Bay - October 7th 2022/10-7-22-mission2.csv"
bbc_oct_2022_3 = "Logs/BBC/12 - Biscayne Bay - October 7th 2022/10-7-22-mission3.csv"
# |---> 13 - November 16th 2022
bbc_nov_2022_1 = "Logs/BBC/13 - Biscayne Bay - November 16th 2022/11-16-22-mission1.csv"
bbc_nov_2022_2 = "Logs/BBC/13 - Biscayne Bay - November 16th 2022/11-16-22-mission2.csv"
#########################################################################################

"""--------------------------------------------------------------------------------------
---------------------------------- MODELS ---------------------------------------------"""
"""
#########################################################################################
# selectDataframe #######################################################################
#########################################################################################
"""
def selectDataframe(dataset: str, selected_parameters: List[str]):
    entire_ds = pd.read_csv(dataset) # read the entire dataset as a dataframe
    # print(entire_ds.columns) # to print the head of the dataframe

    selected_parameters.insert(0, "Latitude")
    selected_parameters.insert(1, "Longitude")
    selected_parameters.insert(2, "Time hh:mm:ss")

    try:
        partial_ds = entire_ds[selected_parameters]
        print("The dataframe was successfully created!")
        # print(partial_ds.columns) #to print the head of the dataframe
        partial_ds = partial_ds.rename(columns={"Latitude": "lat", "Longitude": "lon"})
        return partial_ds, entire_ds
    except ValueError:
        return print("Oops! Some selected water parameters do not exist in this dataset. Try again...")

"""
#########################################################################################
# getDataframe ##########################################################################
#########################################################################################
"""
def getDataframe(ds):
    parameters = [1, 2, 3, 4]
    water_parameters = {
        1: "ODO mg/L",
        2: "Temperature (c)",
        3: "pH",
        4: "Total Water Column (m)"
    }
    selected_parameters = [water_parameters.get(key) for key in parameters]
    # print("selected_parameters: ", selected_parameters)

    # Choose the dataset
    ds_name = [i for i, a in globals().items() if a == ds][0]
    print("The variable name:", ds_name)
    dataset = ds

    # calling function selectDataframe
    partial_ds, entire_ds = selectDataframe(dataset, selected_parameters)

    zoom_lat = partial_ds['lat'].mean()
    zoom_long = partial_ds['lon'].mean()
    # print("zoom_lat: ", zoom_lat)
    # print("zoom_long: ", zoom_long)

    return partial_ds, entire_ds, ds_name

"""
#########################################################################################
# data_profiling ########################################################################
#########################################################################################
"""
def data_profiling(partial_ds):
    profile = ProfileReport(partial_ds, correlations={
        "auto": {"calculate": True},
        "pearson": {"calculate": True},
        "spearman": {"calculate": True},
        "kendall": {"calculate": True},
        "phi_k": {"calculate": True},
        "cramers": {"calculate": True},
      },
      html={'style':{'full_width':True}})
    # profile.to_notebook_iframe() # if its on a notebook
    profile.to_file("output.html")

"""
#########################################################################################
# all_models ############################################################################
#########################################################################################
"""
def all_models(X, y, mds, target, ds_name, tsize = 0.2):
    try:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = tsize, random_state = r_st)

        # Initialize lists to store the results
        results = []
        model_names = []
        best_parameters = []

        # Iterate over the models and evaluate their performance
        for name, model, params in mds:
            # Create GridSearchCV object with the model and hyperparameters
            grid_search = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5)

            # Fit the model with the training data
            grid_search.fit(X_train, y_train)

            # Get the best model
            best_model = grid_search.best_estimator_

            # Make predictions on the testing set
            y_pred = best_model.predict(X_test)

            # print("Best Model Parameters:")
            # print(best_model.get_params())

            # Create a figure and axis
            fig, ax = plt.subplots()
            # Plot the two lines
            ax.plot(y_test.to_list(), label='Measured Data')
            ax.plot(list(np.round(y_pred, 2)), label='Predicted Data')
            ax.set(xlabel='datapoints', ylabel='Value', title=f'{name} - {target}')
            ax.grid()
            # Add a legend
            ax.legend()
            # Show the plot
            # plt.show()

            # save charts in the right place
            # create sublevel folders
            subdir_path = f"./charts/{ds_name}/"
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
                print(f"Directory '{subdir_path}' created.")
            # create subsublevel folders
            subsubdir_path = f"./charts/{ds_name}/{target.replace(' ', '').replace('/', '')}/"
            # print(subsubdir_path)
            if not os.path.exists(subsubdir_path):
                os.makedirs(subsubdir_path)
                print(f"Directory '{subsubdir_path}' created.")
            chartname = f"{name}_{target}.png"
            chartname = chartname.replace(" ", "").replace("/", "")
            # print(subsubdir_path + chartname)
            plt.savefig(subsubdir_path + chartname)

            # Evaluate the model using different metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            nrmse = ((rmse)/(max(y_test)-min(y_test))) * 100
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Store the results in lists
            results.append([mse, rmse, nrmse, mae, r2])
            model_names.append(name)
            best_parameters.append(best_model.get_params())

        # Create a dataframe with the results
        columns = ["MSE", "RMSE", "NRMSE", "MAE", "R-squared"]
        results_df = pd.DataFrame(results, columns=columns, index=model_names)
        filename = f"negativa_{target}"
        filename = filename.replace(" ", "").replace("/", "")
        # create subsublevel folders
        subsubdir_path = f"./charts/{ds_name}/{target.replace(' ', '').replace('/', '')}/"
        results_df.to_csv(f"{subsubdir_path+filename}.csv", index=False)

        # Print the results dataframe
        # print(results_df)
        # print(best_parameters)
        return best_parameters, results_df
    except:
        print("An exception occurred!!!")
        return None, None


"""---------------------------------------------------------------------------------------
---------------------------------------- Main -----------------------------------------"""
if __name__ == '__main__':
    dir_path = "./charts/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")

    # Choose the database in here. e.g.: bbc_nov_2022_1
    partial_ds, entire_ds, ds_name = getDataframe(bbc_nov_2022_1)
    # print(partial_ds.shape)

    List_X = [
        partial_ds[['Temperature (c)', 'pH', 'Total Water Column (m)']],
        partial_ds[['Temperature (c)', 'ODO mg/L', 'Total Water Column (m)']],
        partial_ds[['ODO mg/L', 'pH', 'Total Water Column (m)']]]
    List_y = [
        partial_ds['ODO mg/L'],
        partial_ds['pH'],
        partial_ds['Temperature (c)']]

    targets = ['ODO mg/L', 'pH', 'Temperature (c)']

    for i, (X, y) in enumerate(zip(List_X, List_y)):
        # Define the hyperparameters to tune
        print(i)
        print(X)

        parameters = [
            {'fit_intercept': [True, False]},  # Linear Regrssion
            {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]},  # Random Forest Regressor
            {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},  # Support Vector Regressor
            {'hidden_layer_sizes': [(100,), (50, 50), (50, 100, 50)],  # Multi-Layer Perceptron
             'activation': ['relu', 'tanh'],
             'solver': ['adam', 'lbfgs'],
             'alpha': [0.0001, 0.001, 0.01]}
        ]
        # Define the models to evaluate
        models = [
            ("Linear Regression", LinearRegression(), parameters[0]),
            ("Random Forest Regressor", RandomForestRegressor(random_state = r_st), parameters[1]),
            ("Support Vector Machines", SVR(), parameters[2]),
            ("Multi-Layer Perceptron", MLPRegressor(random_state = r_st), parameters[3]),
        ]

        best_param, results = all_models(X, y, models, targets[i], ds_name)
        print(best_param)
        print(results)

    print("----------- THE END! -----------")
