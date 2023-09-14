# pip install pandas typing plotly
import os
import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
# import plotly.express as px
# pip install scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pydeck as pdk



make_graph = True #Set it to true if you want to use the make_graph.py as an auxiliary function
save_heatmap = True #Heat map graph
save_water_features = True #Water features graph
if make_graph:
    from make_graph import generateGraph, generateCorrHeatmap
    format = 'eps'

r_st = 42


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

st.set_page_config(
    page_title="Machine Learning Methods",
    layout="wide",
    initial_sidebar_state="collapsed"
)
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

    # zoom_lat = partial_ds['lat'].mean()
    # zoom_long = partial_ds['lon'].mean()
    # print("zoom_lat: ", zoom_lat)
    # print("zoom_long: ", zoom_long)

    return partial_ds, entire_ds, ds_name


def all_models(X, y, mds, target, tsize=0.2):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tsize, random_state=r_st)

    # For printing purposes
    X_test = X_test.sort_index()
    y_test = y_test.sort_index()

    # Initialize lists to store the results
    results = []
    model_names = []
    best_parameters = []

    # Iterate over the models and evaluate their performance
    for name, model, params in mds:

        pipeline_model = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('model', model)])
        # Create GridSearchCV object with the model and hyperparameters
        # grid_search = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5)
        grid_search = GridSearchCV(pipeline_model, params, scoring='neg_mean_squared_error', cv=5)

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

        if make_graph:
            pieces = target.replace("(", "").replace(")", "").split()
            ylabel = pieces[0]
            if len(pieces) == 2:
                # print(target)
                if target == 'Temperature (c)':

                    unit = '$[°C]$'
                else:
                    unit = '$[' + pieces[1] + ']$'
                ylabel = ylabel + ' ' + unit

            fig, ax = generateGraph([y_test.to_list(), list(np.round(y_pred, 2))],
                                    ['Measured Data', 'Predicted Data'],
                                    xlabel='Datapoints', ylabel=ylabel, title=None,
                                    font_size=19)
            st.subheader(f"Machine Learning Model: {name}")
            st.pyplot(fig)
        else:
            st.pyplot(plt)
            # plt.savefig(subsubdir_path + chartname)

        # Evaluate the model using different metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        nrmse = ((rmse) / (max(y_test) - min(y_test))) * 100
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store the results in lists
        results.append([mse, rmse, nrmse, mae, r2])
        model_names.append(name)
        best_parameters.append(best_model.get_params())

    # Create a dataframe with the results
    columns = ["MSE", "RMSE", "NRMSE", "MAE", "R-squared"]
    results_df = pd.DataFrame(results, columns=columns, index=model_names)
    st.dataframe(results_df)
    check_definitions = st.expander("Definitions")
    with check_definitions:
        latex_code_mse = r'''
        #### MSE (Mean Squared Error)

        MSE is calculated as the average of the squared differences between the actual and the predicted values. It is given by the formula:


        $\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2$

        Where:
        - $n$ is the number of observations
        - $Y_i$ is the actual value for observation $i$
        - $\hat{Y}_i$ is the predicted value for observation $i$
        '''
        st.markdown(latex_code_mse)

        latex_code_RMSE = r'''
        #### RMSE (Root Mean Squared Error)

        RMSE is simply the square root of the MSE. It is given by the formula:

        $\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2}$
        '''

        st.markdown(latex_code_RMSE)

        latex_code_nmrse = r'''
        #### NRMSE (Normalized Root Mean Squared Error)

        NRMSE is the RMSE divided by the range of the dependent variable (i.e., the difference between its maximum and minimum values). It is given by the formula:
        
        $\text{NRMSE} = \frac{\text{RMSE}}{Y_{\text{max}} - Y_{\text{min}}}$
        '''
        st.markdown(latex_code_nmrse)

        latex_code_mae = r'''
        #### MAE (Mean Absolute Error)
        MAE is calculated as the average of the absolute differences between the actual and the predicted values. It is given by the formula:
        
        $\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|Y_i - \hat{Y}_i|$

        '''
        st.markdown(latex_code_mae)

        latex_code_rSquared = r'''
        #### R-squared (Coefficient of Determination)

        R-squared is a statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variables. It is given by the formula:
        
        $R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$
        
        Where:
        - $\text{SS}_{\text{res}}$ is the residual sum of squares, given by $\sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2$
        - $\text{SS}_{\text{tot}}$ is the total sum of squares, given by $\sum_{i=1}^{n}(Y_i - \bar{Y})^2$
        - $\bar{Y}$ is the mean value of $Y$

        '''
        st.markdown(latex_code_rSquared)

    return best_parameters, results_df



if __name__ == '__main__':
    # Choose the database in here. e.g.: bbc_nov_2022_1
    st.title("Machine Learning Estimator for Water Quality Data")
    st.markdown("""---""")

    st.sidebar.title("Community Outreach")
    st.sidebar.header("Enter your dataset to be analyzed")
    st.sidebar.file_uploader("Please use .csv files only")
    st.sidebar.markdown("""We are actively working on a feature that will allow you to upload your own datasets in CSV format for analysis. This upcoming functionality aims to provide you with the flexibility to use our analytical tools and get info on your own data, having a more personalized user experience.
        \n- Security and Privacy
        As we develop this feature, our top priority is ensuring the security and privacy of your data. We are implementing security measures to safeguard your information at every step.
        \n- Under Development
        Please note that this feature is currently under development. We are working hard to bring it to you as soon as possible, with all the necessary precautions in place to ensure a secure and user-friendly environment for data upload and analysis.
        \nWe appreciate your patience and encourage you to stay tuned for updates on this exciting new addition to our platform!""")
    partial_ds, entire_ds, ds_name = getDataframe(bbc_oct_2022_1)
    zoom_lat = partial_ds['lat'].mean()
    zoom_long = partial_ds['lon'].mean()
    col1,col2 = st.columns(2)
    with col1:
        st.header("Geospatial Distribution of Observations")
        # print("partial_ds: ", partial_ds.describe())
        st.pydeck_chart(pdk.Deck(
            # map_style https://docs.mapbox.com/api/maps/styles/
            map_style='mapbox://styles/mapbox/satellite-streets-v11',
            initial_view_state=pdk.ViewState(
                latitude=zoom_lat,
                longitude=zoom_long,
                zoom=18,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    # data=partial_ds[['lat','lon']],
                    data=partial_ds,
                    get_position='[lon, lat]',
                    get_color='[26, 255, 0, 160]',
                    get_radius=0.25,
                    pickable=True,  # to show tooltip
                ),
            ],
            tooltip={
                # "html": "Lat: {lat} <br/> Long:{lon}",
                # create 0s for the options when not selected
                "html": "Lat: {lat} <br/> Long:{lon} <br/> ODO(mg/L):{ODO mg/L} <br/>"
                        "Temp(C): {Temperature (c)} <br/> pH: {pH} <br/> TWC(m): {Total Water Column (m)} <br/>",
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white"
                }
            }
        ))
        st.caption("Hover over the markers in the map to check the water data")

    with col2:
        st.header("Temporal Data Visualization")

        def generate_plots(plot_par):
            x = entire_ds["Time hh:mm:ss"]
            y_column = partial_ds[plot_par]
            fig, ax = plt.subplots(figsize=(7, 3))
            plt.title(plot_par)
            ax.plot(x, y_column)
            ax.set_ylim(y_column.min(), y_column.max())
            ax.set_xlabel("Time")
            ax.set_ylabel(plot_par)
            st.pyplot(fig)


        def generate_plot_opt(plot_par):
            # st.line_chart(partial_ds[plot_par])
            import plotly.express as px
            course_fig = px.line(entire_ds, x="Time hh:mm:ss", y=plot_par)
            # course_fig = px.line(entire_ds, x="Time hh:mm:ss", y=plot_par,
            #                      title=plot_par)
            # course_fig.update_traces(marker_color='red')
            course_fig.update_layout(
                font=dict(
                    size=20,
                    color="RebeccaPurple"),
                title_font_color="orange",
                title_font_size=20,
                title="Time Series Visualization of "+plot_par
            )
            course_fig.update_layout({
                "plot_bgcolor": "rgb(232, 233, 234)",
                "paper_bgcolor": "rgba(0, 0, 0,0)",
            })

            st.plotly_chart(course_fig, use_container_width=True)


        def generate_plot_natively(plot_par):
            # entire_ds["Time hh:mm:ss"]
            # partial_ds[plot_par]
            # entire_ds["Time hh:mm:ss"][0]
            # st.text(type(entire_ds["Time hh:mm:ss"][0]))
            df = pd.DataFrame({"Time": entire_ds["Time hh:mm:ss"],
                               plot_par: partial_ds[plot_par]})
            # df=df.set_index("time")
            # st.line_chart(df)
            import altair as alt
            chart = alt.Chart(df).mark_line().encode(
                x=alt.X('Time', axis=alt.Axis(labelOverlap="greedy", grid=False)),
                y=alt.Y(plot_par)
            )
            st.altair_chart(chart, use_container_width=True)

            # st.altair_chart(pd.DataFrame({"index":entire_ds["Time hh:mm:ss"],
            #                               plot_par:partial_ds[plot_par]}))


        plot_parameter = st.selectbox("", ["ODO mg/L", "Temperature (c)", "pH", "Total Water Column (m)"])
        st.write("Dataset collected on: {}".format(entire_ds["Date"][0]))

        if plot_parameter:
            with st.spinner('⌛⌛ Generating plot for desired parameter... '):
                # generate_plots(plot_parameter)
                generate_plot_opt(plot_parameter)
                # st.success('Plot was successful!')
    st.markdown("""---""")
    col3, col4 = st.columns(2)
    with col3:
        st.header("Water Parameters Summary Table")
        options = st.multiselect(
            'Select Desired Parameters',
            ["ODO mg/L", "Temperature (c)", "pH", "Total Water Column (m)"])

        # st.write('You selected:', options[0])

        partial_ds[['lat', 'lon'] + options]

    with col4:
        st.header("Descriptive Statistics")
        st.write("A descriptive statistics table offers a comprehensive overview of the central tendencies, dispersion, and shape of the distribution of a dataset. It typically includes key metrics such as the mean, median, and mode, which provide insight into the central tendency of the data. It also outlines measures of variability such as the standard deviation and range, offering a glimpse into the spread of the data points")
        st.dataframe(partial_ds[["Total Water Column (m)", "Temperature (c)", "pH", "ODO mg/L"]].describe())

    st.markdown("""---""")

    st.header("Results from the Machine Learning Estimators")
    col5,col6=st.columns(2)
    with col5:
        st.subheader("Motivation")
        st.write("Coastal regions globally are experiencing rapid transformations due to climate change and anthropogenic impacts, necessitating robust monitoring systems. Autonomous surface and underwater robots (ASV and AUV) have emerged as important tools for high-resolution, large-scale data collection on water properties. However, these robots often face challenges from faulty or missing sensor data, affecting data accuracy and robot functionality. This paper explores the use of machine learning techniques to estimate water property parameters, addressing the challenges of missing or faulty data. By focusing on Biscayne Bay, Florida, this study uses linear regression, random forest, support vector regression, and multilayer perceptron, to predict parameters like dissolved oxygen, pH, and temperature. Initial results indicate the potential of these models to enhance data consistency and offer new perspectives for sensor fusion approaches.")

        st.image("fishkill.png")
        st.caption("Pictures of the fish kill in South Florida between 2020 and 2022.")

        heatmap_check = st.expander("Heatmap")
        with heatmap_check:
            st.markdown("""
                    This heatmap visually represents the correlations between different water property parameters studied in the Biscayne Bay area. Each cell in the grid shows the correlation coefficient between the parameters on its corresponding row and column. The correlation coefficient values range between -1 and +1, where:

                    - +1 indicates a perfect positive linear relationship: as one parameter increases, the other parameter increases proportionally.
                    - -1 indicates a perfect negative linear relationship: as one parameter increases, the other parameter decreases proportionally.
                    - 0 indicates no linear relationship: changes in one parameter do not predict changes in the other parameter.
                    The color gradient, ranging from blue (negative correlation) to red (positive correlation), facilitates a quick visual assessment of the relationships between parameters. Darker shades of blue or red indicate stronger negative or positive correlations, respectively, while lighter shades or white indicate weaker correlations.

                    By analyzing this heatmap, researchers and policymakers can identify which parameters are strongly related and thus better understand the complex interdependencies between different aspects of water quality in the region. This, in turn, can guide more informed decision-making in environmental monitoring and conservation efforts.
                    """)
            heatmap = generateCorrHeatmap(partial_ds, 'pearson',
                                          ['Latitude', 'Longitude', 'ODO $[ml/l]$', 'Temp $[°C]$', 'pH',
                                           'Water\n Column $[m]$'],
                                          {'annot': True, 'xticklabels': True, 'yticklabels': True, 'cmap': 'coolwarm'})
            fig2 = heatmap.get_figure()

            st.pyplot(fig2)

    with col6:
        st.subheader("Parameter Estimation with Advanced ML Techniques")
        main_form = st.form("main_form")
        with main_form:
            chosenParameter = st.selectbox("Choose a parameter to be estimated",options=
            ['Temperature (c)','pH','ODO mg/L'])

            submit = st.form_submit_button("Submit")
        if submit:
            st.write(f"You chose {chosenParameter} parameter.")

            def run_machine_learning(List_x,List_y,target):
                parameters = [
                        {'fit_intercept': [True, False]},  # Linear Regression
                        {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]},  # Random Forest Regressor
                        {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf','poly'], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto']},  # Support Vector Machines
                        {'hidden_layer_sizes': [(100,), (50, 50), (50, 100, 50)],  # Multi-Layer Perceptron
                         'activation': ['relu'],
                         # 'solver': ['adam', 'lbfgs'],
                         'solver': ['adam'],
                         # 'alpha': [0.0001, 0.001, 0.01]
                         'alpha': [0.001, 0.01]
                         }
                    ]
                parameters = [{'model__' + key: par_dict[key] for key in par_dict.keys()} for par_dict in parameters]

                models = [
                    ("Linear Regression", LinearRegression(), parameters[0]),
                    ("Random Forest Regressor", RandomForestRegressor(random_state=r_st), parameters[1]),
                    ("Support Vector Regressors", SVR(), parameters[2]),
                    ("Multi-Layer Perceptron", MLPRegressor(random_state=r_st), parameters[3]),
                ]
                with st.spinner('🦾🤖 Machine Learning at Work 🚀 Brewing the Perfect Estimations...🧠💡'):
                    best_param, results = all_models(List_x, List_y, models, target)
                # render_animation()
                # st.write(best_param)
                # st.write(results)

            if chosenParameter == 'ODO mg/L':
                X=partial_ds[['Temperature (c)', 'pH', 'Total Water Column (m)']]
                List_y=partial_ds['ODO mg/L']
                target = 'ODO mg/L'
                run_machine_learning(X, List_y, target)
            elif chosenParameter == 'pH':
                List_x = partial_ds[['Temperature (c)', 'ODO mg/L', 'Total Water Column (m)']]
                List_y=partial_ds['pH']
                target = 'pH'
                run_machine_learning(List_x, List_y, target)
            elif chosenParameter == 'Temperature (c)':
                List_x=partial_ds[['ODO mg/L', 'pH', 'Total Water Column (m)']]
                List_y=partial_ds['Temperature (c)']
                target = 'Temperature (c)'
                run_machine_learning(List_x, List_y, target)
