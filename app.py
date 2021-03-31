import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

st.markdown("""<style>.main {background-color: #F5F5F5;}
    </style>""", unsafe_allow_html=True)
st.title("Will it rain tomorrow in Australia?")
@st.cache
def get_data():
    # Read and clean the data set
    df = pd.read_csv('weatherAUS.csv')
    df.drop(df[df.RainTomorrow.isna()].index, inplace=True)
    df.Date = pd.to_datetime(df.Date)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop('Date', axis=1, inplace=True)

    # Scaling Features
    numerical = ['Temp3pm', 'Cloud3pm', 'Pressure3pm', 'Humidity3pm', 'WindGustSpeed', 'Sunshine', 'Evaporation', 'Rainfall', 'MaxTemp', 'MinTemp']
    df2 = df[numerical].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df2)
    scaled_features = pd.DataFrame(scaled_features, columns=numerical)

    return df, scaled_features

@st.cache
def get_model_data(df):
    X = df.drop(['RainTomorrow'], axis=1)
    y = df['RainTomorrow']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numerical = [var for var in df.columns if df[var].dtype!='O']

    X_train = X_train.copy()
    X_test = X_test.copy()
    for dset in [X_train, X_test]:
      for col in numerical:
        col_median = X_train[col].median()
        dset.loc[:, col].fillna(col_median, inplace=True)

    for dset in [X_train, X_test]:
      dset['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
      dset['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
      dset['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
      dset['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)

    def max_value(dset, variable, top):
      return np.where(dset[variable]>top, top, dset[variable])

    for dset in [X_train, X_test]:
      dset['Rainfall'] = max_value(dset, 'Rainfall', 5)
      dset['Evaporation'] = max_value(dset, 'Evaporation', 22)
      dset['WindSpeed9am'] = max_value(dset, 'WindSpeed9am', 55)
      dset['WindSpeed3pm'] = max_value(dset, 'WindSpeed3pm', 57)

    encoder = ce.BinaryEncoder(cols=['RainToday'])
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                         pd.get_dummies(X_train.Location),
                         pd.get_dummies(X_train.WindGustDir),
                         pd.get_dummies(X_train.WindDir9am),
                         pd.get_dummies(X_train.WindDir3pm)], axis=1)

    X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                         pd.get_dummies(X_test.Location),
                         pd.get_dummies(X_test.WindGustDir),
                         pd.get_dummies(X_test.WindDir9am),
                         pd.get_dummies(X_test.WindDir3pm)], axis=1)
    cols = X_train.columns
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])

    target_encoder = ce.BinaryEncoder(cols=['RainTomorrow'])
    y_train = target_encoder.fit_transform(y_train)
    y_test = target_encoder.transform(y_test)

    return X_train, X_test, y_train, y_test

# df, scaled_features = get_data()

option = st.sidebar.selectbox("Page navigation", ('Home Page', 'Data set', 'Model'))

if option == 'Home Page':
    st.subheader('People have attempted to predict the weather informally for millennia and formally since the 19th century. The Australian state of New South Wales was recently hit with its worse rain event in three decades after extreme and sustained rain led to major flooding.')
    st.subheader('This webapp investigates a weather data set produced by the Bureau of Meteorology and hosts a Random Forest model which has been trained on the data to predict whether it will rain on a given day. The model is interactive and allows users to adjust the hyperparameters to see how they affect the accuracy.')
    st.markdown('Use the page navigation on the left sidebar to learn about the data set and explore its features or jump straight to the model section.')
    st.markdown('The data set was sourced from Kaggle [here](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package).')

    st.image('https://images.scribblelive.com/2021/3/22/1bc1dccb-dab7-4d6d-8096-8f2aed7866ef.jpg',
             caption='Recent flooding in Kempsey, NSW. Photo: Kevin Weismantel.')

    st.markdown('This webapp was built as a personal practice exercise using the Python library Streamlit which is a '
                'powerful tool for converting Data Science and Machine Learning projects into webapps quickly. You can '
                'visit my [Github page](https://github.com/Toodoi) for the Python sourcecode.')

# if option == 'Model':
#     st.header('Random Forest Model')
#     st.text('I found this dataset on Kaggle.com')
#
#
#     st.header('Time to train the model')
#     st.text('Here you get to choose the hyperparameters of the model and see how performance changes')
#     st.text('Adding more trees and allowing them to use more data will increase the time it takes to generate.')
#     st.text('It may take some time to receive a response when you max these settings.')
#
#     sel_col, disp_col = st.beta_columns(2)
#
#     max_samples = sel_col.slider('Adjust the maximum data points allowed per tree', min_value=5000, max_value=30000, value=5000, step=5000)
#
#     n_estimators = sel_col.slider('Adjust the number of trees in the Random Forest', min_value=1, max_value=20, value=1, step=1)
#
#     X_train, X_test, y_train, y_test = get_model_data(df)
#     rfm = RandomForestRegressor(max_samples=max_samples, n_estimators=n_estimators, max_features=0.5, min_samples_leaf=7)
#
#     rfm.fit(X_train, y_train)
#     preds = rfm.predict(X_test)
#     rf_preds = np.where(preds > 0.5, 1, 0)
#
#     disp_col.subheader('Accuracy of a naive model predicting rain everyday is: 77.1%')
#     score = round(accuracy_score(y_test, rf_preds), 3)*100
#     disp_col.subheader(f'Accuracy of the Random Forest model is: {score}%')
#     disp_col.write()
#
#     graph_preds = np.stack([tree.predict(X_test) for tree in rfm.estimators_])
#
#     fig, ax = plt.subplots()
#     gy = [accuracy_score(y_test, np.where(graph_preds[:i + 1].mean(0) > 0.5, 1, 0)) for i in range(n_estimators)]
#     gx = range(1, n_estimators+1)
#     ax.plot(gx, gy, marker='s')
#     ax.set_xticks(gx)
#     ax.set_xticklabels(gx)
#     plt.xlabel('Number of Trees')
#     plt.ylabel('Model Accuracy (%)')
#     plt.title('Model Accuracy vs Number of Trees in Forest')
#
#     if n_estimators > 1:
#         st.pyplot(fig=fig)
#
#     def rf_feat_importance(m, df):
#         return pd.DataFrame({'Feature': list(df.columns), 'Importance (%)': np.round(m.feature_importances_*100, 1)}
#                             ).sort_values('Importance (%)', ascending=False)
#     fi = rf_feat_importance(rfm, X_test)
#     fi = fi.head(8)
#     # fig = go.Figure(data=go.Table(columnwidth=[300,100], header=dict(values=list(fi.columns),
#     #                 fill_color='#FD8E72',
#     #                 align='center'),
#     #                 cells=dict(values=[fi['Feature'], fi['Importance (%)']], fill_color='#E5ECF6', align='center')))
#     # fig.update_layout(margin=dict(l=5,r=5,b=10,t=10), paper_bgcolor='#F5F5F5')
#     # st.plotly_chart(fig)
#     st.subheader("Feature importance scores tell us which features in the data set are most important in predicting "
#                  "the target feature. In this data set Humidity3pm consistently scores very high.")
#     x = [feature[0] for feature in fi['Feature']]
#     y = list(fi['Importance (%)'])
#     fig = go.Figure(data=[go.Bar(x=x, y=y)])
#     fig.update_layout(title='Feature Importance', yaxis_title_text='Importance (%)', margin=dict(l=5, r=5, b=10, t=30),
#                       paper_bgcolor='#F5F5F5')
#     st.plotly_chart(fig)
#
#
# if option == 'Data set':
#
#     st.subheader('This data set was produced by the Australian Bureau of Meteorology and contains 10 years of daily '
#                  'weather observations from many locations across Australia. The data set has 142,193 records and 24 '
#                  'features excluding the target feature.')
#
#     rand_rows = df.iloc[np.random.permutation(len(df))].head(10)
#     st.subheader('Below are 10 randomly generated rows from the data set. Don\'t forget to scroll across to see more features.'
#                  ' You can also press the button to generate a new set.')
#     if st.button('Generate New Rows'):
#         rand_rows = df.iloc[np.random.permutation(len(df))].head(10)
#     st.write(rand_rows)
#
#     st.subheader("The target feature in the data set is 'RainTomorrow' which has binary yes/no values. We are using the "
#                  "other 24 features which refer to daily numerical and categorical measurements to attempt to predict whether it "
#                  "will rain the next day. The chart below shows the target variable is somewhat class imbalanced with 77.1% "
#                  "of 'no' values and 22.1% 'yes'.")
#
#     # RainTomorrow class distribution bar graph
#     x = list(df.RainTomorrow.value_counts().index)
#     y = df.RainTomorrow.value_counts()
#     fig = go.Figure(data=[go.Bar(x=x, y=y, hovertext=[f'{round(y[0]/(y[0]+y[1])*100),1}%', f'{round(y[1]/(y[0]+y[1])*100),1}%'],
#                                  marker=dict(color=['#e33030', '#7fbf67']))])
#     fig.update_layout(margin=dict(l=5, r=5, b=10, t=10), paper_bgcolor='#F5F5F5')
#     st.plotly_chart(fig)
#
#     st.subheader('The boxplots below show the distribution for some of the numerical features. I have standardised these features'
#                  ' with mean 0 and standard deviation of 1 to check if there are any outliers. You can see that Rainfall '
#                  'and Evaporation are strongly positively skewed with many outliers.')
#
#     st.image('Boxplots.png') ## The code created this png
#     # # Box plots for the scaled features
#     # N = len(scaled_features.columns)
#     # c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, N)]
#     # fig = go.Figure()
#     # for i, feature in enumerate(scaled_features):
#     #     fig.add_trace(go.Box(x=scaled_features[feature], name=feature, boxpoints='suspectedoutliers', marker_color=c[i],
#     #                          jitter=0.3, marker_size=1.5))
#     # fig.update_layout(title='Boxplots of Scaled Numerical Features', margin=dict(l=5, r=5, b=10, t=35),
#     #     paper_bgcolor='#F5F5F5', showlegend=False
#     # )
#     # st.plotly_chart(fig)
#
#     st.subheader('For this exercise I am using a Random Forest model for predictions. The Random Forest is an ensemble of decision trees'
#                  ' which are robust to outliers. If I was using a regression-based model it would be important to try to transform '
#                  'the distribution of these features to be more normal. I decided to transform the features anyway however '
#                  'it did not have a great impact on the model\'s accuracy.')
#
#     st.subheader('Taking a closer look at the Evaporation feature using a histogram, we can see again that it is highly '
#                  'skewed to the right. Segmenting by the target variable RainTomorrow, it looks like the lower the '
#                  'evaporation is for a given day, the less likely it is to rain on the next day.')
#
#     st.image('EvapHist.png') ## The code created this png
#     # # Evaporation histogram
#     # df3 = df[df.Evaporation < 100]
#     # fig = px.histogram(df3, x="Evaporation", color="RainTomorrow",
#     #                    marginal="box",  # or violin, rug
#     #                    hover_data=df.columns)
#     # fig.update_layout(xaxis_title_text='Daily Evaporation (mm)', yaxis_title_text='Count',
#     #                   margin=dict(l=5, r=25, b=5, t=10), paper_bgcolor='#F5F5F5')
#     # st.plotly_chart(fig)
#
#     st.subheader('The last chart we will look at is a histogram of Humidity levels recorded at 3pm. I chose this data '
#                  'in retrospect because as you will see, it ends up being the most important feature to the Random '
#                  'Forest for predicting the target feature. As we can see, the histograms segmented by the target '
#                  'feature below have fairly different distributions which may be why the feature has so much predictive power.')
#
#     st.image('HumidHist.png') ## The code created this png
#     # # Humidity3pm Histogram
#     # H3pm1 = df[df.RainTomorrow == 'Yes'].Humidity3pm
#     # H3pm2 = df[df.RainTomorrow == 'No'].Humidity3pm
#     # fig = go.Figure()
#     # fig.add_trace(go.Histogram(x=H3pm1, name='Rain Tomorrow'))
#     # fig.add_trace(go.Histogram(x=H3pm2, name='No Rain Tomorrow'))
#     #  # Overlay both histograms
#     # fig.update_layout(barmode='overlay', paper_bgcolor='#F5F5F5', margin=dict(l=5, r=5, b=10, t=35),
#     #                   xaxis_title_text='Humidity at 3pm', yaxis_title_text='Count')
#     # # Reduce opacity to see both histograms
#     # fig.update_traces(opacity=0.75)
#     # st.plotly_chart(fig)
#
#     st.subheader("Now feel free to visit the 'Model' page to see how accurately a Random Forest can predict "
#                  "next day rain.")




