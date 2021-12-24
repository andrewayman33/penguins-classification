import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction DSS Project

""")

st.sidebar.header('User Input Features')


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv("C:\\Users\\Andrew\\Desktop\\streamlit_freecodecamp-main\\streamlit_freecodecamp-main\\app_8_classification_penguins\\penguins_cleaned.csv")
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open("C:\\Users\\Andrew\\Desktop\\streamlit_freecodecamp-main\\streamlit_freecodecamp-main\\app_8_classification_penguins\\penguins_clf.pkl", 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

#data visualizations
def load_data():
    data = pd.read_csv("C:\\Users\\Andrew\\Desktop\\streamlit_freecodecamp-main\\streamlit_freecodecamp-main\\app_8_classification_penguins\\penguins_cleaned.csv")
    return data

visuals = load_data()
st.bar_chart(visuals['species'])
st.line_chart(visuals['body_mass_g'])
chartdata=pd.DataFrame(visuals[:],columns=['bill_length_mm','bill_depth_mm','body_mass_g'])
st.area_chart(chartdata)
st.bar_chart(visuals['island'])