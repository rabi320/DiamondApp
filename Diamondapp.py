import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor

#pipeline tools
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#preprocees
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from PIL import Image

#data
diamond = pd.read_csv("diamonds.csv")
diamond.drop("Unnamed: 0",axis = 1, inplace = True)

X = diamond.drop("price", axis = 1)
y = diamond["price"]

Continuous = X.describe().columns.tolist()

#outlier detector


# this is the main function in which we define our webpage 

# giving the webpage a title
st.title("Diamond Price Estimator 💎")
image = Image.open('diamond.jpg')
st.image(image)
st.write('---')

# here we define some of the front end elements of the web page like 
# the font and background color, the padding and the text to be displayed
html_temp = """
<div style ="background-color:cyan;padding:13px">
<h1 style ="color:black;text-align:center;font-family:Comic Sans MS;">How much is your diamond worth?</h1>
</div>
"""

# this line allows us to display the front end aspects we have 
# defined in the above code
st.markdown(html_temp, unsafe_allow_html = True)

# the following lines create text boxes in which the user can enter 
# the data required to make the prediction

def user_input_features():
    carat = st.sidebar.slider('Carat', 0.2, 5.01)
    cut = st.sidebar.selectbox("Quality of the cut", options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    color = st.sidebar.selectbox("Diamond color, from J (worst) to D (best)", options = ['J', 'I', 'H', 'G', 'F', 'E','D'])
    clarity = st.sidebar.selectbox("Clarity of the diamond (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))", options = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1",'IF'])
    x = st.sidebar.slider('Length', 0.01, 10.74)
    y = st.sidebar.slider('Width', 0.01, 58.9)
    z = st.sidebar.slider('Depth', 0.01, 31.8)
    depth = st.sidebar.slider('total depth (%)', 43, 95,step = 1)
    table = st.sidebar.slider('Width of top of diamond relative to widest point (%)', 43, 95,step = 1)    
    data = {'carat': carat,
            'cut': cut,
            'color': color,
            'clarity': clarity,
            'depth': depth,
            'table': table,
            'x': x,
            'y': y,
            'z': z}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

def Outlier_Detector(X,factor):
    X = pd.DataFrame(X).copy()
    for i in range(X.shape[1]):
        x = pd.Series(X.iloc[:,i]).copy()
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)
        X.iloc[((X.iloc[:,i] < lower_bound) | (X.iloc[:,i] > upper_bound)),i] = np.nan 
    return X

#creating outlier_remover object using FunctionTransformer with factor=3
Outlier = FunctionTransformer(Outlier_Detector,kw_args={'factor':3})

contiuous_transformer = Pipeline(steps=[
('outlier', Outlier),
('imputer', SimpleImputer(strategy='mean')),
('scaler', StandardScaler())
])

# building categorical transformers (worst to best)
cut_enc = OrdinalEncoder(categories=[["Fair", "Good", "Very Good", "Premium","Ideal"]])
color_enc = OrdinalEncoder(categories=[['J', 'I', 'H', 'G', 'F', 'E','D']])
clarity_enc = OrdinalEncoder(categories=[["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1",'IF']])


# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', contiuous_transformer, Continuous),
        ('cuts', cut_enc, ["cut"]),
        ('colors', color_enc, ["color"]),
        ('clarities', clarity_enc, ["clarity"])
    ])

model = ExtraTreesRegressor(n_estimators = 200, max_features = 5,  random_state=42)


# Bundle preprocessing and modeling code in a pipeline
reg = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])


reg.fit(X,y)

  
# the below line ensures that when the button called 'Predict' is clicked, 
# the prediction function defined above is called to make the prediction 
# and store it in the variable result
if st.button("Estimate Diamond Price"):
    result = reg.predict(df)[0]
    st.write(f'The Diamond is worth {result}$')
