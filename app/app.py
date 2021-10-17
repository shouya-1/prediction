from flask import Flask, render_template, request
#importing required libraries & dataset for the project
import pandas as pd
import numpy as np  
import sklearn
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/shortenurl')
def shortenurl():
    glu=request.args['glu']
    bmi=request.args['bmi']
    age=request.args['age']
    dataset_new = pd.read_csv("./data.csv")   #importing files using pandas
    dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)
    # Replacing NaN with mean values
    dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
    dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
    dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
    dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
    dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)
    data1 = pd.DataFrame(dataset_new)
    # Selecting features - [Glucose, Insulin, BMI]
    X = data1.iloc[:, [1, 4, 5]].values
    Y = data1.iloc[:, 8].values
    # Splitting X and Y
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'] )
    KMeans_Clustering = KMeans(n_clusters =2, random_state=0)
    KMeans_Clustering.fit(X_train)
    out = KMeans_Clustering.predict([[glu,bmi,age]])
    if (out==0):
        prediction ="No Diabetes"
    else:
        prediction ="Diabetes"    
    
     
    return render_template('display.html', m1=prediction)
