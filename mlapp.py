

import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('6 class csv.csv')
pickled_model = pickle.load(open('model.pkl', 'rb'))
map={'Blue White':'Blue-white','Blue-White':'Blue-white','Yellowish-white':'yellow-white','Yellowish White':'yellow-white','White':'white','yellowish':'Yellowish','Blue white':'Blue-white',}
df.replace({'Star color':map},inplace=True)
numericolumns=df.drop(columns=['Star type','Star color','Spectral Class'])

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numericolumns=list(numericolumns)
catcolumns=list((set(df.columns)-set(numericolumns)))
catcolumns.remove('Star type')

def transform(dataframe1):
  modelencoder=OneHotEncoder()
  modelscaler=MinMaxScaler()
  full_pipe=ColumnTransformer([('num',modelscaler,numericolumns),('catcolumns',modelencoder,catcolumns)])
  if 'Star type' in dataframe1:
    return (full_pipe.fit_transform(dataframe1.drop(columns='Star type')),full_pipe)
  else:
    return (full_pipe.fit_transform(dataframe1),full_pipe)

def labelsfunc(dataframe1):
  return pd.get_dummies(df['Star type'])

dfsample=pd.DataFrame(columns=df.columns)
dfsample.drop(columns='Star type',inplace=True)

def convertpredtostring(pred):
  map={0:"RED DWARF",1:"BROWN DWARF",2:"WHITE DWARF",3:"MAIN SEQUENCE",4:"SUPERGIANT",5:"HYPERGIANT"}
  return map[pred]
arr=np.array([9940,25.4,1.711,11.18,'white','A'])#user input

def predict(arr):

    dfsample.loc[0]=arr
    a=transform(df)[1]
    val=a.transform(dfsample)
    g=pickled_model.predict(val).argmax()
    return convertpredtostring(g)


from flask import Flask,render_template,request
 
app = Flask(__name__)
entries=[]
@app.route('/')
def initial():
    return render_template('machineform2.html')

@app.route('/pred',methods=["GET","POST"])

def defaullt():
    
    
    if request.method=="POST":
        entries=[x for x in request.form.values()]
        ans=predict(entries)
        
        
    return render_template('machineform2.html',prediction=ans)

if __name__ == '__main__':
 

    app.run()