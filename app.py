import numpy as np
from flask import Flask,render_template,url_for,request,redirect,jsonify
import pickle
import os
import nltk
from datetime import datetime
nltk.download('punkt')

app = Flask(__name__)
model=pickle.load(open('spam_classifier.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    # url=request.get_data(as_text=True)[5:]
    text=int(input())
    prediction=model.predict(text)



    return render_template('index.html',prediction_text='Email is {}'.format(prediction))




if __name__=="__main__":
    app.run(debug=True)
