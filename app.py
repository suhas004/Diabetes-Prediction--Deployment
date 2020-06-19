# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:48:37 2020

@author: Suhas
"""


from flask import Flask,render_template,url_for,request
import pandas as pd 
import os
import pickle

import numpy as np

# Load the Random Forest CLassifier model
model = 'xgboost_random_model.pkl'
classifier = pickle.load(open(model, 'rb'))

app = Flask(__name__)




@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = float(request.form['pregnancies'])
        glu = float(request.form['glucose'])
        bp = float(request.form['bloodpressure'])
        ap = float(request.form['skinthickness'])
        insu = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])
        
        data = np.array([[preg,glu, bp, ap, insu, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
	app.run(debug=True)