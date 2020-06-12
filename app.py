#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 07:55:42 2020

@author: vatsal
"""



from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open("LE.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
poly = pickle.load(open('Poly.pkl', "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    int_features = [np.array(int_features)]
    int_features = scaler.transform(int_features)
    int_features = poly.transform(int_features)
    prediction = model.predict(int_features)
    prediction = abs(prediction)
    
    output = round(prediction[0], 2)

    return render_template('index.html', 
                           prediction_text='Approximately upvote for question will be: {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
