import numpy as np
import pandas as pd
import lightgbm
from flask import Flask, request, jsonify, render_template
import pickle


with open('model_1.pkl', 'rb') as f:
    model_1 = pickle.load(f)

with open('model_2.pkl', 'rb') as f:
    model_2 = pickle.load(f)

app=Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    #return "<p>Hello, World!</p>"
    return render_template('main.html')

@app.route("/predict", methods=['POST'])
def predict():
    
    # get data from html page input fields
    values = [int(i) for i in request.form.values()]
    keys = [i for i in request.form.keys()]
    # save data to dictionary
    X_dict = {keys[i]: values[i] for i in range(len(keys))}
    
    X_1, X_2 = prepare_data(X_dict)
    
    casual = int(np.around(model_1.predict(X_1).clip(0), 0)[0])
    registered = int(np.around(model_2.predict(X_2).clip(0), 0)[0])
    
    total = casual + registered
    
    return render_template('main.html', prediction_text="Predicted demand for bicycles at this hour of the day: {} bikes".format(total))

if __name__=="__main__":
    app.run()