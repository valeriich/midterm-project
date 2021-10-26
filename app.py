import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


app= Flask(__name__, static_url_path='/static')
model_1=pickle.load(open('model_1.pkl', 'rb'))
model_2=pickle.load(open('model_2.pkl', 'rb'))

@app.route('/index')
def predictionpage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    input_values = [float(i) for i in request.form.values()]  #fetching the input values
    #df_row=[[i] for i in input_values]                        #This will form the input row
    #df_keys = [i for i in request.form.keys()]    #fetching the input keys


    #Declaring dictionary to convert into dataframe in the next step.
    #html_dict = {df_keys[i]: df_row[i] for i in range(len(df_keys))}



    #func_dict=creating_input_to_model(html_dict)
    #df=pd.DataFrame(func_dict)
    #df[df.columns[df.columns.isin(rescaling_cols)]] = model.scaler.transform(df[df.columns[df.columns.isin(rescaling_cols)]])

    #Prediction of the trained model
    #prediction= my_model.predict(df)
    #Output derived from the ML model
    #output= round(prediction[0], 2)

    #Output sent to the html page
    return render_template('index.html', prediction_text='Prediction: \n {} cycle rents.'.format(input_values[0]))

if __name__=="__main__":
    app.run(debug=True)