import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template


#with open('model_1.pkl', 'rb') as f:
#    model_1 = pickle.load(f)

#with open('model_2.pkl', 'rb') as f:
#    model_2 = pickle.load(f)

app=Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    #return "<p>Hello, World!</p>"
    return render_template('main.html')

@app.route("/predict", methods=['POST'])
def predict():
    
    # get data from html page input fields
    values = [i for i in request.form.values()]
    keys = [i for i in request.form.keys()]
    # save data to dictionary
    X_dict = {keys[i]: values[i] for i in range(len(keys))}
    
    # normalize weather data
    X_dict['temp'] += 8
    X_dict['temp'] /= (39 + 8)
    X_dict['windspeed'] /= 67
    X_dict['hum'] /= 100
    
    # creating 'day_type' feature
    ### 2 - working day
    ### 1 - weekend
    ### 0 - holiday
    #if X_dict['holiday'] == 1:
    #    X_dict['day_type'] = 0
    #elif (X_dict['weekday'] == 6) or (X_dict['weekday'] == 0):
    #    X_dict['day_type'] = 1
    #else:
    #    X_dict['day_type'] = 2
    # don't need 'holiday' feature anymore
    #del X_dict['holiday']
    
    # binning hour into 'RegisteredHourBins' feature
    #bins = np.array([1.5, 5.5, 6.5, 8.5, 16.5, 18.5, 20.5, 22.5])
    #labels = np.arange(len(bins)-1)
    #X_dict['RegisteredHourBins'] = pd.cut([X_dict['hr']], bins=bins, labels=labels).fillna(0).astype(int)[0]
    
    # binning hour into 'CasualHourBins' feature
    #bins = np.array([7.5, 8.5, 10.5, 17.5, 19.5, 21.5])
    #labels = np.arange(len(bins)-1)
    #X_dict['CasualHourBins'] = pd.cut([X_dict['hr']], bins=bins, labels=labels).fillna(0).astype(int)[0]
    
    
    
    
    return render_template('main.html', prediction_text="Predicted demand:\ncasual users {}\nregistered users {}".format(X_dict['hum'], X_dict['temp']))

if __name__=="__main__":
    app.run()