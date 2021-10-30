import numpy as np
import pandas as pd
import lightgbm
from flask import Flask, request, jsonify, render_template
import matplotlib
import matplotlib.pyplot as plt
import shap
import pickle

# separate feature spaces for each model (casual and registered users)
features1 = ['temp', 'hum', 'windspeed', 'hr', '3_days_sum_casual',
            'rolling_mean_12_hours_casual','season', 'yr', 'mnth',  
            'day_type', 'weathersit', 'CasualHourBins', 'weekday']

features2 = ['temp', 'hum', 'windspeed', 'hr', '3_days_sum_registered',
            'rolling_mean_12_hours_registered', 'season', 'yr', 'mnth',  
            'day_type', 'weathersit', 'RegisteredHourBins', 'weekday']

plot1 = 'static/plot1.png'
plot2 = 'static/plot2.png'

with open('model_1.pkl', 'rb') as f:
    model_1 = pickle.load(f)

with open('model_2.pkl', 'rb') as f:
    model_2 = pickle.load(f)

app=Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    return render_template('main.html')

@app.route("/predict", methods=['POST'])
def predict():
    
    # get data from html page input fields
    values = [int(i) for i in request.form.values()]
    keys = [i for i in request.form.keys()]
    # save data of the sample to dictionary
    X = {keys[i]: values[i] for i in range(len(keys))}
    
    # normalize weather data
    X['temp'] += 8
    X['temp'] /= (39 + 8)
    X['windspeed'] /= 67
    X['hum'] /= 100
    
    # creating 'day_type' feature
    ### 2 - working day
    ### 1 - weekend
    ### 0 - holiday
    if X['holiday'] == 1:
        X['day_type'] = 0
    elif (X['weekday'] == 6) or (X['weekday'] == 0):
        X['day_type'] = 1
    else:
        X['day_type'] = 2
    # don't need 'holiday' feature anymore
    del X['holiday']
    
    # binning hour into 'RegisteredHourBins' feature
    bins = np.array([1.5, 5.5, 6.5, 8.5, 16.5, 18.5, 20.5, 22.5])
    labels = np.arange(len(bins)-1)
    label = pd.cut([X['hr']], bins=bins, labels=labels).fillna(0).astype(int)[0]
    remap_labels = {0: 0, 1: 1, 2: 5, 3: 3, 4: 6, 5: 4, 6: 2}
    X['RegisteredHourBins'] = remap_labels[label]
    
    # binning hour into 'CasualHourBins' feature
    bins = np.array([7.5, 8.5, 10.5, 17.5, 19.5, 21.5])
    labels = np.arange(len(bins)-1)
    label = pd.cut([X['hr']], bins=bins, labels=labels).fillna(0).astype(int)[0]
    remap_labels = {0: 0, 1: 2, 2: 4, 3: 3, 4: 1}
    X['CasualHourBins'] = remap_labels[label]
    
    # predicting future, so year is 1
    X['yr'] = 1
    
    # constructing vectors of the sample
    X_1 = np.array([X[feature] for feature in features1]).reshape(1, -1)
    X_2 = np.array([X[feature] for feature in features2]).reshape(1, -1)
    
    casual = int(np.around(model_1.predict(X_1).clip(0), 0)[0])
    registered = int(np.around(model_2.predict(X_2).clip(0), 0)[0])
    
    total = casual + registered
    
    # explain the model's predictions using SHAP
    explainer = shap.Explainer(model_1, feature_names=features1)
    shap_values = explainer(X_1)

    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0], max_display=14, show=False)
    plt.title('Explanation for casual users prediction')
    plt.savefig(plot1, format = "svg",dpi = 300, bbox_inches = 'tight')
    plt.clf()
    
    # explain the model's predictions using SHAP
    explainer = shap.Explainer(model_2, feature_names=features2)
    shap_values = explainer(X_2)

    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0], max_display=14, show=False)
    plt.title('Explanation for registered users prediction')
    plt.savefig(plot2, format = "svg",dpi = 300, bbox_inches = 'tight')
    plt.clf()
    
    return render_template('main.html', prediction_text="Predicted demand for bicycles at this hour of the day: {} bikes".format(total), url_plot1=plot1, url_plot2=plot2)

if __name__=="__main__":
    app.run(debug=True)