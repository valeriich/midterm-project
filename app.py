
from flask import Flask, request, jsonify, render_template



app=Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    #return "<p>Hello, World!</p>"
    return render_template('main.html')

@app.route("/predict", methods=['POST'])
def predict():
    # get data from html page inputs
    values = [i for i in request.form.values()]
    keys = [i for i in request.form.keys()]
    X_dict = {keys[i]: values[i] for i in range(len(keys))}
        
    return render_template('main.html', prediction_text="Predicted demand:\ncasual users {}\nregistered users {}".format(X_dict['season'], X_dict['rolling_mean_12_hours_casual']))

if __name__=="__main__":
    app.run()