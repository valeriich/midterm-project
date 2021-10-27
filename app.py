
from flask import Flask, request, jsonify, render_template



app=Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    #return "<p>Hello, World!</p>"
    return render_template('main.html')

@app.route("/predict", methods=['POST'])
def predict():

    input_values = [float(i) for i in request.form.values()]
        
    return render_template('main.html', prediction_text="Predicted demand:\ncasual users {}\nregistered users {}".format(input_values[0], input_values[1]))

if __name__=="__main__":
    app.run()