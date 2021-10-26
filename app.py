
from flask import Flask, request, jsonify, render_template



app=Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    #return "<p>Hello, World!</p>"
    return render_template('main.html')

@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':
        s = int(request.form['season'])
    
    return render_template('main.html', prediction_text="You Can Sell The Car at {}".format(s))

if __name__=="__main__":
    app.run()