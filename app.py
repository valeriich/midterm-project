
from flask import Flask, request, jsonify, render_template



app=Flask(__name__)

@app.route('/')
def index():
    #return "<p>Hello, World!</p>"
    return render_template('main.html')


if __name__=="__main__":
    app.run()