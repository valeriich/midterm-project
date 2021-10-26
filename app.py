
from flask import Flask, request, jsonify, render_template



app=Flask(__name__)

@app.route('/')
def hello():
    #return "<p>Hello, World!</p>"
    return render_template('hello.html')


if __name__=="__main__":
    app.run()