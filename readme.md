<center><h1 align="center">Boom-Bikes Bike Rentals.</h1></center>

<p>A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state. They also want to expand their business outside US. Hence they want their potential partners to forecast the number of bikes rented by customers in various conditions.</p>

<center><h1 align="center">Business Goal.</h1></center>
<p>The objective of this project is to model the demand for shared bikes with the available independent variables. It will be used by the management and potential partners to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market. </p>

<center><h5 align="center"><a href="https://boombikes-prediction-api.herokuapp.com/">Click here to visit the home page</a></h6></center>
<center><h5 align="center"><a href="https://boombikes-prediction-api.herokuapp.com/index">Click here to visit the prediction page directly</a></h6></center>

<center><h3 align="center">Below is the preview of the application home-page.</h3></center>
<div align="center">
    <img src="/static/img/homepage.jpg" width="400px"/>
</div>

<center><h3 align="center">Below is the preview of the prediction page</h3></center>
<div align="center">
	<img src="/static/img/prediction.jpg" width="400px"/>
</div>


<center><h1 align="center">Components of the project.</h1></center>
<h3>Project is divided into 2 parts:</h3>
<h4>Part 1: Building Machine Learning Model and Prediction.</h4>
<ol type="1">
<li> Data: day.csv</li>
<li><a href="https://github.com/ds-souvik/Prediction-of-Bike-Rental-Count-Linear-Regression-and-Deployment-along-with-deployment/blob/master/BoomBikes%20Prediction%20model.ipynb">Boombikes Prediction model.ipynb:</a>The main ML notebook. It consists of the following tasks:
	<ul>
	<li>Data understanding and Data cleaning.</li>
	<li>Detailed EDA and inferences in every step.</li>
	<li>Data Preparation for modelling.</li>
	<li>Training the model.</li>
	<li>Model Evaluation: p-value, VIF, r-squared, adjusted r squared, residual analysis.</li>
	<li>Hyper parameter tuning.</li>
	<li>Prediction and inferences.</li></ul>
</li>
</ol>

<h4>Part 2: Deployment.</h4>
<ol type="1">
<li><a href="https://github.com/ds-souvik/Prediction-of-Bike-Rental-Count-Linear-Regression-and-Deployment-along-with-deployment/blob/master/model.py">model.py</a>: It mostly is the subpart of Boombikes Prediction model.ipynb having the code where model is built and trained. At the end, the trained model is saved in the memory of a pickle package- model.pkl.</li>
<li><a href="https://github.com/ds-souvik/Prediction-of-Bike-Rental-Count-Linear-Regression-and-Deployment-along-with-deployment/blob/master/app.py">app.py</a>: It contains the code of my flask app. Flask is a micro web framework written in Python.</li>
Basic flow of the app: 
	<ol type="1"></ol>
	<li>A flask object is instantiated: An object of Flask class is our WSGI application. Flask constructor takes the name of current module (__name__) as argument.<br>
	app= Flask(__name__, static_url_path='/static') enambles us to use the static folder while building the front-end.</li>
	<li>A pickle object is created named my_model.</li>
	<li>App routing: App routing is used to map the specific URL with the associated function that is intended to perform some task. @app.route('/') is used to access apply function home() in it. It executes the frontend using <a href="https://github.com/ds-souvik/Prediction-of-Bike-Rental-Count-Linear-Regression-and-Deployment-along-with-deployment/blob/master/templates/home.html">home.html</a> which inside it has an href to execute 'index' on click to use funtion predictionpage() from the app. Function predictionpage() opens our prediction page <a href="https://github.com/ds-souvik/Prediction-of-Bike-Rental-Count-Linear-Regression-and-Deployment-along-with-deployment/blob/master/templates/index.html">index.html</a>. Clicking on predict button after giving correct inputs to the fields executes the function predict() which fetches data from the front-end, scales the required fields, performs prediction using the my_model object, formats the output and sends the predicted text to the front-end. </li>
</ol>

### Type of Cloud service used for this project is Platform As A Service(PAAS)- Heroku.
<a href="https://github.com/ds-souvik/Prediction-of-Bike-Rental-Count-Linear-Regression-and-Deployment-along-with-deployment/blob/master/Heroku%20deployment.docx">Click here</a> to read the document where I have shown the steps I followed for deployment.

## Technologies and IDEs Used.

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=170>](https://flask.palletsprojects.com/en/1.1.x/) [<img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" width=280>](https://gunicorn.org) [<img src="https://upload.wikimedia.org/wikipedia/en/a/a9/Heroku_logo.png" width="180" height="73">](https://en.wikipedia.org/wiki/Heroku) [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/HTML5_logo_and_wordmark.svg/120px-HTML5_logo_and_wordmark.svg.png" width="100" height="100">](https://en.wikipedia.org/wiki/HTML) [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/CSS3_logo_and_wordmark.svg/120px-CSS3_logo_and_wordmark.svg.png" width="75" height="100">](https://en.wikipedia.org/wiki/CSS)

[<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/300px-Pandas_logo.svg.png" width="180">](https://en.wikipedia.org/wiki/Pandas_(software)) [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/220px-NumPy_logo_2020.svg.png" width="180">](https://en.wikipedia.org/wiki/NumPy) [<img src="https://upload.wikimedia.org/wikipedia/en/thumb/5/56/Matplotlib_logo.svg/300px-Matplotlib_logo.svg.png" width="180" height="80">](https://en.wikipedia.org/wiki/Matplotlib) [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/220px-Scikit_learn_logo_small.svg.png">](https://en.wikipedia.org/wiki/Scikit-learn)

[<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/250px-Jupyter_logo.svg.png" width="88" height="100">](https://en.wikipedia.org/wiki/Project_Jupyter) &nbsp; &nbsp; [<img src="https://upload.wikimedia.org/wikipedia/en/thumb/d/d2/Sublime_Text_3_logo.png/150px-Sublime_Text_3_logo.png" width="100" height="100">](https://en.wikipedia.org/wiki/Sublime_Text) &nbsp; &nbsp; [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/PyCharm_Logo.svg/64px-PyCharm_Logo.svg.png" width="100" height="100">](https://en.wikipedia.org/wiki/PyCharm)

## Machine learning Engineer and Developer
[<img target="_blank" src="https://github.com/ds-souvik/Prediction-of-Bike-Rental-Count-Linear-Regression-and-Deployment-along-with-deployment/blob/master/static/img/me2.jpg" width=140 height=160>](https://www.linkedin.com/in/souvik-ganguly-4a9924105/)|
-|
[Souvik Ganguly](https://www.linkedin.com/in/souvik-ganguly-4a9924105/) |)





