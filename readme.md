<center><h1 align="center">MLZoomcamp 2021 midterm project.</h1></center>

<center><h2 align="center">1. Description of the problem</h2></center>
<p>Bicycle is transportation alternative in big cities, and bike rentals services have increased tremendously. Today, there exists great interest in these Bike sharing systems due to their important role in traffic, environmental and health issues. User is able to easily rent a bike from a particular position and return back at another position. The characteristics of data being generated by these systems make them attractive for the research. We can learn from the attributes related to the number of bike rentals how to predict demand. Knowing the demand for bike rentals can help avoid shortages or logistical problems and optimize maintenance and cut down other costs while increasing usage. In this project we try to forecast demand using machine learning. 

<center><h2 align="center">2. Dataset</h2></center>
<a href="https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset"> Dataset used is from UCI Machine Learning Repository . Bike Sharing Dataset contains the hourly and daily count of rental bikes between years 2011 and 2012 in US Capital bikeshare system with the corresponding weather and seasonal information.</p>

<center><h2 align="center">3. How the solution will be used</h2></center>
Instead of wrapping up the data to a daily frequency, we will try to forecast hourly demand, suggesting it will help to manage bike rentals service more precisely and agile. Based on hourly forecast, analyst can build the daily curve of possible demand and react fast on changes. In this project we focus just on creation of some model to predict hourly demand, leaving generation of daily curve as next step of problem solving, which is just consecutive forecasts on the day horizon using information about weather forecast f.e.

Also, secondary goal of the project is to provide explanation of the hourly prediction, so analyst can see the magnitude of the factors influence. Machine learning models interpretability helps to comprehend why certain decisions or predictions have been maid. Interpretability is also important to debug machine learning models and make informed decisions about how to improve them.

<center><h2 align="center">4. Description of the repository</h2></center>

1) `model_development.ipynb`  - notebook that covers:

* Data preparation and data clearning,
* EDA, feature importance analysis,
* Feature engineering,
* Model selection process and parameter tuning.

2) `training.py` - python script to train the final model and save it to a pickle file.

3) `app.py` - python script to run Flask application (loading model, prediction, serving it via a web serice).

4) `Procfile` - specifies the command that is executed by the application on startup. In our case it calls web application to be served by gunicorn.

5) `model_1.pkl`, `model_2.pkl` - pickle files of LightGBM models (for casual and registered users respectively).

6) `hour.csv` - dataset.

7) `requirements.txt` - environment's package list.

8) `templates/main.html` - web-page of the service.

9) `static/` - supplimentary folder for images and CSS files.

10) `readme.md`.

<center><h2 align="center">5. How to run the project</h2></center>
This project was deployed on Heroku platform.
<center><h5 align="center"><a href="https://bike-rentals-demand-prediction.herokuapp.com/">Click here to visit the web-service page</a></h6></center>




