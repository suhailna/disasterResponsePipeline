# Disaster Response Pipeline
## Udacity - Disaster Response Pipeline Project

<img src="https://github.com/suhailna/disasterResponsePipeline/blob/main/disaster_response_pipeline/image1.JPG" alt="My cool logo"/>

### Description:
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life 
disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

### Dependencies:
* Python 3.5+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Model Loading and Saving Library: Pickle
* Web App and Data Visualization: Flask, Plotly

### Installation:
This repository was written in HTML and Python , and requires the Python packages: pandas, numpy, re, pickle, nltk, flask, json, plotly, sklearn, sqlalchemy, sys, warnings.

### File Description:

* process_data.py: This code extracts data from both CSV files: messages.csv (containing message data) and categories.csv (classes of messages) and creates an SQLite database containing a merged and cleaned version of this data.
* train_classifier.py: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
* ETL Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of process_data.py. process_data.py automates this notebook.
* ML Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of train_classifier.py. In particular, it contains the analysis used to tune the ML model and determine which model to use. train_classifier.py automates the model fitting process contained in this notebook.
* disaster_messages.csv, disaster_categories.csv contain sample messages (real messages that were sent during disaster events) and categories datasets in csv format.


### Instructions:
Run the following commands in the project's root directory to set up your database and model.

Run the following commands in the project's root directory to set up your database and model.

* To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
* To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
* Run the following command in the app's directory to run your web app. python run.py
* Go to http://0.0.0.0:3001/

### Screenshots:

<img src="https://github.com/suhailna/disasterResponsePipeline/blob/main/disaster_response_pipeline/image1.JPG" alt="My cool logo"/>
<img src="https://github.com/suhailna/disasterResponsePipeline/blob/main/disaster_response_pipeline/image2.JPG" alt="My cool logo"/>
<img src="https://github.com/suhailna/disasterResponsePipeline/blob/main/disaster_response_pipeline/image3.JPG" alt="My cool logo"/>
<img src="https://github.com/suhailna/disasterResponsePipeline/blob/main/disaster_response_pipeline/image4.JPG" alt="My cool logo"/>
<img src="https://github.com/suhailna/disasterResponsePipeline/blob/main/disaster_response_pipeline/image5.JPG" alt="My cool logo"/>
