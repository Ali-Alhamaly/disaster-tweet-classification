# Disaster Response Tweet Classification

In this project, I'll analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

## Installation

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.

## Project Motivation

the motivation is to learn new skills in natural language processing (NLP) and also to apply `sklearn' machine learning pipelines.  

## File Descriptions

the 'CSV' files are the data provided by Figure Eight

The two jupyter notebooks contain preparation walkthrough codes to make and ETL pipelines, to process the raw data, and NLP/machine learning pipelines, to train a simple classifier to classify the disaster messages

`process_data.py' is a code that runs from the command line and process data  and then outputs it to SQLite database 

'train_classifier.py' is a code that runs from the command line which creates a classification model to the data and stores the model.

the 'app' folder contains the python code that runs a web app to display some results of the model

## Results

the web app shows the main results of this projec