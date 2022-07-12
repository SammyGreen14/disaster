# UdacityProject2
Emergency Response Pipeline
Written by Sammy Green
 

## Motivation
This project takes messages received during an emergency and classifies them to increase response speed and accuracy.
 
## Data
Figure Eight data set which includes the initial messages and an english translation. Additionally we have a file with the 36 categories the messages will be categorized into.
 
## Installation
Packages used include sys, nltk, numpy, pandas, pickle, re, sklearn, sqlalchemy
 
## Aditional Notes
na were replaced with 0 as 0 represents the category does not apply to the message

## GitHub Repository
https://github.com/SammyGreen14/disaster
 
## Files
data/disaster_categories.csv: CSV file with the category data
data/disaster_messages.csv: CSV file with the messages data in original language and English
data/process_data.py: Python file that reads in and cleans the data
models/train_classifier.py: Python file that trains the model using an ML Pipeline
models/classifier.pkl: pickle file where the model will be saved to
app/run.py: Python file that runs the process start to finish of ETL, ML Pipeline, and creating the app
app/templates/go.html: HTML file which formats the categories for classification on the app
app/templates/master.html: HTML file which formats the app
