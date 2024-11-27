## End to End ML Project
# Depression Professional Classification : 

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Run](#run)
  * [Deployement on render](#deployement-on-render)
  * [Directory Tree](#directory-tree)
  * [To Do](#to-do)
  * [Bug / Feature Request](#bug---feature-request)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [License](#license)
  * [Credits](#credits)


## Demo
Link: [https://ipcc.rohitswami.com](https://ipcc.rohitswami.com)

[![](https://i.imgur.com/5gj4USj.png)](https://ipcc.rohitswami.com)

## Overview
This repository contains code for training multiple machine learning classifiers on a given dataset to predict mental health outcomes based on demographic, lifestyle, and work-related factors. The models are implemented using the scikit-learn library and include a wide range of supervised learning algorithms suitable for classification tasks.

The following classifiers are applied to the dataset:

-Logistic Regression
-Random Forest Classifier
-Gradient Boosting Classifier
-AdaBoost Classifier
-HistGradient Boosting Classifier
-Support Vector Classifier (SVC)
-Linear Support Vector Classifier (LinearSVC)
-K-Nearest Neighbors Classifier (KNN)
-Gaussian Naive Bayes
-Multinomial Naive Bayes
-Bernoulli Naive Bayes
-Decision Tree Classifier
-Linear Discriminant Analysis (LDA)
-Quadratic Discriminant Analysis (QDA)
-We have trained all of these algorithms and evaluated their performance, -selecting the best-performing models based on classification accuracy.


## Motivation
Mental health has become an increasingly critical issue globally, affecting individuals' overall well-being and productivity. Factors such as work pressure, lifestyle choices, and demographic variables are known to have a significant impact on mental health outcomes. With the growing importance of understanding these factors, this dataset was created to analyze the relationship between various lifestyle and work-related aspects and mental health conditions like depression and suicidal thoughts.

By leveraging machine learning, this dataset aims to:

Identify potential risk factors for mental health issues like depression, anxiety, and suicidal tendencies.
Understand how work-life balance, financial stress, and job satisfaction impact mental well-being.
Predict mental health outcomes based on demographic and lifestyle data to provide targeted interventions for high-risk individuals.
The ultimate goal of this project is to improve mental health support strategies, inform workplace mental health policies, and provide insights into better supporting individuals who may be struggling with mental health issues.

## Technical Aspect

This project is divided into two major parts:

Training Machine Learning Models:

We train multiple machine learning algorithms on the mental health dataset.
All models are implemented using scikit-learn, a Python library for machine learning.
Evaluation is performed using performance metrics such as accuracy, precision, recall, and F1-score.
Building and Hosting a Flask Web App on Render:

A Flask web application is built to interact with the trained models and make real-time predictions based on user input.
The app is deployed using Render to provide easy access via the web.
Users can submit their data via a simple web interface and receive mental health predictions
    - 

## Installation
The Code is written in Python 3.10. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```



```


## Deployement on render
Set the environment variable on Heroku as mentioned in _STEP 1_ in the __Run__ section. [[Reference](https://devcenter.heroku.com/articles/config-vars)]

![](https://i.imgur.com/TmSNhYG.png)

Our next step would be to follow the instruction given on [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.

## Directory Tree 
```Professional\
    ├── data/                     # Dataset files
    ├── notebook/                 # Jupyter notebooks
    ├── static/                   # Static files (images, CSS, JS)
    ├── templates/                # HTML files for Flask
    ├── venv1/                    # Virtual environment
    ├── .gitignore                # Git ignore file
    ├── app.py                    # Main Flask application
    ├── best_model.pkl            # Trained machine learning model
    ├── README.md                 # Project documentation
    ├── requirements.txt          # Dependencies for the project
    └── tempCodeRunnerFile.py     # Temporary code file (IDE generated)
```

## To Do
1. Convert the app to run without any internet connection, i.e. __PWA__.
2. Add a better vizualization chart to display the classification.

## Bug / Feature Request
If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/rowhitswami/Indian-Paper-Currency-Classification/issues/new) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/rowhitswami/Indian-Paper-Currency-Classification/issues/new). Please include sample queries and their corresponding results.

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://keras.io/img/logo.png" width=200>](https://keras.io/) [<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=170>](https://flask.palletsprojects.com/en/1.1.x/) [<img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" width=280>](https://gunicorn.org) [<img target="_blank" src="https://www.kindpng.com/picc/b/301/3012484.png" width=200>](https://aws.amazon.com/s3/) 

[<img target="_blank" src="https://sentry-brand.storage.googleapis.com/sentry-logo-black.png" width=270>](https://www.sentry.io/) [<img target="_blank" src="https://openjsf.org/wp-content/uploads/sites/84/2019/10/jquery-logo-vertical_large_square.png" width=100>](https://jquery.com/)

<!-- ## Team
[![Rohit Swami](https://avatars1.githubusercontent.com/u/16516296?v=3&s=144)](https://rohitswami.com/) |
-|
[Rohit Swami](https://rohitswami.com/) |)

## License
[![Apache license](https://img.shields.io/badge/license-apache-blue?style=for-the-badge&logo=appveyor)](http://www.apache.org/licenses/LICENSE-2.0e)

Copyright 2020 Rohit Swami

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Credits
- [Google Images Download](https://github.com/hardikvasa/google-images-download) - This project wouldn't have been possible without this tool. It saved my enormous amount of time while collecting the data. A huge shout-out to its creator [Hardik Vasa](https://github.com/hardikvasa). -->
