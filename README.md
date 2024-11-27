## End to End ML Project
# Depression Professional Classification : 

## Table of Content
  <!-- * [Demo](#demo) -->
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


<!-- ## Demo
<!-- Link: [https://ipcc.rohitswami.com](https://ipcc.rohitswami.com) --> 


## Depression Professional Classification 
![Depression](pic.jpg)

<!-- [![](https://imgur.com/s4FWb9b)](https://ipcc.rohitswami.com) -->

## Overview
This repository contains code for training multiple machine learning classifiers on a given dataset to predict mental health outcomes based on demographic, lifestyle, and work-related factors. The models are implemented using the scikit-learn library and include a wide range of supervised learning algorithms suitable for classification tasks.

The following classifiers are applied to the dataset:

-Logistic Regression <br>
-Random Forest Classifier<br>
-Gradient Boosting Classifier<br>
-AdaBoost Classifier<br>
-HistGradient Boosting Classifier<br>
-Support Vector Classifier (SVC)<br>
-Linear Support Vector Classifier (LinearSVC)<br>
-K-Nearest Neighbors Classifier (KNN)<br>
-Gaussian Naive Bayes<br>
-Multinomial Naive Bayes<br>
-Bernoulli Naive Bayes<br>
-Decision Tree Classifier<br>
-Linear Discriminant Analysis (LDA)<br>
-Quadratic Discriminant Analysis (QDA)<br>



-We have trained all of these algorithms and evaluated their performance<br>
-selecting the best-performing models based on classification accuracy.


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
<br>

Building and Hosting a Flask Web App on Render:

A Flask web application is built to interact with the trained models and make real-time predictions based on user input.
The app is deployed using Render to provide easy access via the web.
Users can submit their data via a simple web interface and receive mental health predictions
    - 

## Installation
The Code is written in Python 3.10. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:

# To clone the repository

```bash

git clone gh repo clone Creator-Turbo/-depression-status

```
# Install dependencies: (all lib)
```bash
pip install -r requirements.txt
```



## Run
To train the machine learning models:

1. Prepare your dataset (make sure it’s in the correct format).
>> Data : https://www.kaggle.com/datasets/ikynahidwin/depression-professional-dataset


# To run the Flask web app locally
```bash
python app.py

```
# Deployment on Render

## To deploy the Flask web app on Render:
Push your code to GitHub.<br>
Go to Render and create a new web service.<br>
Connect your GitHub repository to Render.<br>
Set up the environment variables if required (e.g., API keys, database credentials).<br>
Deploy and your app will be live!



## Directory Tree 
```
    Professional\
    ├── data/                     # Dataset files
    ├── notebook/                 # Jupyter notebooks
    ├── static/                   # Static files (images, CSS, JS)
    ├── templates/                # HTML files for Flask         
    ├── .gitignore                # Git ignore file
    ├── app.py                    # Main Flask application
    ├── best_model.pkl            # Trained machine learning model
    ├── pic.jpg                   # Trained machine learning model
    ├── README.md                 # Project documentation
    ├── requirements.txt          # Dependencies for the project
    └── tempCodeRunnerFile.py     # Temporary code file (IDE generated)
```

## To Do

Implement cross-validation for model evaluation.
Enhance hyperparameter tuning for better performance.
Add user authentication for personalized predictions in the Flask app.
Expand the dataset with additional mental health indicators.
Improve web interface design for better user experience. 


## Bug / Feature Request
If you encounter any bugs or want to request a new feature, please open an issue on GitHub. We welcome contributions!




## Technologies Used
Python 3.10  
scikit-learn  
Flask (for web app development)  
Render (for hosting and deployment)  
pandas (for data manipulation)  
numpy (for numerical operations)  
matplotlib (for visualizations)  


![](https://forthebadge.com/images/badges/made-with-python.svg)


[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/260px-Scikit_learn_logo_small.svg.png" width=170>](https://scikit-learn.org/stable/)
[<img target="_blank" src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*RWkQ0Fziw792xa0S" width=170>](https://pandas.pydata.org/docs/)
  [<img target="_blank" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSDzf1RMK1iHKjAswDiqbFB8f3by6mLO89eir-Q4LJioPuq9yOrhvpw2d3Ms1u8NLlzsMQ&usqp=CAU" width=280>](https://matplotlib.org/stable/index.html) 
 [<img target="_blank" src="https://icon2.cleanpng.com/20180829/okc/kisspng-flask-python-web-framework-representational-state-flask-stickker-1713946755581.webp" width=200>](https://flask.palletsprojects.com/en/stable/) 
 [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/512px-NumPy_logo_2020.svg.png" width=200>](https://aws.amazon.com/s3/) 
 <!-- [<img target="_blank" src=https://cdn.worldvectorlogo.com/logos/seaborn-1.svg>](https://seaborn.pydata.org/)  -->




## Team
This project was developed by:

Bablu kumar pandey

<!-- Collaborator Name -->




## Credits

Special thanks to the contributors of the scikit-learn library for their fantastic machine learning tools.

