from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd 

# Load the pre-trained model
# with open('best_model.pkl', 'rb') as file:
#     model = pickle.load(file)
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

print(model.feature_names_in_)

app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define the required fields
        required_fields = [
             'Gender', 'Age', 'Work Pressure', 'Job Satisfaction', 'Sleep Duration', 
            'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Work Hours', 'Financial Stress', 'Family History of Mental Illness'
        ]


        # Get input data from the form
        gender = [request.form['Gender']]
        age = [int(request.form['Age'])]
        work_pressure = [int(request.form['Work Pressure'])]
        job_satisfaction = [int(request.form['Job Satisfaction'])]
        sleep_duration = [request.form['Sleep Duration']]
        dietary_habits = [request.form['Dietary Habits']]
        suicidal_thoughts = [request.form['Have you ever had suicidal thoughts ?']]
        work_hours = [int(request.form['Work Hours'])]
        financial_stress = [int(request.form['Financial Stress'])]
        family_history = [request.form['Family History of Mental Illness']]

        # Preprocess the input data into a format suitable for the model
        input_data = [
            1 if gender == 'Male' else 0,  # Gender encoding
            age,
            work_pressure,
            job_satisfaction,
            1 if sleep_duration == 'Less than 5 hours' else (2 if sleep_duration == '5-6 hours' else 3),  # Sleep encoding
            1 if dietary_habits == 'Healthy' else (2 if dietary_habits == 'Moderate' else 3),  # Dietary encoding
            1 if suicidal_thoughts == 'Yes' else 0,  # Suicidal thoughts encoding
            work_hours,
            financial_stress,
            1 if family_history == 'Yes' else 0  # Family history encoding
        ]
        
        # Convert to a pandas DataFrame, ensuring the same column order as the training data
        columns = [
            'Gender', 'Age', 'Work Pressure', 'Job Satisfaction', 'Sleep Duration', 
            'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Work Hours', 'Financial Stress', 'Family History of Mental Illness'
        ]
        
        # Convert input_data into a pandas DataFrame with column names
        input_df = pd.DataFrame([input_data], columns=columns)
        
        input_df = input_df.apply(pd.to_numeric, errors='coerce')
        input_df = input_df.fillna(0, inplace=True)  # Replace NaNs with 0
        # print(input_df.shape)
       
       
        print(input_df)
        print(model)
        # Make prediction
        prediction = model.predict(input_df)
    
        print(prediction)

        # Interpret the result
        if prediction[0] == 1:
            result = "Risk of Depression"
        else:
            result = "No Risk of Depression"
        
        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"Error occurred: {e}"



if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)

