from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd 

# Load the pre-trained model
# with open('best_model.pkl', 'rb') as file:
#     model = pickle.load(file)
with open('D:\\Professional\\notebook\\best_model.pkl', 'rb') as file:
    model = pickle.load(file)

print(model.feature_names_in_)

app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# # Route to handle prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input data from the form
#         gender = request.form['gender']
#         age = int(request.form['age'])
#         work_pressure = int(request.form['work_pressure'])
#         job_satisfaction = int(request.form['job_satisfaction'])
#         sleep_duration = request.form['sleep_duration']
#         dietary_habits = request.form['dietary_habits']
#         suicidal_thoughts = request.form['Have you ever had suicidal thoughts ?']
#         work_hours = int(request.form['work_hours'])
#         financial_stress = int(request.form['financial_stress'])
#         family_history = request.form['Family History of Mental Illness']

#         # Preprocess the input data into a format suitable for the model
#         # Example: Encoding categorical variables and other preprocessing steps
#         input_data = [
#             1 if gender == 'Male' else 0,  # Gender encoding
#             age,
#             work_pressure,
#             job_satisfaction,
#             1 if sleep_duration == 'Less than 5 hours' else (2 if sleep_duration == '5-6 hours' else 3),  # Sleep duration encoding
#             1 if dietary_habits == 'Healthy' else (2 if dietary_habits == 'Moderate' else 3),  # Dietary habits encoding
#             1 if suicidal_thoughts == 'Yes' else 0,  # Suicidal thoughts encoding
#             work_hours,
#             financial_stress,
#             1 if family_history == 'Yes' else 0  # Family history encoding
#         ]
        
#        # Convert to a pandas DataFrame, ensuring the same column order as the training data
#         columns = [
#             'Gender', 'Age', 'Work Pressure', 'Job Satisfaction', 'Sleep Duration', 
#             'Dietary Habits', 'Suicidal Thoughts', 'Work Hours', 'Financial Stress', 'Family History'
#         ]
        
#         for field in columns:
#           if field not in request.form:
#             return f"Error: Missing {field} field in the form.", 400

# # Convert input_data into a pandas DataFrame with column names
#         input_df = pd.DataFrame([input_data], columns=columns)
        
#         # Now input_df is a DataFrame, which can be passed to the model for prediction
        
#         # Make prediction
#         prediction = model.predict(input_df)

#         # Interpret the result
#         if prediction[0] == 1:
#             result = "Risk of Depression"
#         else:
#             result = "No Risk of Depression"
        
#         return render_template('index.html', prediction=result)

#     except Exception as e:
#         return f"Error occurred: {e}"



# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define the required fields
        required_fields = [
            'gender', 'age', 'work_pressure', 'job_satisfaction', 'sleep_duration', 
            'dietary_habits', 'Have you ever had suicidal thoughts ?', 'work_hours', 
            'financial_stress', 'Family History of Mental Illness'
        ]

        # Check if any required field is missing
        for field in required_fields:
            if field not in request.form:
                return f"Error: Missing {field} field in the form.", 400

        # Get input data from the form
        gender = request.form['gender']
        age = int(request.form['age'])
        work_pressure = int(request.form['work_pressure'])
        job_satisfaction = int(request.form['job_satisfaction'])
        sleep_duration = request.form['sleep_duration']
        dietary_habits = request.form['dietary_habits']
        suicidal_thoughts = request.form['Have you ever had suicidal thoughts ?']
        work_hours = int(request.form['work_hours'])
        financial_stress = int(request.form['financial_stress'])
        family_history = request.form['Family History of Mental Illness']

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
        print(input_df.shape)
       
        input_df = input_df.fillna(0, inplace=True)

        
        # Make prediction
        prediction = model.predict(input_df)

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

