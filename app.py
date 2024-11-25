from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd 

# Load the pre-trained model
# with open('best_model.pkl', 'rb') as file:
#     model = pickle.load(file)
with open('D:\\Professional\\notebook\\best_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        gender = request.form['gender']
        age = int(request.form['age'])
        work_pressure = int(request.form['work_pressure'])
        job_satisfaction = int(request.form['job_satisfaction'])
        sleep_duration = request.form['sleep_duration']
        dietary_habits = request.form['dietary_habits']
        suicidal_thoughts = request.form['suicidal_thoughts']
        work_hours = int(request.form['work_hours'])
        financial_stress = int(request.form['financial_stress'])
        family_history = request.form['family_history']

        # Preprocess the input data into a format suitable for the model
        # Example: Encoding categorical variables and other preprocessing steps
        input_data = [
            1 if gender == 'Male' else 0,  # Gender encoding
            age,
            work_pressure,
            job_satisfaction,
            1 if sleep_duration == 'Less than 5 hours' else (2 if sleep_duration == '5-6 hours' else 3),  # Sleep duration encoding
            1 if dietary_habits == 'Healthy' else (2 if dietary_habits == 'Moderate' else 3),  # Dietary habits encoding
            1 if suicidal_thoughts == 'Yes' else 0,  # Suicidal thoughts encoding
            work_hours,
            financial_stress,
            1 if family_history == 'Yes' else 0  # Family history encoding
        ]
        
       # Convert to a pandas DataFrame, ensuring the same column order as the training data
        columns = [
            'Gender', 'Age', 'Work Pressure', 'Job Satisfaction', 'Sleep Duration', 
            'Dietary Habits', 'Suicidal Thoughts', 'Work Hours', 'Financial Stress', 'Family History'
        ]
        
        input_df = pd.DataFrame([input_data], columns=columns)

        # Convert the DataFrame to a 2D numpy array
        input_array = input_df.values

         # Now reshape to ensure it's 2D (1 sample, 10 features)
        input_array = input_array.reshape(1, -1)  # Ensuring shape (1, 10)

        
        # Make prediction
        prediction = model.predict(input_array)

        # Interpret the result
        if prediction[0] == 1:
            result = "Risk of Depression"
        else:
            result = "No Risk of Depression"
        
        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
