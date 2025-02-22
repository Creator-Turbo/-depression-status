from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd 


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
    
        # Get input data from the form
        gender = request.form['Gender']
        age = int(request.form['Age'])
        work_pressure = int(request.form['Work Pressure'])
        job_satisfaction = int(request.form['Job Satisfaction'])
        sleep_duration = request.form['Sleep Duration']
        dietary_habits = request.form['Dietary Habits']
        suicidal_thoughts = request.form['Have you ever had suicidal thoughts ?']
        work_hours = int(request.form['Work Hours'])
        financial_stress = int(request.form['Financial Stress'])
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
        


        # Log input data
        print("Input Data:", input_data)

     

        # Convert to a 2D numpy array
        input_df = np.array(input_data, dtype=float).reshape(1, -1)  # Ensure numeric data
        print(input_df)
        
        
        
        # # Convert input_data into a pandas DataFrame with column names
        input_df = pd.DataFrame(input_df)
        # # input_df=input_df.
        # Handle NaN values if any
        input_df = input_df.fillna(0)  
        print(input_df.head())
        

       
         # Convert to a 2D numpy array
       # Ensure no missing values exist
        if input_df.isnull().values.any():
            raise ValueError("Input data contains NaN values after preprocessing.")
        
        else :
            print("not null values")

        #    Make prediction
        prediction = model.predict(input_df)[0]
    
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

