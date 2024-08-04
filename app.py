from flask import Flask, request, render_template
import pickle
import pandas as pd
import random

# Load the trained model and preprocessor
with open('model_for_G3.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    form_data = request.form.to_dict()
    print("From Data:", form_data) # Debug message

    # Convert form data to DataFrame
    input_data = pd.DataFrame([form_data])
    
    # Ensure columns are in the correct order and convert to appropriate types
    columns = [
        'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 
        'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 
        'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2'
    ]
    
    # Create empty DataFrame with all required columns
    empty_df = pd.DataFrame(columns=columns)
    
    # Concatenate the input data with the empty DataFrame
    input_data = pd.concat([empty_df, input_data], ignore_index=True)
    
    # Convert appropriate columns to numeric
    numeric_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 
                        'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
    for feature in numeric_features:
        input_data[feature] = pd.to_numeric(input_data[feature], errors='coerce')
    
    # Perform prediction
    prediction = model.predict(input_data)
    
    # Compare G1 and G2
    G1 = pd.to_numeric(form_data.get('G1', 0), errors='coerce')
    G2 = pd.to_numeric(form_data.get('G2', 0), errors='coerce')

    # Add this line to check the values of G1 and G2
    print(f"G1: {G1}, G2: {G2}") # Check values of G1 and G2
    
    if G1 > G2:
        improvement_message = "G1 can be improved."
    elif G2 > G1:
        improvement_message = "G2 can be improved."
    else:
        improvement_message = "G1 and G2 are equal."
    print("Improvement Message:",improvement_message) # Debug message

    # Motivational quotes
    motivational_quotes = [
        "Believe you can and you're halfway there.",
        "The only limit to our realization of tomorrow is our doubts of today.",
        "You are never too old to set another goal or to dream a new dream.",
        "Success is not the key to happiness. Happiness is the key to success.",
        "The future belongs to those who believe in the beauty of their dreams."
    ]
    motivational_quote = random.choice(motivational_quotes)
    print("Motivational Quote:", motivational_quote)
    
    return render_template(
        'index.html', 
        prediction_text=f'Predicted Final Grade (G3): {prediction[0]:.2f}',
        improvement_message=improvement_message,
        motivational_quote=motivational_quote
    )


if __name__ == "__main__":
    app.run(debug=True)
