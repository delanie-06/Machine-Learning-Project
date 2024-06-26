from flask import Flask, render_template, request
import pickle
import pandas as pd
import joblib
app = Flask(__name__)

# Load the model
with open('model_saved.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    try:
        cal_fat = float(request.form.get('cal_fat'))
        total_fat = float(request.form.get('total_fat'))
        sat_fat = float(request.form.get('sat_fat'))
        trans_fat = float(request.form.get('trans_fat'))
        cholestrol = float(request.form.get('cholestrol'))
        sodium = float(request.form.get('sodium'))
        total_carb = float(request.form.get('total_carb'))
        fiber = float(request.form.get('fiber'))
        sugar = float(request.form.get('sugar'))
        protein = float(request.form.get('protein'))
        vit_a = float(request.form.get('vit_a'))
        vit_c = float(request.form.get('vit_c'))
        calcium = float(request.form.get('calcium'))
        scaler = joblib.load('MinMaxScaler.pkl')
    except ValueError:
        return render_template('index.html', prediction="Invalid input. Please enter numeric values.")

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[cal_fat, total_fat, sat_fat, trans_fat, cholestrol, sodium,total_carb, fiber, sugar,protein,  vit_a, vit_c,calcium]], columns=['cal_fat', 'total_fat', 'sat_fat','trans_fat','cholesterol','sodium','total_carb','fiber','sugar','protein','vit_a','vit_c','calcium'])
    
    # Make prediction using the loaded model
    try:
        prediction = model.predict(input_data)
        result = float(prediction)
    except Exception as e:
        return render_template('index.html', prediction=f"Prediction failed: {str(e)}")

    # Render the result template with the prediction
    return render_template('index.html', prediction=f"Final Prediction is {result}")

if __name__ == '__main__':
    app.run(debug=True)
    
    #trans_fat	cholesterol	sodium	total_carb	fiber	sugar	protein	vit_a	vit_c	calcium