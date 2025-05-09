import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
expected_columns = pickle.load(open('columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    features_framed = pd.DataFrame([features], columns=[
        "Order Date", "Ship Date", "Ship Mode", "Segment", "City", "State", 
        "Region", "Category", "Sub-Category", "Quantity", "Discount", "Profit"
    ])

    # Convert string to datetime
    features_framed["Order Date"] = pd.to_datetime(features_framed["Order Date"], format="%m/%d/%Y", errors="coerce")
    features_framed["Ship Date"] = pd.to_datetime(features_framed["Ship Date"], format="%m/%d/%Y", errors="coerce")

    # Calculate shipping duration and date features
    features_framed["Shipping_Duration_Days"] = (features_framed["Ship Date"] - features_framed["Order Date"]).dt.days
    features_framed["Order_Year"] = features_framed["Order Date"].dt.year
    features_framed["Order_Month"] = features_framed["Order Date"].dt.month
    features_framed["Order_Day"] = features_framed["Order Date"].dt.day
    features_framed["Order_Weekday"] = features_framed["Order Date"].dt.weekday

    # Drop original date columns
    features_framed.drop(columns=["Order Date", "Ship Date"], inplace=True)

    # Convert numeric fields properly
    numeric_columns = ["Quantity", "Discount", "Profit"]
    for col in numeric_columns:
        features_framed[col] = pd.to_numeric(features_framed[col], errors='coerce')
    features_framed[numeric_columns] = features_framed[numeric_columns].fillna(0)

    # One-hot encode categorical columns
    one_hot_cols = ["Category", "Sub-Category", "Segment", "Ship Mode", "Region"]
    features_framed = pd.get_dummies(features_framed, columns=one_hot_cols, drop_first=True)

    # Label encode 'State'
    state_encoder = LabelEncoder()
    features_framed["State_Encoded"] = state_encoder.fit_transform(features_framed["State"])
    features_framed.drop(columns=["State"], inplace=True)

    # Group cities
    top_cities = [
        'New York City', 'Los Angeles', 'Philadelphia', 'San Francisco',
        'Seattle', 'Houston', 'Chicago', 'Columbus', 'San Diego', 'Dallas'
    ]
    features_framed["City_Grouped"] = features_framed["City"].apply(lambda x: x if x in top_cities else "Other")
    features_framed = pd.get_dummies(features_framed, columns=["City_Grouped"], drop_first=True)
    features_framed.drop(columns=["City"], inplace=True)

    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in features_framed.columns:
            features_framed[col] = 0

    # Reorder to match model input
    features_framed = features_framed[expected_columns]

    # Predict
    prediction = model.predict(features_framed)
    sales_prediction = round(np.expm1(prediction[0]))  # Inverse of log1p

    return render_template('index.html', prediction_text=f"Sales Should be {sales_prediction}")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
