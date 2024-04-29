from fastapi import FastAPI, HTTPException  # Import necessary components from FastAPI and HTTPException for error handling.
from pydantic import BaseModel  # Import BaseModel from Pydantic to define data models.
from typing import List  # Import List from typing to specify data types in data models.
from joblib import load  # Import load function from joblib to load pre-trained models.
import datetime

app = FastAPI()  # Create an instance of FastAPI to define and manage your web application.

# Load pre-trained model and scaler objects from disk. These are used for making predictions and scaling input data, respectively.
model = load('_breast_cancer_model.joblib')
scaler = load('_scaler.joblib')

# Define a data model for incoming prediction requests using Pydantic.
# This model ensures that data received via the API matches the expected format.
class QueryData(BaseModel):
    features: List[float]  # Define a list of floating point numbers to represent input features for prediction.


@app.get("/")
async def read_root():
    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time according to your preference
    formatted_datetime = now.strftime("%d-%m-%Y %H:%M:%S")  # Example format (YYYY-MM-DD HH:MM:SS)

    # Print the formatted date and time
    return f"Hello, it is now {formatted_datetime}"

# Decorator to create a new route that accepts POST requests at the "/predict/" URL.
# This endpoint will be used to receive input data and return model predictions.
# Declaring async before a function definition is a way to handle asynchronous operations in FastAPI. 
# It allows the server to handle many requests efficiently by not blocking the server during operations 
# like network calls or while waiting for file I/O.
@app.post("/predict/")
async def make_prediction(query: QueryData):
    try:
        # The input data is received as a list of floats, which needs to be scaled (normalized) using the previously loaded scaler.
        scaled_features = scaler.transform([query.features])
        
        # Use the scaled features to make a prediction using the loaded model.
        # The model returns a list of predictions, and we take the first item since we expect only one.
        prediction = model.predict(scaled_features)
        
        # Return the prediction as a JSON object. This makes it easy to handle the response programmatically on the client side.
        return {"prediction": int(prediction[0])}
    except Exception as e:
        # If an error occurs during the prediction process, raise an HTTPException which will be sent back to the client.
        raise HTTPException(status_code=400, detail=str(e))
