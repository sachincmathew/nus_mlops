import requests
import json
import pandas as pd

# Define the API endpoint
url = "http://127.0.0.1:8000/predict/"

# Read data from dummy.csv using pandas
try:
  data = pd.read_csv("dummy.csv", header=None).values.tolist()
except FileNotFoundError:
  print("Error: File 'dummy.csv' not found. Please ensure the file exists.")
  exit(1)

# Ensure data contains 30 features
if len(data[0]) != 30:
  print("Error: 'dummy.csv' does not contain 30 features. Please check the file format.")
  exit(1)

# Convert data to a list of floats (assuming each row has 30 features)
data = [float(value) for value in data[0]]  # Access the first row

# Prepare data dictionary
data = {
  "features": data
}

# Send a POST request
response = requests.post(url, json=data)

# Print the response
print("Status Code:", response.status_code)
print("Response Body:", response.json())
