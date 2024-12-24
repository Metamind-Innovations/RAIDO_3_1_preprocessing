import requests

# Obtain the access token
token_response = requests.post("http://localhost:8001/time_series/token", data={"username": "user", "password": "pass"})
token = token_response.json()["access_token"]

# Set up headers and files
headers = {"Authorization": f"Bearer {token}"}
files = {"csv": open("power_small.csv", "rb")}
params = {"method": "fill"}

# Make the POST request
response = requests.post("http://localhost:8001/time_series/missing_data/impute", headers=headers, files=files, params=params)
print(response.json())