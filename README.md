# RAIDO - Data Preprocessing Tool

## Description
This codebase includes the main data preprocessing functionalities that will be developed within RAIDO's T3.1. Currently, the avaialble functionalities
support images and time series data. The implementations of the functionalities are in src/images/ and src/time_series/ for images and time series, respectively.
Additionally, the API is implemented in api/routes/images/ and api/routes/time_series/ using FastAPI and the directory follows the same structure as the src/ directories.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Documentation](#documentation)

## Installation
Create a virtual environment and install the dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Activate the virtual environment:
```bash
source .venv/bin/activate
```

## Usage
You can use the avaialble data preprocessing functionalities in two different ways:

1. Using the API: To use the API, run:
```bash
uvicorn api_main:app --reload
```
This will start the FastAPI server and you can access the API documentation at http://127.0.0.1:8000/docs.

2. Using the functions in src/: To this end, you can try out the functionalities in src/time_series/ and src/images/ in main.py which is intentionally left empty.


## Features
- Image Preprocessing:
    - Balancing [WIP]
    - Dimensionality Reduction
    - Data Enrichment [WIP]
    - Feature Engineering [WIP]
    - Missing Data
    - Noise Removal
    - Outliers
    - Visualization
- Time Series Preprocessing:
    - Balancing [WIP]
    - Dimensionality Reduction [WIP]
    - Data Enrichment [WIP]
    - Feature Engineering [WIP]
    - Missing Data
    - Noise Removal [WIP]
    - Outliers
    - Visualization [WIP]

## Documentation
*[TODO] Further reading or API documentation links to be added here*

## Testing
To test the API, first run the server:
```bash
uvicorn api_main:app --reload
```

Then, run the test file:
```bash
python3 test_multi_file_endpoints.py
```
