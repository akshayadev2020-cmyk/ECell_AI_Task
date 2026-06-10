 E-Cell AI Task 1: SEC 10-K Filing Intelligence System

This repository contains an end-to-end Machine Learning pipeline to analyze and classify 10-K SEC financial filings based on risk levels.

 Project Structure
- api/: FastAPI application for live risk prediction.
- models/: Serialized CatBoost model and TF-IDF vectorizer.
- notebooks/: Exploratory analysis and the final project report (PDF).
- src/: Modular Python source code for data processing and training.

 How to Run

1. Install dependencies:

   pip install -r requirements.txt

2.Run the API

uvicorn api.app:app --reload

3.Access the Interface:

Go to http://127.0.0.1:8000/docs to test the /predict endpoint.