\# Docker Deployment Report

\## Hotel Cancellation Prediction System



\### 1. Objective

To deploy the Hotel Cancellation Prediction machine learning model

as a REST API using FastAPI and Docker.



\### 2. Deployment Components

\- FastAPI for REST API development.

\- Uvicorn as the ASGI server.

\- Joblib for loading the trained model and preprocessing object.

\- Docker for containerization.

\- Swagger UI for API testing.



\### 3. Files Created

\- api/main.py

\- Dockerfile

\- .dockerignore



\### 4. Model and Preprocessing

\- Model: Random Forest Classifier

\- Model file: models/hotel\_cancellation\_model.joblib

\- Preprocessor: models/preprocessor.joblib

\- Model version: 1.0.0



\### 5. API Endpoints

\- GET / : Displays API information.

\- GET /health : Checks API and model status.

\- POST /predict : Predicts hotel booking cancellation.



\### 6. Docker Implementation

Docker image build command:



docker build -t hotel-cancellation-api .



Docker container run command:



docker run -d --name hotel-cancellation-container -p 8001:8000 hotel-cancellation-api



\### 7. API Testing

Swagger UI URL:



http://127.0.0.1:8001/docs



Health endpoint:



http://127.0.0.1:8001/health



The health endpoint was tested successfully.



Docker container status was verified using:



docker ps



Docker logs were checked using:



docker logs hotel-cancellation-container



\### 8. Result

The Hotel Cancellation Prediction API was successfully

containerized using Docker. The Docker container started,

and the FastAPI application ran without startup errors.

