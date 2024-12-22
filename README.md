# Heart Failure Prediction Web Application

This web application predicts the likelihood of heart failure based on user-provided medical data using machine learning models like XGBoost and Neural Networks. It also visualizes data analysis, training metrics, and model evaluation results.

## Data Source
The dataset used for this project is available on Kaggle: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

## Features

- **Data Analysis**: Visualizes numerical and categorical data distributions, outliers, and correlations.
- **Model Training**: Allows users to train models using XGBoost or Neural Networks with adjustable hyperparameters.
- **Prediction**: Accepts user inputs to predict heart disease probability.

## File Structure

### Backend
- **`src/api_router/train.py`**: Handles model training (XGBoost and Neural Network).
- **`src/api_router/predict.py`**: Processes prediction requests based on trained models.

### Frontend
- **`src/templates/index.html`**: Main layout with navigation.
- **`src/templates/components/data.html`**: Displays data visualizations.
- **`src/templates/components/model.html`**: Provides model training interface.
- **`src/templates/components/predict.html`**: Input form for predictions.


### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/kuongan/Heart-Failure-Prediction.git
   cd Heart-Failure-Prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   uvicorn src.app:app --reload
   ```

4. Open the app in your browser:
   ```
   http://127.0.0.1:8000
   ```

## Usage

### Data Analysis
Navigate to the "Data" section to view:
- Data distributions
- Outliers and preprocessing steps
- Correlation matrices

### Model Training
- Go to the "Train Model" section.
- Select a model (XGBoost or Neural Network).
- Adjust hyperparameters and start training.

### Prediction
- Visit the "Predict" section.
- Enter medical data (e.g., age, cholesterol levels).
- Get prediction results .

