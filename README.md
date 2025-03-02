# Airbnb Price Prediction - Data Cleaning & Modeling Pipeline

## Overview

This repository contains a comprehensive pipeline for processing, cleaning, and modeling Airbnb listing data for price prediction. The project follows a structured approach, including dataset loading, cleaning, feature engineering, and training multiple regression models. The primary goal is to predict listing prices based on various property attributes and location features.

## 📁 Project Structure
```
📂 data/              # Raw datasets
📂 processed_data/    # Cleaned datasets
📂 notebooks/         # Jupyter notebooks (if applicable)
📂 scripts/           # Python scripts for data processing
📄 README.md          # Project documentation
```

## Function Overview
- `jo(directory, filename)`: Joins a directory and filename.
- `load_data(base_dir)`: Loads raw datasets from the data directory.
- `delete_processed_data(base_dir)`: Deletes all files inside the processed_data directory before saving new ones.
- `load_cleaned_data(base_dir)`: Loads cleaned datasets from the processed_data directory.
- `save_cleaned_data(df, base_dir, filename_prefix)`: Saves a cleaned dataframe with a timestamped filename.
- `drop_unnecessary_columns(df)`: Drops irrelevant columns for price prediction.
- `clean_missing_values(df)`: Removes columns with excessive missing values but retains essential ones.
- `fill_numeric_missing_values(df)`: Fills missing values in numeric columns using mean or median.
- `process_boolean_columns(df)`: Converts boolean columns ('t'/'f') to numeric (1/0) and fills missing values.
- `process_amenities(df, top_n=10)`: Extracts top amenities and creates binary features.
- `process_bathrooms(df)`: Extracts numeric bathroom values and creates shared/private indicators.
- `extract_bathrooms(text)`: Extracts numeric bathroom count from text descriptions.
- `clean_price(df)`: Converts price values to numeric format and removes outliers.
- `clean_host_acceptance_rate(df)`: Converts host acceptance rate to numeric values.
- `categorize_property_type(df)`: Categorizes property types into apartment, house, or other.
- `calculate_distance_to_center(df)`: Computes distance from listing to city center using the Haversine formula.
- `process_neighbourhood(df, top_n=5)`: Keeps top N neighborhoods and one-hot encodes them.
- `filter_data(df)`: Removes outliers and irrelevant data based on predefined thresholds.


## Installation & Setup

To set up the project environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

# Data Processing Pipeline

## 1. Load Raw Data
- The script loads raw CSV files from the `data/` directory.
- If files are missing, a warning is displayed.

## 2. Data Cleaning
The cleaning pipeline includes:

- Dropping unnecessary columns.
- Handling missing values in numeric, boolean, and categorical columns.
- Processing `property type`, `room type`, and `amenities`.
- Converting price values to numeric format.
- Calculating distance to city center using the Haversine formula.
- Encoding neighborhood and amenity data.

## 3. Feature Engineering
- Extracting key features: `accommodates`, `bedrooms`, `bathrooms`, `property type`, `distance to city center`, and more.
- One-hot encoding categorical variables.
- Creating interaction terms for predictive models.

## 4. Model Training & Evaluation
- The script trains and evaluates multiple regression models.


- **Linear Regression**
- **LASSO Regression (Hyperparameter Tuned)**
- **Decision Tree Regressor**
- **Random Forest Regressor (Hyperparameter Tuned)**
- **Gradient Boosting Regressor (GBM)**
- **XGBoost Regressor**

Each model is evaluated using RMSE (Root Mean Squared Error), and results are stored in a structured format.

## 5. Results & Performance
- RMSE values and model execution times are logged.
- The best-performing model for each dataset is identified.

## How to Run the Pipeline
Run the script in the command line:

```sh
python run_pipeline.py
```

This will:

- Load raw Airbnb listing data.
- Clean and preprocess the dataset.
- Save the cleaned dataset to `processed_data/`.
- Train multiple models and evaluate performance.
- Display RMSE and model comparison results.

## Author & Contact
If you have any questions or feedback, feel free to reach out.

📧 Email: [sarkhan.mammadli@example.com](mailto:sarkhan.mammadli@example.com)
🔗 LinkedIn: [linkedin.com/in/mammadli](https://linkedin.com/in/mammadli)  
💻 GitHub: [github.com/mammadli](https://github.com/mammadli)  

Feel free to reach out with any questions or feedback!
