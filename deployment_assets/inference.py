##Inference code for sagemaker model

import subprocess
import sys
import importlib

def install_and_import(package):
    try:
        # Try importing the package
        importlib.import_module(package)
    except ImportError:
        # If the package is not present, install it
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        # After successful installation, import the package
        globals()[package] = importlib.import_module(package)

# Example usage
install_and_import('category_encoders')
install_and_import('sagemaker')
install_and_import("pytz")

import pytz
from category_encoders import TargetEncoder
import numpy as np
import pandas as pd
import json
import os
import pickle
from io import StringIO
import sagemaker
from datetime import datetime

def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        csv_data = request_body
        simulated_data = pd.read_csv(StringIO(csv_data), parse_dates=['SCHEDULEDDAY', 'APPOINTMENTDAY'])
        print(type(simulated_data))
        print(simulated_data.head())
        
        # Preprocessing
        preprocessed_simulated_set = preprocess_data(simulated_data, is_simulated=True)
        print(type(preprocessed_simulated_set))
        print(preprocessed_simulated_set.head())
        
        # Create 'GENDER_AGE' feature
        preprocessed_simulated_set = create_age_gender_feature(preprocessed_simulated_set)
        print(type(preprocessed_simulated_set))
        print(preprocessed_simulated_set.head())
        
        # Create 'DAYS_TILL_APPOINTMENT' feature 
        preprocessed_simulated_set = calculate_days_till_appointment(preprocessed_simulated_set)
        print(type(preprocessed_simulated_set))
        print(preprocessed_simulated_set.head())
        
        # Extract the year, month, and day from the 'SCHEDULEDDAY' and 'APPOINTMENTDAY'
        preprocessed_simulated_set = extract_datetime_features(preprocessed_simulated_set)
        print(type(preprocessed_simulated_set))
        print(preprocessed_simulated_set.head())
        
        # Map the 'GENDER'
        preprocessed_simulated_set = encode_gender(preprocessed_simulated_set)
        print(type(preprocessed_simulated_set))
        print(preprocessed_simulated_set.head())
        
        # One-hot encode 'GENDER_AGE'
        preprocessed_simulated_set = one_hot_encode_gender_age(preprocessed_simulated_set)
        print(type(preprocessed_simulated_set))
        print(preprocessed_simulated_set.head())
        
        # Target encode 'NEIGHBOURHOOD'
        one_hot_encoded_train_set = pd.read_csv(os.path.join(model_dir, 'one_hot_encoded_train_set.csv'))
        _, preprocessed_simulated_set = target_encode_neighbourhood(one_hot_encoded_train_set, preprocessed_simulated_set)
        print(type(preprocessed_simulated_set))
        print(preprocessed_simulated_set.head())
        
        # Filter simulate set
        X_simulated_filtered, _ = filter_data_with_important_features(preprocessed_simulated_set)
        print(type(X_simulated_filtered))
        print(X_simulated_filtered.head())
        
        return X_simulated_filtered
    else:
        # Handle other content-types here or raise an exception
        raise ValueError("Unsupported content type: {}".format(request_content_type))

def set_global_variable(value):
    # Declare the variable as global within the function
    global model_dir
    # Modify the global variable
    model_dir = value
        
def model_fn(model_dir):
    """
    Load the XGBoost model from the model_dir directory.
    """
    with open(os.path.join(model_dir, 'xgb_model.pkl'), 'rb') as model_file:
        model = pickle.load(model_file)

    set_global_variable(model_dir)
    
    return model     

def predict_fn(input_data, model):
    # Make predictions
    predictions = model.predict(input_data)

    # Reset index to bring 'APPOINTMENTID' back as a column
    input_data_reset = input_data.reset_index()

    # Combine predictions with features
    predictions_df = input_data_reset.copy()
    predictions_df['PREDICTED_NO_SHOW'] = predictions

    # Specify the Vancouver time zone
    vancouver_tz = pytz.timezone('America/Vancouver')

    # Convert current time to Vancouver time and format as timestamp
    timestamp = datetime.now(tz=pytz.utc).astimezone(vancouver_tz).strftime("%Y%m%d_%H%M%S")

    output_dir = '/opt/ml/output/data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'medical_appointment_no_show_prediction_{timestamp}.csv')
    # Save the combined DataFrame to a CSV file
    predictions_df.to_csv(output_path, index=False)

    # Upload the CSV file to S3
    s3_bucket = 'predicting-medical-appointment-no-shows'
    s3_key = f'predictions'

    sagemaker.s3.S3Uploader.upload(local_path=output_path, 
                                   desired_s3_uri=f's3://{s3_bucket}/{s3_key}')

    print(f"Predictions with features saved to {s3_bucket}/{s3_key}")
    os.remove(output_path)

    return predictions

def preprocess_data(data, is_simulated=False):
    """
    Preprocesses the medical appointment data.

    :param data: The DataFrame containing the medical appointment data.
    :param is_simulated: A boolean flag indicating whether the data is a simulated set with NaN target values.
    :return: The preprocessed DataFrame.
    """
    # Drop 'PATIENTID' column
    data = data.drop('PATIENTID', axis=1)
    
    # Set 'APPOINTMENTID' as index
    data = data.set_index('APPOINTMENTID')
    
    # Map 'NO_SHOW' to 0 and 1 only if it's not a simulated set
    if not is_simulated:
        data['NO_SHOW'] = data['NO_SHOW'].map({'No': 0, 'Yes': 1})
    
    # Ensure the difference between 'APPOINTMENTDAY' and 'SCHEDULEDDAY' is non-negative
    data = data[(data['APPOINTMENTDAY'] - data['SCHEDULEDDAY']).dt.days + 1 >= 0]
    
    # Ensure 'AGE' is non-negative
    data = data[data['AGE'] >= 0]
    
    return data

def create_age_gender_feature(data):
    """
    Creates a combined feature of gender and bucketized age in the given dataset.
    
    :param data: The DataFrame to process.
    :return: The DataFrame with the new combined feature.
    """
    bins = [0, 13, 18, 31, 51, np.inf]
    labels = ['0-12', '13-17', '18-30', '31-50', '50+']

    # Create AGE_BUCKET and then create GENDER_AGE
    data['AGE_BUCKET'] = pd.cut(data['AGE'], bins=bins, labels=labels, right=False)
    data['GENDER_AGE'] = data['GENDER'] + '_' + data['AGE_BUCKET'].astype(str)

    # Drop the AGE_BUCKET column
    data.drop('AGE_BUCKET', axis=1, inplace=True)
    
    # Reorder columns to make 'NO_SHOW' the last column, if it exists
    if 'NO_SHOW' in data.columns:
        cols = list(data.columns)
        cols.remove('NO_SHOW')
        cols.append('NO_SHOW')
        data = data[cols]

    return data

def calculate_days_till_appointment(data):
    """
    Adds a feature representing the number of days till the appointment.

    :param data: Pandas DataFrame to be processed.
    :return: DataFrame with the new feature added.
    """
    # Calculate the number of days till the appointment
    data['DAYS_TILL_APPOINTMENT'] = (data['APPOINTMENTDAY'] - data['SCHEDULEDDAY']).dt.days + 1

    # If 'NO_SHOW' is a column, reorder columns to make it the last column
    if 'NO_SHOW' in data.columns:
        cols = list(data.columns)
        cols.remove('NO_SHOW')
        cols.append('NO_SHOW')
        data = data[cols]

    return data

def extract_datetime_features(data):
    """
    Extracts year, month, and day from 'SCHEDULEDDAY' and 'APPOINTMENTDAY' and adds them as new features.

    :param data: Pandas DataFrame to be processed.
    :return: DataFrame with new datetime features.
    """
    # Extracting year, month, and day from SCHEDULEDDAY
    data['SCHEDULEDDAY_YEAR'] = data['SCHEDULEDDAY'].dt.year
    data['SCHEDULEDDAY_MONTH'] = data['SCHEDULEDDAY'].dt.month
    data['SCHEDULEDDAY_DAY'] = data['SCHEDULEDDAY'].dt.day

    # Extracting year, month, and day from APPOINTMENTDAY
    data['APPOINTMENTDAY_YEAR'] = data['APPOINTMENTDAY'].dt.year
    data['APPOINTMENTDAY_MONTH'] = data['APPOINTMENTDAY'].dt.month
    data['APPOINTMENTDAY_DAY'] = data['APPOINTMENTDAY'].dt.day

    # Dropping the original datetime columns
    data.drop(['SCHEDULEDDAY', 'APPOINTMENTDAY'], axis=1, inplace=True)
    
    # Columns to check for unique values
    year_columns_to_check = ['SCHEDULEDDAY_YEAR', 'APPOINTMENTDAY_YEAR']

    # Check and drop year columns with only one unique value
    for col in year_columns_to_check:
        if data[col].nunique() == 1:
            #print(f"Dropping {col} as it has only one unique value.")
            data.drop([col], axis=1, inplace=True)

    # If 'NO_SHOW' is a column, reorder columns to make it the last column
    if 'NO_SHOW' in data.columns:
        cols = list(data.columns)
        cols.remove('NO_SHOW')
        cols.append('NO_SHOW')
        data = data[cols]

    return data

def encode_gender(data):
    """
    Encodes the 'GENDER' column in the dataset, mapping 'M' to 1 and 'F' to 0.

    :param data: Pandas DataFrame to be processed.
    :return: DataFrame with the encoded 'GENDER' column.
    """
    gender_mapping = {'M': 1, 'F': 0}
    data['GENDER'] = data['GENDER'].map(gender_mapping)
    return data

def one_hot_encode_gender_age(data):
    """
    One-hot encodes the 'GENDER_AGE' column in the dataset.

    :param data: Pandas DataFrame to be processed.
    :return: DataFrame with one-hot encoded 'GENDER_AGE' column.
    """
    data = pd.get_dummies(data, columns=['GENDER_AGE'], drop_first=True, dtype=int)

    # Reorder columns to make 'NO_SHOW' the last column, if it exists
    if 'NO_SHOW' in data.columns:
        cols = list(data.columns)
        cols.remove('NO_SHOW')
        cols.append('NO_SHOW')
        data = data[cols]

    return data

def target_encode_neighbourhood(train_data, test_data):
    """
    Target encodes the 'NEIGHBOURHOOD' column in the training and testing datasets.

    :param train_data: Training dataset as a Pandas DataFrame.
    :param test_data: Testing dataset as a Pandas DataFrame.
    :return: Tuple of DataFrames with target encoded 'NEIGHBOURHOOD' column.
    """
    encoder = TargetEncoder()

    # Fit and transform on the training data
    train_data['NEIGHBOURHOOD'] = encoder.fit_transform(train_data['NEIGHBOURHOOD'], train_data['NO_SHOW'])

    # Transform on the testing data
    test_data['NEIGHBOURHOOD'] = encoder.transform(test_data['NEIGHBOURHOOD'])

    return train_data, test_data

def filter_data_with_important_features(data, target_column='NO_SHOW'):
    """
    Filters the given dataset with important features, adds missing features with default values, 
    and separates the target variable.
    
    :param data: The dataset to be filtered.
    :param important_features_path: The path to the pickle file containing important features.
    :param target_column: The name of the target column.
    :return: A tuple containing the filtered features (X) and the target variable (Y).
    """
    # Load the important features from the pickle file
    with open(os.path.join(model_dir, 'X_train_important_features.pkl'), 'rb') as model_file:
        important_features = pickle.load(model_file)
        
    # Initialize a DataFrame with zeros for all important features
    X_filtered = pd.DataFrame(0, index=data.index, columns=important_features)

    # Update the DataFrame with the data from the original dataset
    # Ensure to exclude the target column
    X_filtered.update(data[important_features])

    # Extract the target variable if it exists in the dataset
    Y = data[target_column] if target_column in data.columns else None

    return X_filtered, Y