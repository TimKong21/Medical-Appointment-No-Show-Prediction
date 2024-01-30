
import pandas as pd
import numpy as np

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
