
import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler

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

def split_data(data, target_col='NO_SHOW', test_size=0.2, random_state=42):
    """
    Splits the data into training and test sets without separating X and y.
    
    :param data: The DataFrame to split.
    :param target_col: The name of the target column.
    :param test_size: The proportion of the dataset to include in the test split.
    :param random_state: The seed used by the random number generator.
    :return: A tuple containing the training set and test set.
    """
    return train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data[target_col]
    )

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

def apply_scaling(train_set, test_set):
    """
    Applies scaling to the feature columns of the train and test sets.
    
    :param train_set: The training set DataFrame.
    :param test_set: The test set DataFrame.
    :return: Scaled versions of the train and test sets.
    """
    # Initialize the scaler
    scaler = StandardScaler()

    # List of feature columns to scale, excluding the target column 'NO_SHOW'
    feature_cols = train_set.columns.difference(['NO_SHOW'])

    # Create copies of the original data
    train_set_scaled = train_set.copy()
    test_set_scaled = test_set.copy()

    # Apply scaling to the feature columns in the training and test sets
    train_set_scaled[feature_cols] = scaler.fit_transform(train_set[feature_cols])
    test_set_scaled[feature_cols] = scaler.transform(test_set[feature_cols])

    return train_set_scaled, test_set_scaled
