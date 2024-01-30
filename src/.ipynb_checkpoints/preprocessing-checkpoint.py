
import pandas as pd

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
    #data = data[(data['APPOINTMENTDAY'] - data['SCHEDULEDDAY']).dt.days + 1 >= 0]
    
    # Ensure 'AGE' is non-negative
    #data = data[data['AGE'] >= 0]
    
    return data
