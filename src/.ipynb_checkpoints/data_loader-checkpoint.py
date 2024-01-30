
import pandas as pd
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
from sklearn.model_selection import train_test_split

def load_data_and_create_simulated_set(snowflake_creds, stratify_colname='TARGET_COLUMN', test_size=0.2, random_state=42):
    """
    Connects to Snowflake, retrieves data, and creates a stratified simulated set.
    
    :param snowflake_creds: A credentials object with USER_NAME and PASSWORD attributes.
    :param stratify_colname: The name of the column to stratify by.
    :param test_size: The proportion of the dataset to include in the simulated set.
    :param random_state: The seed used by the random number generator.
    :return: A tuple containing the full dataset and the simulated set.
    """
    
    # Define the Snowflake connection parameters
    engine = create_engine(URL(
            account="mtb69989.us-east-1",
            user=snowflake_creds.USER_NAME,
            password=snowflake_creds.PASSWORD,
            role="ACCOUNTADMIN",
            warehouse="COMPUTE_WH",
            database="MEDICAL_APPOINTMENT_NO_SHOW",
            schema="APPOINTMENT_SCHEMA"
        ))

    # Define the SQL query that retrieves the appointment data
    query = """
    SELECT * FROM MEDICAL_APPOINTMENT_NO_SHOW.APPOINTMENT_SCHEMA.APPOINTMENT_DATA;
    """

    # Use a context manager to ensure the connection is closed after executing the query
    try:
        with engine.connect() as conn:
            # Execute the query and load the result into a Pandas DataFrame
            data = pd.read_sql(query, conn)
            # Convert column names to uppercase
            data.columns = [col.upper() for col in data.columns.tolist()]
            
    except Exception as e:  # Catch any exceptions that occur
        print(f"An error occurred: {e}")
        return None, None

    # Optionally, save the data frame to our local disk
    data.to_pickle("../data/input/full_data.pkl")

    # Create a stratified simulated set
    _, simulated_set = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[stratify_colname] if stratify_colname in data.columns else None
    )
    
    # Set the target column in the simulated set to empty values
    if stratify_colname in simulated_set.columns:
        simulated_set[stratify_colname] = ''

    # Save the simulated set to a CSV file
    simulated_set.to_csv('../data/input/simulated_set.csv', index=False)
    
    return data, simulated_set

def load_local_data(full_data_path="../data/input/full_data.pkl", simulated_data_path="../data/input/simulated_set.csv"):
    """
    Loads the full dataset and simulated set from local disk.
    
    :param full_data_path: The path to the full data pickle file.
    :param simulated_data_path: The path to the simulated set CSV file.
    :return: A tuple containing the full dataset and the simulated set.
    """
    try:
        # Load the full dataset from a pickle file
        full_data = pd.read_pickle(full_data_path)
        
        # Load the simulated set from a CSV file
        simulated_set = pd.read_csv(simulated_data_path, parse_dates=['SCHEDULEDDAY', 'APPOINTMENTDAY'])
        
        return full_data, simulated_set
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None, None
