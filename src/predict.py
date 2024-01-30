
from data_loader import *
from preprocessing import *
from feature_engineering import *
from modeling import *

def main():
    # Load simulated data
    _, simulated_data = load_local_data()

    # Preprocessing
    preprocessed_simulated_set = preprocess_data(simulated_data, is_simulated=True)

    # Create 'GENDER_AGE' feature
    preprocessed_simulated_set = create_age_gender_feature(preprocessed_simulated_set)

    # Create 'DAYS_TILL_APPOINTMENT' feature 
    preprocessed_simulated_set = calculate_days_till_appointment(preprocessed_simulated_set)

    # Extract the year, month, and day from the 'SCHEDULEDDAY' and 'APPOINTMENTDAY'
    preprocessed_simulated_set = extract_datetime_features(preprocessed_simulated_set)

    # Map the 'GENDER'
    preprocessed_simulated_set = encode_gender(preprocessed_simulated_set)

    # One-hot encode 'GENDER_AGE'
    preprocessed_simulated_set = one_hot_encode_gender_age(preprocessed_simulated_set)

    # Target encode 'NEIGHBOURHOOD'
    one_hot_encoded_train_set = pd.read_csv('../data/processed/one_hot_encoded_train_set.csv')
    _, preprocessed_simulated_set = target_encode_neighbourhood(one_hot_encoded_train_set, preprocessed_simulated_set)

    # Filter simulate set
    X_simulated_filtered, _ = filter_data_with_important_features(preprocessed_simulated_set)

    # Loads a trained model and makes predictions
    load_model_and_predict(X_simulated_filtered)

if __name__ == "__main__":
    main()
