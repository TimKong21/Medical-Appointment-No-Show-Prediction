
from data_loader import *
from preprocessing import *
from feature_engineering import *
from modeling import *

def main():
    # Load full data
    full_data, _ = load_local_data()

    # Preprocessing
    preprocessed_full_data = preprocess_data(full_data, is_simulated=False)

    # Train test split
    train_set, test_set = split_data(preprocessed_full_data)

    # Create 'GENDER_AGE' feature
    train_set = create_age_gender_feature(train_set)
    test_set = create_age_gender_feature(test_set)

    # Create 'DAYS_TILL_APPOINTMENT' feature
    train_set = calculate_days_till_appointment(train_set)
    test_set = calculate_days_till_appointment(test_set)

    # Extract the year, month, and day from the 'SCHEDULEDDAY' and 'APPOINTMENTDAY' 
    train_set = extract_datetime_features(train_set)
    test_set = extract_datetime_features(test_set)

    # Map the 'GENDER'
    train_set = encode_gender(train_set)
    test_set = encode_gender(test_set)

    # One-hot encode 'GENDER_AGE'
    train_set = one_hot_encode_gender_age(train_set)
    train_set.to_csv('../data/processed/one_hot_encoded_train_set.csv', index=False)
    test_set = one_hot_encode_gender_age(test_set)

    # Target encode 'NEIGHBOURHOOD'
    train_set, test_set = target_encode_neighbourhood(train_set, test_set)

    # Filter and split train set to X and y
    X_train_filtered, y_train = filter_data_with_important_features(train_set)
    X_test_filtered, y_test = filter_data_with_important_features(test_set)

    # Train and evaluate an XGBoost model
    train_and_evaluate_xgboost(X_train_filtered, y_train, X_test_filtered, y_test)

    # Loads a trained model and performs detailed evaluation
    evaluate_model(X_test_filtered, y_test)

if __name__ == "__main__":
    main()
