
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def select_and_save_important_features(train_data, target_column='NO_SHOW', output_path='../data/features/combined_important_features.pkl'):
    """
    Selects important features based on Logistic Regression and Decision Tree models and saves them to a file.
    
    :param train_data: The training dataset.
    :param target_column: The name of the target column.
    :param output_path: The path to save the important features.
    """
    # Prepare the data
    X = train_data.drop(target_column, axis=1)
    y = train_data[target_column]

    # Initialize and fit the Logistic Regression model
    logistic_model = LogisticRegression(class_weight='balanced', random_state=42)
    logistic_model.fit(X, y)

    # Get coefficients and filter important features
    logistic_coef = pd.DataFrame(logistic_model.coef_.reshape(-1, 1), index=X.columns, columns=['Coefficient'])
    important_features_logistic = logistic_coef[abs(logistic_coef['Coefficient']) >= 0.01].index.values.tolist()

    # Initialize and fit the Decision Tree model
    decision_tree_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    decision_tree_model.fit(X, y)

    # Get feature importances and filter important features
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': decision_tree_model.feature_importances_})
    feature_importances = feature_importances.sort_values('Importance', ascending=False)
    important_features_tree = feature_importances[feature_importances['Importance'] >= 0.01]['Feature'].values.tolist()

    # Combine the important features from both models
    combined_important_features_union = list(set(important_features_logistic) | set(important_features_tree))

    # Save the combined important features to a file
    with open(output_path, 'wb') as f:
        pickle.dump(combined_important_features_union, f)

    print(f"Important features saved to {output_path}")

def filter_data_with_important_features(data, important_features_path='../data/features/X_train_important_features.pkl', target_column='NO_SHOW'):
    """
    Filters the given dataset with important features, adds missing features with default values, 
    and separates the target variable.
    
    :param data: The dataset to be filtered.
    :param important_features_path: The path to the pickle file containing important features.
    :param target_column: The name of the target column.
    :return: A tuple containing the filtered features (X) and the target variable (Y).
    """
    # Load the important features from the pickle file
    with open(important_features_path, 'rb') as f:
        important_features = pickle.load(f)

    # Initialize a DataFrame with zeros for all important features
    X_filtered = pd.DataFrame(0, index=data.index, columns=important_features)

    # Update the DataFrame with the data from the original dataset
    # Ensure to exclude the target column
    X_filtered.update(data[important_features])

    # Extract the target variable if it exists in the dataset
    Y = data[target_column] if target_column in data.columns else None

    return X_filtered, Y

def tune_hyperparameters(X_train, y_train, max_evals=100):
    """
    Tune hyperparameters for the XGBoost model using Hyperopt.

    :param X_train: Training features.
    :param y_train: Training labels.
    :param max_evals: The maximum number of evaluations during optimization.
    :return: A dictionary containing the best hyperparameters.
    """
    best_score = 0  # Initialize the best score
    iteration = 0  # Initialize the iteration counter

    # Define the objective function
    def objective(params):
        nonlocal best_score  # Use nonlocal to refer to the outer scope's best_score
        nonlocal iteration  # Use nonlocal to refer to the outer scope's iteration
        iteration += 1  # Increment the iteration counter
        
        clf = XGBClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            colsample_bytree=params['colsample_bytree'],
            subsample=params['subsample'],
            min_child_weight=int(params['min_child_weight']),
            scale_pos_weight=(0.798 / 0.202),
            random_state=42
        )
        
        # Stratified K-Fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(clf, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1).mean()
        
        if score > best_score:
            best_score = score
            print(f"New best score at iteration {iteration}: {best_score}")
            display("Best parameters so far:", params)
        
        return {'loss': -score, 'status': STATUS_OK}

    # Define the parameter space
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 1000, 1),
        'max_depth': hp.quniform('max_depth', 3, 14, 1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'gamma': hp.uniform('gamma', 0, 0.5),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
        'subsample': hp.uniform('subsample', 0.6, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1)
    }

    # Initialize a trials object
    trials = Trials()

    # Run the hyperparameter optimization
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )

    # Save the best hyperparameters to a pickle file
    with open('../data/hyperparameters/XGBoost_hyperparameters.pkl', 'wb') as f:
        pickle.dump(best, f)

    print("Best hyperparameters saved to '../data/hyperparameters/XGBoost_hyperparameters.pkl'")
    
def train_and_evaluate_xgboost(X_train, y_train, X_test, y_test, hyperparameters_path='../data/hyperparameters/XGBoost_hyperparameters.pkl'):
    """
    Train and evaluate an XGBoost model using the best hyperparameters.

    :param X_train: Training features.
    :param y_train: Training labels.
    :param X_test: Testing features.
    :param y_test: Testing labels.
    :param hyperparameters_path: Path to the pickle file containing the best hyperparameters.
    """
    # Load the best hyperparameters
    with open(hyperparameters_path, 'rb') as f:
        best_params = pickle.load(f)

    # Convert float values to int for certain parameters
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    best_params['scale_pos_weight'] = 0.798 / 0.202
    best_params['random_state'] = 42

    # Initialize the XGBoost model
    xgb_model_final = XGBClassifier(**best_params)

    # Fit the model on the filtered training set
    xgb_model_final.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_final = xgb_model_final.predict(X_test)
    y_pred_prob = xgb_model_final.predict_proba(X_test)[:, 1]

    # Evaluate the model
    roc_auc_final = roc_auc_score(y_test, y_pred_prob)
    f1_score_final = f1_score(y_test, y_pred_final)

    print(f"Final Model Evaluation:")
    print(f"ROC AUC: {roc_auc_final}")
    print(f"F1 Score: {f1_score_final}")
    
    # Save the trained model to a pickle file
    with open('../model/xgb_model.pkl', 'wb') as model_file:
        pickle.dump(xgb_model_final, model_file)

    print("Model saved to '../model/xgb_model.pkl'")

def load_model_and_predict(X_simulated, model_path='../model/xgb_model.pkl', output_path='../data/output/predictions_with_features.csv'):
    """
    Loads a trained model, makes predictions on the provided dataset, and saves the predictions along with features.

    :param X_simulated: The dataset to make predictions on.
    :param model_path: Path to the trained model file.
    :param output_path: Path to save the predictions with features.
    :return: None
    """
    # Load the trained model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Make predictions
    predictions = model.predict(X_simulated)

    # Reset index to bring 'APPOINTMENTID' back as a column
    X_simulated_reset = X_simulated.reset_index()

    # Combine predictions with features
    predictions_df = X_simulated_reset.copy()
    predictions_df['PREDICTED_NO_SHOW'] = predictions

    # Save the combined DataFrame to a CSV file
    predictions_df.to_csv(output_path, index=False)

    print(f"Predictions with features saved to {output_path}")
    
def evaluate_model(X_test_filtered, y_test, model_path="../model/xgb_model.pkl"):
    """
    Loads a trained model and performs detailed evaluation on the test set.

    :param X_test_filtered: Filtered features of the test set.
    :param y_test: Target variable of the test set.
    :param model_path: Path to the trained model file.
    :return: None
    """
    # Load the trained model
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)

    # Make predictions on the test set
    y_pred_test = loaded_model.predict(X_test_filtered)
    y_pred_prob_test = loaded_model.predict_proba(X_test_filtered)[:, 1]

    # Evaluate on Test Set
    print("Evaluation on Test Set:")
    evaluate_predictions(y_test, y_pred_test, y_pred_prob_test)

def evaluate_predictions(y_true, y_pred, y_pred_prob):
    """
    Generates and displays evaluation metrics and plots for predictions.

    :param y_true: True target values.
    :param y_pred: Predicted target values.
    :param y_pred_prob: Predicted probabilities.
    :return: None
    """
    # Compute and display confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['No-Show', 'Show']).plot()
    plt.title('Confusion Matrix')
    plt.show()

    # Generate and print classification report
    report = classification_report(y_true, y_pred, target_names=['No-Show', 'Show'])
    print("Classification Report:")
    print(report)

    # Compute and plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    # Compute and plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=1, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')
    plt.show()
