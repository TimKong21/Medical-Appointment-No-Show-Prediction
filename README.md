# Predicting Patient No-Shows in Healthcare Appointments

## Business Problem

A significant issue in medical setting is patients failing to attend scheduled doctor appointments despite receiving instructions (no-shows). Our client, a medical ERP solutions provider, seeks to tackle this by introducing a machine learning model into their software. This model aims to predict patient attendance, enabling medical providers to optimize appointment management.

<p align="center">
    <img src="Notebook_images/Patient no show.png" alt=" Intro image" style="width: 80%"/>
</p>

## Dataset Description

The dataset from [Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments) utilized in this project comprises appointment records from medical institutions, capturing various attributes related to patients and their appointments. Key features include:

- **Patient demographics**: Age and gender.

- **Health characteristics**: The presence of conditions such as diabetes or hypertension.

- **Appointment-specific details**: Scheduled and appointment dates, and whether the patient received a reminder SMS.

- **Target**: Binary indicator representing whether a patient was a no-show or attended their appointment.

    | No | Column Name | Description |
    | --- | --- | --- |
    | 01 | PatientId | Identification of a patient |
    | 02 | AppointmentID | Identification of each appointment |
    | 03 | Gender | Male or Female. Female is the greater proportion, women take way more care of their health in comparison to men. |
    | 04 | ScheduledDay | The day someone called or registered the appointment, this is before the appointment of course. |
    | 05 | AppointmentDay | The day of the actual appointment, when they have to visit the doctor. |
    | 06 | Age | How old is the patient. |
    | 07 | Neighbourhood | Where the appointment takes place. |
    | 08 | Scholarship | True or False. Indicates whether the patient is enrolled in Brasilian welfare program Bolsa Família. |
    | 09 | Hipertension | True or False. Indicates if the patient has hypertension. |
    | 10 | Diabetes | True or False. Indicates if the patient has diabetes. |
    | 11 | Alcoholism | True or False. Indicates if the patient is an alcoholic. |
    | 12 | Handcap | True or False. Indicates if the patient is handicapped. |
    | 13 | SMS_received | True or False. Indicates if 1 or more messages sent to the patient. |
    | 14 | No-show | True or False (Target variable). Indicates if the patient missed their appointment. |

## Technical Highlights

The approach to solving the challenge of predicting patient no-shows involved a comprehensive workflow, focusing on both the development of a predictive model and its practical application within an existing system. Here's an overview of the approach taken:

- **Data Storage and Initial Analysis**: Utilized Snowflake for secure data storage and conducted exploratory data analysis (EDA) to understand the dataset's characteristics and identify potential predictive features.

- **Data Loading and Preprocessing**: Initial steps involved loading the data from Snowflake, followed by preprocessing tasks such as handling missing values, encoding categorical variables, and normalizing features to prepare the dataset for modeling.

- **Feature Engineering and Selection**: Engineered meaningful features from the raw data, such as calculating the time interval between scheduling and appointment dates. The selection of features was based on their importance as determined through analysis using logistic regression and decision tree models, focusing on retaining features with at least 1% importance from either model.

- **Dataset and Model Selection**: Various machine learning algorithms, including Logistic Regression, Decision Tree, Random Forest, and XGBoost, were evaluated across different datasets (original, upsampled, downsampled, and SMOTE-enhanced) to identify the best-performing model. XGBoost emerged as the optimal choice, particularly when trained on the original dataset, after adjusting the `scale_pos_weight` parameter to address class imbalance effectively.

- **API Development for Model Deployment**: Created an API for the model, facilitating its integration into the client's ERP system. This step involved deploying the model to AWS SageMaker, setting up an AWS Lambda function for model invocation, and configuring an Amazon API Gateway to expose the model as a RESTful service.

- **Testing and Validation**: Conducted thorough testing of the deployed model using Postman, validating its functionality and ensuring its readiness for real-world application.

## Project Structure

The project is organized into several directories and files, each serving a specific purpose in the development, deployment, and documentation of the machine learning model.

Below is an overview of the project structure and the contents of each component:

```markdown
Medical-Appointment-No-Show-Prediction
├── data/
│   ├── input/                  # Raw data files.
│   ├── processed/              # Data files that have been cleaned and preprocessed.
│   ├── output/                 # Output data files, including model predictions.
│   ├── features/               # Contains the important features used for filtering the data.
│   └── hyperparameters/        # Contains the best hyperparameters obtained from Hyperopt tuning.
├── src/
│   ├── data_loader.py          # Script for loading and preprocessing data.
│   ├── preprocessing.py        # Script containing data preprocessing functions.
│   ├── feature_engineering.py  # Script for feature engineering tasks.
│   ├── modeling.py             # Contains model training, evaluation, and prediction scripts.
│   ├── train.py                # Main script for training the model.
│   ├── predict.py              # Script for making predictions using the trained model.
│   ├── requirements.txt        # Lists the Python dependencies required for the project.
│   └── snowflake_creds.py      # Contains credentials for Snowflake database access.
├── model/                      # Trained model files and artifacts.
├── deployment_assets/          # Files and scripts used for deploying the model.
├── Snowflake_assets/           # Original data file for database creation and SQL queries for exploratory analysis.
├── Project Notebook.ipynb      # Jupyter notebook detailing the model development process.
├── Project Documentation.pdf   # Comprehensive documentation of the project.
├── Model Deployment.ipynb      # Jupyter notebook detailing the model deployment process.
```

## Usage

1. **Clone the Repository**

    Clone the project repository to local machine.

    ```bash
    git clone https://github.com/TimKong21/Medical-Appointment-No-Show-Prediction.git
    cd Medical-Appointment-No-Show-Prediction
    ```

2. **Set Up a Virtual Environment**

   Create and activate a virtual environment to manage the project's dependencies.
   
   ```bash
   # Create a virtual environment
   python -m venv env
   ```
   Activate the virtual environment.
   ```bash
   # On Windows
   env\Scripts\activate

   # On MacOS/Linux
   source env/bin/activate
   ```
3. **Install Dependencies**

   Install the required Python dependencies.
   
   ```bash
   pip install -r src/requirements.txt
   ```

4. **Model Training**
   
   Train the model and make predictions.

   ```bash
   cd src
   python train.py
   python predict.py
   ```

5. **Model Deployment**
   
   For deploying the model to AWS SageMaker and setting up the necessary AWS services for model invocation and API exposure, follow step 1 to step 6 on  `Model Deployment.ipynb`. This notebook provides detailed steps for deploying the model to AWS SageMaker, creating an AWS Lambda function, and configuring an Amazon API Gateway to expose the model as a RESTful service.

6. **Testing and Validating with POSTMAN**
   
   After deployment, follow step 7 on `Model Deployment.ipynb` to test and validate the model's functionality using POSTMAN. This involves sending requests to the deployed model's API endpoint and verifying the responses to ensure the model operates as expected.

For a comprehensive understanding of the project, refer to:

- `Project Notebook.ipynb` for detail model development process.

- `Model Deployment.ipynb` for detail model deployment process.

- `Project Documentation.pdf` for comprehensive project documentation.