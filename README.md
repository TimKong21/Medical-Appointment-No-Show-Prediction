# Predicting Patient No-Shows in Healthcare Appointments

## Business Problem

A significant issue in medical setting is patients failing to attend scheduled doctor appointments despite receiving instructions (no-shows). Our client, a medical ERP solutions provider, seeks to tackle this by introducing a machine learning model into their software. This model aims to predict patient attendance, enabling medical providers to optimize appointment management.

<p align="left">
    <img src="Notebook_images/Patient no show.png" alt="Image" style="width: 90%; height: 25%;" />
</p>

## Dataset Description

The dataset from [Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments) utilized in this project comprises appointment records from medical institutions, capturing various attributes related to patients and their appointments. Key features include:
- **Patient demographics**: age and gender
- **Health characteristics**: the presence of conditions such as diabetes or hypertension
- **Appointment-specific details**: scheduled and appointment dates, and whether the patient received a reminder SMS
- **Target**: binary indicator representing whether a patient was a no-show or attended their appointment.

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

This rich dataset provides a comprehensive view of factors potentially influencing patient attendance, enabling the development of a nuanced predictive model.

## Solution Approach

In addressing the challenge of predicting patient no-shows for healthcare appointments, our solution approach focuses on both the technical development of a predictive model and its practical integration into the client's existing systems.
Here's how we tackled the problem:

- **Model Development:** Created a machine learning model to assess the likelihood of patient no-shows, enhancing appointment scheduling efficiency.
- **System Integration:** Deployed the model with an API for integration into the client's ERP system, this allows real-time predictions, streamlining the ERP's existing workflow.

<p align="center">
    <img src="Notebook_images/High level structure.png" alt="Image" style="width: 100%; height: 25%;" />
</p>

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
├── Snowflake_assets/           # Original data file for database creation and SQL queries for EA.
├── Notebook_images/            # Images used in the Model Deployment.ipynb notebook.
├── Project Notebook.ipynb      # Jupyter notebook detailing the model development process.
├── Project Documentation.pdf   # Comprehensive documentation of the project.
├── Model Deployment.ipynb      # Jupyter notebook detailing the model deployment process.
└── README.md                   # Project information and instructions.
```

## Usage

To get started with this project and replicate the results or deploy the model, follow the steps outlined below:
- To train the model locally, first set up the project environment, install the required Python dependencies:
    ```bash
    pip install -r src/requirements.txt
    ```

    Then, run the following scripts:

    ```bash
    python src/train.py
    python src/predict.py
    ```

- To train model on AWS Sagemaker, upload all project assets to the workspace, and run `Code Modularization.ipynb`.
- For throughout model building process, refer `project notebook.ipynb`.
- For detail model deployment steps, refer `model deployment.ipynb`.
- For comprehensive project overview, please refer to `Project Documentation.pdf`.