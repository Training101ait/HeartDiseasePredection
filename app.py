# Heart Disease Prediction using Ensemble Models
# This file is designed to run locally on your laptop

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import joblib
import os
import requests
from io import BytesIO
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

# Function to load the dataset
def load_dataset(file_path='heart_disease.csv'):
    try:
        # Try to load from the provided path
        print(f"Attempting to load dataset from {file_path}")
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print("Creating a sample dataset for demonstration...")
        
        # Create a synthetic dataset for demonstration purposes
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic data
        age = np.random.randint(30, 80, n_samples)
        gender = np.random.choice(['Male', 'Female'], n_samples)
        chest_pain = np.random.choice(['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'], n_samples)
        blood_pressure = np.random.randint(90, 200, n_samples)
        cholesterol = np.random.randint(120, 400, n_samples)
        blood_sugar = np.random.choice(['> 120 mg/dl', '<= 120 mg/dl'], n_samples)
        ecg = np.random.choice(['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'], n_samples)
        max_heart_rate = np.random.randint(60, 202, n_samples)
        exercise_angina = np.random.choice(['Yes', 'No'], n_samples)
        st_depression = np.random.uniform(0, 6.2, n_samples)
        slope = np.random.choice(['Upsloping', 'Flat', 'Downsloping'], n_samples)
        vessels = np.random.choice(['0', '1', '2', '3'], n_samples)
        thalassemia = np.random.choice(['Normal', 'Fixed defect', 'Reversible defect'], n_samples)
        
        # Create features
        features = {
            'Age': age,
            'Sex': gender,
            'Chest Pain Type': chest_pain,
            'Resting Blood Pressure': blood_pressure,
            'Serum Cholesterol': cholesterol,
            'Fasting Blood Sugar': blood_sugar,
            'Resting ECG': ecg,
            'Max Heart Rate': max_heart_rate,
            'Exercise Induced Angina': exercise_angina,
            'ST Depression': st_depression,
            'ST Slope': slope,
            'Number of Major Vessels': vessels,
            'Thalassemia': thalassemia
        }
        
        # Calculate probability of heart disease based on features
        prob = 1 / (1 + np.exp(-(
            0.05 * (age - 50) + 
            0.8 * (gender == 'Male') + 
            0.7 * (chest_pain == 'Typical Angina') + 
            0.01 * (blood_pressure - 120) +
            0.005 * (cholesterol - 200) +
            0.6 * (blood_sugar == '> 120 mg/dl') +
            0.5 * (ecg != 'Normal') +
            -0.01 * (max_heart_rate - 150) +
            0.9 * (exercise_angina == 'Yes') +
            0.3 * st_depression +
            0.4 * (slope == 'Downsloping') +
            0.6 * (vessels != '0') +
            0.7 * (thalassemia != 'Normal')
        )))
        
        # Assign heart disease status
        heart_disease = np.random.binomial(1, prob)
        features['Heart Disease Status'] = ['Yes' if hd == 1 else 'No' for hd in heart_disease]
        
        # Create DataFrame
        data = pd.DataFrame(features)
        print("Sample dataset created!")
        
        # Save the synthetic dataset
        data.to_csv('heart_disease.csv', index=False)
        print("Sample dataset saved to 'heart_disease.csv'")
        
        return data

# Create preprocessor
def create_preprocessor(data):
    # Identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object']).drop('Heart Disease Status', axis=1).columns.tolist()
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor, categorical_cols, numerical_cols

# Evaluate model function
def evaluate_model(model, X_test, y_test, preprocessor, show_plots=True):
    # Generate predictions
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    if show_plots:
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    
    return accuracy, y_pred_prob

# Make prediction for a single patient
def predict_patient(model, preprocessor, selector, patient_data):
    """
    Make a prediction for a single patient.
    
    Args:
        model: The trained machine learning model
        preprocessor: The data preprocessor
        selector: The feature selector
        patient_data: Dictionary with patient information
        
    Returns:
        risk_score: Percentage risk of heart disease
        risk_category: Category of risk (Low, Moderate, High)
    """
    # Create DataFrame from patient data
    input_data = {}
    
    # Populate patient data (handle both original and engineered features)
    for col in patient_data.keys():
        input_data[col] = [patient_data[col]]
    
    input_df = pd.DataFrame(input_data)
    
    # Process numerical features if needed
    numerical_cols_orig = input_df.select_dtypes(include=['number']).columns.tolist()
    
    # Generate engineered features
    # Log transformations
    for col in numerical_cols_orig:
        if f"{col}_log" in patient_data:
            input_df[f"{col}_log"] = np.log1p(input_df[col])
        elif min(input_df[col]) > 0:  # Only if positive data
            input_df[f"{col}_log"] = np.log1p(input_df[col])
    
    # Polynomial features
    for col in numerical_cols_orig:
        if f"{col}_squared" in patient_data:
            input_df[f"{col}_squared"] = input_df[col] ** 2
        else:
            input_df[f"{col}_squared"] = input_df[col] ** 2
    
    # Create interaction features between numerical columns
    if len(numerical_cols_orig) >= 2:
        for i in range(len(numerical_cols_orig)):
            for j in range(i+1, len(numerical_cols_orig)):
                col1 = numerical_cols_orig[i]
                col2 = numerical_cols_orig[j]
                input_df[f"{col1}_times_{col2}"] = input_df[col1] * input_df[col2]
                input_df[f"{col1}_plus_{col2}"] = input_df[col1] + input_df[col2]
                input_df[f"{col1}_div_{col2}"] = input_df[col1] / (input_df[col2] + 1e-5)  # Avoid division by zero
                input_df[f"{col1}_minus_{col2}"] = input_df[col1] - input_df[col2]
    
    # Preprocess the input data
    input_processed = preprocessor.transform(input_df)
    if hasattr(input_processed, 'toarray'):
        input_processed = input_processed.toarray()
    
    # Apply feature selection
    input_selected = selector.transform(input_processed)
    
    # Make prediction
    if hasattr(model, 'predict_proba'):
        prediction = model.predict_proba(input_selected)[0][1]
    else:
        prediction = model.predict(input_selected)[0]
    
    risk_score = float(prediction) * 100
    
    # Determine risk category
    if risk_score < 20:
        risk_category = "Low Risk"
    elif risk_score < 50:
        risk_category = "Moderate Risk"
    else:
        risk_category = "High Risk"
    
    return risk_score, risk_category

# Create the ensemble model for better performance
def create_ensemble_model(X_train, y_train, X_test, y_test, categorical_cols, numerical_cols, preprocessor):
    print("Training ensemble model...")
    
    # Process the data
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert to dense arrays if sparse
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()
    
    # Apply SMOTE for imbalanced data
    print("Applying SMOTE for handling class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
    print(f"SMOTE applied - training data shape: {X_train_smote.shape}")
    
    # Feature selection using Random Forest
    print("Performing feature selection...")
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
    X_train_selected = selector.fit_transform(X_train_smote, y_train_smote)
    X_test_selected = selector.transform(X_test_processed)
    print(f"Selected features: {X_train_selected.shape[1]} out of {X_train_smote.shape[1]}")
    
    # Define the best parameters for each model based on optuna optimization
    xgb_params = {
        'learning_rate': 0.05,
        'max_depth': 6,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 3,
        'scale_pos_weight': 2.0,
        'random_state': 42
    }
    
    lgbm_params = {
        'learning_rate': 0.05,
        'n_estimators': 500,
        'num_leaves': 31,
        'max_depth': 10,
        'min_data_in_leaf': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42
    }
    
    rf_params = {
        'n_estimators': 500,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42
    }
    
    gb_params = {
        'learning_rate': 0.05,
        'n_estimators': 500,
        'max_depth': 6,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'subsample': 0.8,
        'random_state': 42
    }
    
    mlp_params = {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'batch_size': 'auto',
        'learning_rate': 'adaptive',
        'max_iter': 1000,
        'early_stopping': True,
        'random_state': 42
    }
    
    # Create classifiers
    xgb_model = XGBClassifier(**xgb_params)
    lgbm_model = LGBMClassifier(**lgbm_params)
    rf_model = RandomForestClassifier(**rf_params)
    gb_model = GradientBoostingClassifier(**gb_params)
    mlp_model = MLPClassifier(**mlp_params)
    
    # Create a stacking classifier
    estimators = [
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('rf', rf_model),
        ('gb', gb_model),
        ('mlp', mlp_model)
    ]
    
    # Use a more advanced meta-classifier for stacking
    meta_clf = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    
    # Create stacking ensemble
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_clf,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    # Create voting ensemble as backup
    voting = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=[2, 2, 1, 1, 1],
        n_jobs=-1
    )
    
    # Train individual models and evaluate
    models = {
        'XGBoost': xgb_model,
        'LightGBM': lgbm_model,
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'Neural Network': mlp_model
    }
    
    best_score = 0
    best_model = None
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_selected, y_train_smote)
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name} accuracy: {accuracy:.4f}")
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
    
    # Train stacking ensemble
    print("Training stacking ensemble...")
    stacking.fit(X_train_selected, y_train_smote)
    stacking_pred = stacking.predict(X_test_selected)
    stacking_accuracy = accuracy_score(y_test, stacking_pred)
    results['Stacking Ensemble'] = stacking_accuracy
    print(f"Stacking ensemble accuracy: {stacking_accuracy:.4f}")
    
    # Train voting ensemble
    print("Training voting ensemble...")
    voting.fit(X_train_selected, y_train_smote)
    voting_pred = voting.predict(X_test_selected)
    voting_accuracy = accuracy_score(y_test, voting_pred)
    results['Voting Ensemble'] = voting_accuracy
    print(f"Voting ensemble accuracy: {voting_accuracy:.4f}")
    
    # Choose the best performing model
    if stacking_accuracy > best_score and stacking_accuracy >= voting_accuracy:
        best_model = stacking
        best_score = stacking_accuracy
        print("Selected stacking ensemble as the best model")
    elif voting_accuracy > best_score and voting_accuracy > stacking_accuracy:
        best_model = voting
        best_score = voting_accuracy
        print("Selected voting ensemble as the best model")
    else:
        best_name = max(results.items(), key=lambda x: x[1])[0]
        print(f"Selected {best_name} as the best model")
    
    # Plot model performance comparison
    plt.figure(figsize=(12, 6))
    plt.bar(results.keys(), results.values())
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.show()
    
    # Get feature importances from the best model if available
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'feature_importances_') and best_model == stacking:
        importances = best_model.final_estimator_.coef_[0]
    elif hasattr(best_model, 'estimators_'):
        # For voting classifier, get average feature importance from models that support it
        importances = np.zeros(X_train_selected.shape[1])
        count = 0
        for name, model in best_model.named_estimators_.items():
            if hasattr(model, 'feature_importances_'):
                importances += model.feature_importances_
                count += 1
        if count > 0:
            importances /= count
    else:
        # If feature importances not available, use permutation importance
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(best_model, X_test_selected, y_test, n_repeats=10, random_state=42)
        importances = perm_importance.importances_mean
    
    if 'importances' in locals():
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        indices = np.argsort(importances)[::-1]
        plt.bar(range(min(20, len(importances))), importances[indices[:20]])
        plt.title('Top 20 Feature Importances')
        plt.xticks(range(min(20, len(importances))), range(1, min(20, len(importances))+1))
        plt.xlabel('Feature Rank')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()
    
    # Create detailed evaluation of the best model
    print("\nDetailed evaluation of the best model:")
    y_pred = best_model.predict(X_test_selected)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # ROC Curve
    if hasattr(best_model, 'predict_proba'):
        y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()
    
    # Save the best model
    joblib.dump(best_model, "best_heart_disease_model.joblib")
    
    # Save feature selector and SMOTE
    joblib.dump(selector, "feature_selector.joblib")
    joblib.dump(smote, "smote_processor.joblib")
    joblib.dump(preprocessor, "preprocessor.joblib")
    
    print(f"Best model saved with accuracy: {best_score:.4f}")
    
    return best_model, selector, smote, best_score, X_train_selected.shape[1]

# Main function to run the entire pipeline
def main():
    print("Starting Heart Disease Prediction System...")
    
    # Load dataset
    print("\n1. Loading Dataset")
    data = load_dataset()
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Display dataset information
    print("\nFirst 5 rows of the dataset:")
    print(data.head())
    
    print("\nData types:")
    print(data.dtypes)
    
    print("\nMissing values:")
    print(data.isnull().sum())
    
    # Handle missing values if any
    data = data.dropna()
    
    # Exploratory Data Analysis
    print("\n2. Performing Exploratory Data Analysis")
    
    # Count of heart disease status
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Heart Disease Status', data=data)
    plt.title('Heart Disease Status Distribution')
    plt.show()
    
    # Age distribution by heart disease status
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Age', hue='Heart Disease Status', multiple='stack')
    plt.title('Age Distribution by Heart Disease Status')
    plt.show()
    
    # Correlation heatmap for numerical features
    plt.figure(figsize=(12, 10))
    numerical_data = data.select_dtypes(include=['number'])
    if 'Heart Disease Status' in data.columns:
        numerical_data['Target'] = data['Heart Disease Status'].map({'Yes': 1, 'No': 0})
    
    sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Prepare data for modeling
    print("\n3. Preparing Data for Modeling")
    
    # Extract features and target
    X = data.drop("Heart Disease Status", axis=1)
    y = data["Heart Disease Status"].map({"Yes": 1, "No": 0})
    
    # Feature Engineering - Add feature interactions for numerical columns
    print("Performing feature engineering...")
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    # Log transformations for skewed features
    for col in numerical_cols:
        skewness = X[col].skew()
        if abs(skewness) > 0.5:  # If moderately skewed
            if min(X[col]) > 0:  # Only apply log to positive data
                X[f"{col}_log"] = np.log1p(X[col])
    
    # Polynomial features for numerical columns
    for col in numerical_cols:
        X[f"{col}_squared"] = X[col] ** 2
    
    # Interaction features
    if len(numerical_cols) >= 2:
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                col1 = numerical_cols[i]
                col2 = numerical_cols[j]
                X[f"{col1}_times_{col2}"] = X[col1] * X[col2]
                X[f"{col1}_plus_{col2}"] = X[col1] + X[col2]
                X[f"{col1}_div_{col2}"] = X[col1] / (X[col2] + 1e-5)  # Avoid division by zero
                X[f"{col1}_minus_{col2}"] = X[col1] - X[col2]
    
    # Create preprocessor
    preprocessor, categorical_cols, numerical_cols = create_preprocessor(data)
    
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Fit the preprocessor on training data
    print("Fitting preprocessor on training data...")
    preprocessor.fit(X_train)
    
    # Check if model is already trained and saved
    if os.path.exists("best_heart_disease_model.joblib") and \
       os.path.exists("preprocessor.joblib") and \
       os.path.exists("feature_selector.joblib") and \
       os.path.exists("smote_processor.joblib"):
        print("\nLoading existing model and components...")
        best_model = joblib.load("best_heart_disease_model.joblib")
        selector = joblib.load("feature_selector.joblib")
        smote = joblib.load("smote_processor.joblib")
        
        # Evaluate the loaded model
        X_test_processed = preprocessor.transform(X_test)
        if hasattr(X_test_processed, 'toarray'):
            X_test_processed = X_test_processed.toarray()
        
        X_test_selected = selector.transform(X_test_processed)
        y_pred = best_model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Loaded model accuracy: {accuracy:.4f}")
        best_score = accuracy
        n_features = X_test_selected.shape[1]
    else:
        # Train and evaluate ensemble model
        print("\nTraining new model...")
        best_model, selector, smote, best_score, n_features = create_ensemble_model(
            X_train, y_train, X_test, y_test, categorical_cols, numerical_cols, preprocessor
        )
    
    # Interactive prediction section
    print("\n4. Interactive Prediction")
    print("\nNow you can make predictions for individual patients.")
    
    # Sample patient data
    sample_patient = {col: X_test.iloc[0][col] for col in X_test.columns}
    
    print("\nSample patient data:")
    for key, value in sample_patient.items():
        if key in data.columns:  # Only show original features, not engineered ones
            print(f"{key}: {value}")
    
    # Get risk score for sample patient
    risk_score, risk_category = predict_patient(best_model, preprocessor, selector, sample_patient)
    
    print(f"\nPrediction for sample patient: {risk_score:.1f}% - {risk_category}")
    
    # Instructions for using the models
    print("\nInstructions for using this model:")
    print("1. The ensemble model has been trained with multiple algorithms")
    print(f"2. The best model achieved {best_score:.2%} accuracy")
    print(f"3. The model uses {n_features} selected features from the original dataset")
    print("4. The following files have been saved for deployment:")
    print("   - best_heart_disease_model.joblib")
    print("   - preprocessor.joblib")
    print("   - feature_selector.joblib")
    print("   - smote_processor.joblib")
    
    print("\nTo use this model in your application, call the predict_patient() function.")
    
    return best_model, preprocessor, selector, smote

# Example of a custom input function
def custom_patient_input():
    """Allow user to input patient data manually"""
    print("\nEnter patient information:")
    
    # Get categorical features
    gender = input("Gender (Male/Female): ")
    chest_pain = input("Chest Pain Type (Typical Angina/Atypical Angina/Non-anginal Pain/Asymptomatic): ")
    blood_sugar = input("Fasting Blood Sugar (> 120 mg/dl/<= 120 mg/dl): ")
    ecg = input("Resting ECG (Normal/ST-T wave abnormality/Left ventricular hypertrophy): ")
    exercise_angina = input("Exercise Induced Angina (Yes/No): ")
    slope = input("ST Slope (Upsloping/Flat/Downsloping): ")
    vessels = input("Number of Major Vessels (0/1/2/3): ")
    thalassemia = input("Thalassemia (Normal/Fixed defect/Reversible defect): ")
    
    # Get numerical features
    age = float(input("Age: "))
    blood_pressure = float(input("Resting Blood Pressure (mmHg): "))
    cholesterol = float(input("Serum Cholesterol (mg/dl): "))
    max_heart_rate = float(input("Max Heart Rate: "))
    st_depression = float(input("ST Depression: "))
    
    # Create patient data dictionary
    patient_data = {
        'Age': age,
        'Sex': gender,
        'Chest Pain Type': chest_pain,
        'Resting Blood Pressure': blood_pressure,
        'Serum Cholesterol': cholesterol,
        'Fasting Blood Sugar': blood_sugar,
        'Resting ECG': ecg,
        'Max Heart Rate': max_heart_rate,
        'Exercise Induced Angina': exercise_angina,
        'ST Depression': st_depression,
        'ST Slope': slope,
        'Number of Major Vessels': vessels,
        'Thalassemia': thalassemia
    }
    
    return patient_data

# Execute the main function if this file is run directly
if __name__ == "__main__":
    # Train or load the model and make sample prediction
    best_model, preprocessor, selector, smote = main()
    
    # Ask user if they want to make a custom prediction
    while True:
        choice = input("\nWould you like to make a prediction with custom data? (yes/no): ").lower()
        if choice in ['y', 'yes']:
            patient_data = custom_patient_input()
            risk_score, risk_category = predict_patient(best_model, preprocessor, selector, patient_data)
            print(f"\nPrediction result: {risk_score:.1f}% - {risk_category}")
        elif choice in ['n', 'no']:
            print("Thank you for using the Heart Disease Prediction System.")
            break
        else:
            print("Invalid choice. Please enter 'yes' or 'no'.") 