import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
import os

# Function to load and preprocess data
def load_and_preprocess_data():
    try:
        data = pd.read_csv('heart_disease.csv')
        print(f"Dataset shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

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

# Create ensemble model
def create_ensemble_model(X_train, y_train, X_test, y_test, categorical_cols, numerical_cols, preprocessor):
    st.write("Training ensemble model...")
    
    # Process the data
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert to dense arrays if sparse
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()
    
    # Apply SMOTE for imbalanced data
    st.write("Applying SMOTE for handling class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
    st.write(f"SMOTE applied - training data shape: {X_train_smote.shape}")
    
    # Feature selection using Random Forest
    st.write("Performing feature selection...")
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
    X_train_selected = selector.fit_transform(X_train_smote, y_train_smote)
    X_test_selected = selector.transform(X_test_processed)
    st.write(f"Selected features: {X_train_selected.shape[1]} out of {X_train_smote.shape[1]}")
    
    # Define model parameters
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
    
    # Create stacking ensemble
    estimators = [
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('rf', rf_model),
        ('gb', gb_model),
        ('mlp', mlp_model)
    ]
    
    meta_clf = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_clf,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    # Create voting ensemble
    voting = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=[2, 2, 1, 1, 1],
        n_jobs=-1
    )
    
    # Train and evaluate models
    models = {
        'XGBoost': xgb_model,
        'LightGBM': lgbm_model,
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'Neural Network': mlp_model
    }
    
    results = {}
    for name, model in models.items():
        st.write(f"Training {name}...")
        model.fit(X_train_selected, y_train_smote)
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        st.write(f"{name} accuracy: {accuracy:.4f}")
    
    # Train stacking ensemble
    st.write("Training stacking ensemble...")
    stacking.fit(X_train_selected, y_train_smote)
    stacking_pred = stacking.predict(X_test_selected)
    stacking_accuracy = accuracy_score(y_test, stacking_pred)
    results['Stacking Ensemble'] = stacking_accuracy
    st.write(f"Stacking ensemble accuracy: {stacking_accuracy:.4f}")
    
    # Train voting ensemble
    st.write("Training voting ensemble...")
    voting.fit(X_train_selected, y_train_smote)
    voting_pred = voting.predict(X_test_selected)
    voting_accuracy = accuracy_score(y_test, voting_pred)
    results['Voting Ensemble'] = voting_accuracy
    st.write(f"Voting ensemble accuracy: {voting_accuracy:.4f}")
    
    # Choose the best model
    best_model = stacking if stacking_accuracy >= voting_accuracy else voting
    best_score = max(stacking_accuracy, voting_accuracy)
    
    # Save models and preprocessors
    joblib.dump(best_model, "best_heart_disease_model.joblib")
    joblib.dump(preprocessor, "preprocessor.joblib")
    joblib.dump(selector, "feature_selector.joblib")
    joblib.dump(smote, "smote_processor.joblib")
    
    return best_model, preprocessor, selector, smote, best_score

# Function to make predictions
def predict_heart_disease(patient_data, model, preprocessor, selector):
    try:
        # Create DataFrame from patient data
        input_df = pd.DataFrame([patient_data])
        
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
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Streamlit app
def main():
    st.title("Heart Disease Prediction System")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Select Page", ["Train Model", "Make Prediction"])
    
    if page == "Train Model":
        st.header("Model Training")
        
        # Load and preprocess data
        data = load_and_preprocess_data()
        if data is None:
            return
        
        # Display dataset information
        st.subheader("Dataset Information")
        st.write("First 5 rows of the dataset:")
        st.write(data.head())
        
        st.write("\nData types:")
        st.write(data.dtypes)
        
        st.write("\nMissing values:")
        st.write(data.isnull().sum())
        
        # Handle missing values
        data = data.dropna()
        
        # Create preprocessor
        preprocessor, categorical_cols, numerical_cols = create_preprocessor(data)
        
        # Split the data
        X = data.drop("Heart Disease Status", axis=1)
        y = data["Heart Disease Status"].map({"Yes": 1, "No": 0})
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model button
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                best_model, preprocessor, selector, smote, best_score = create_ensemble_model(
                    X_train, y_train, X_test, y_test, categorical_cols, numerical_cols, preprocessor
                )
                st.success(f"Model trained successfully with accuracy: {best_score:.4f}")
    
    else:  # Make Prediction page
        st.header("Make Prediction")
        
        # Load models
        try:
            model = joblib.load('best_heart_disease_model.joblib')
            preprocessor = joblib.load('preprocessor.joblib')
            selector = joblib.load('feature_selector.joblib')
            smote = joblib.load('smote_processor.joblib')
        except Exception as e:
            st.error("Please train the model first before making predictions.")
            return
        
        # Create input form
        st.subheader("Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=50)
            gender = st.selectbox("Gender", ["Male", "Female"])
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", 
                                           min_value=0, max_value=300, value=120)
            cholesterol = st.number_input("Cholesterol Level (mg/dl)", 
                                        min_value=0, max_value=600, value=200)
            exercise = st.selectbox("Exercise Habits", ["Low", "Medium", "High"])
            smoking = st.selectbox("Smoking", ["Yes", "No"])
            family_history = st.selectbox("Family History of Heart Disease", ["Yes", "No"])
            diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        
        with col2:
            bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
            high_bp = st.selectbox("High Blood Pressure", ["Yes", "No"])
            low_hdl = st.selectbox("Low HDL Cholesterol", ["Yes", "No"])
            high_ldl = st.selectbox("High LDL Cholesterol", ["Yes", "No"])
            alcohol = st.selectbox("Alcohol Consumption", ["Low", "Medium", "High"])
            stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
            sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
            sugar = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"])
        
        # Additional inputs
        triglyceride = st.number_input("Triglyceride Level (mg/dl)", 
                                     min_value=0, max_value=1000, value=150)
        fasting_blood_sugar = st.number_input("Fasting Blood Sugar (mg/dl)", 
                                            min_value=0, max_value=300, value=100)
        crp = st.number_input("CRP Level (mg/L)", 
                             min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        homocysteine = st.number_input("Homocysteine Level (μmol/L)", 
                                     min_value=0.0, max_value=50.0, value=10.0, step=0.1)
        
        # Create patient data dictionary
        patient_data = {
            "Age": age,
            "Gender": gender,
            "Blood Pressure": blood_pressure,
            "Cholesterol Level": cholesterol,
            "Exercise Habits": exercise,
            "Smoking": smoking,
            "Family Heart Disease": family_history,
            "Diabetes": diabetes,
            "BMI": bmi,
            "High Blood Pressure": high_bp,
            "Low HDL Cholesterol": low_hdl,
            "High LDL Cholesterol": high_ldl,
            "Alcohol Consumption": alcohol,
            "Stress Level": stress,
            "Sleep Hours": sleep,
            "Sugar Consumption": sugar,
            "Triglyceride Level": triglyceride,
            "Fasting Blood Sugar": fasting_blood_sugar,
            "CRP Level": crp,
            "Homocysteine Level": homocysteine
        }
        
        # Make prediction
        if st.button("Predict Heart Disease Risk"):
            risk_score, risk_category = predict_heart_disease(patient_data, model, 
                                                            preprocessor, selector)
            
            if risk_score is not None:
                # Display results
                st.header("Prediction Results")
                
                # Create a progress bar for risk score
                st.progress(risk_score / 100)
                st.write(f"Risk Score: {risk_score:.1f}%")
                st.write(f"Risk Category: {risk_category}")
                
                # Display interpretation
                st.subheader("Interpretation")
                if risk_category == "Low Risk":
                    st.success("The patient is at low risk of heart disease. Regular check-ups are recommended.")
                elif risk_category == "Moderate Risk":
                    st.warning("The patient is at moderate risk of heart disease. Lifestyle changes and regular monitoring are recommended.")
                else:
                    st.error("The patient is at high risk of heart disease. Immediate medical attention and lifestyle changes are strongly recommended.")
                
                # Display feature importance
                st.subheader("Key Contributing Factors")
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = preprocessor.get_feature_names_out()
                    
                    # Create a DataFrame for feature importances
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    })
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    # Plot feature importance
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', 
                              data=importance_df.head(10))
                    plt.title('Top 10 Most Important Features')
                    st.pyplot(plt)
                else:
                    st.info("Feature importance information is not available for this model.")

if __name__ == "__main__":
    main() 