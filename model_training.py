import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pickle
import warnings
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.models import save_model
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Check if running in Google Colab
def is_colab():
    try:
        import google.colab
        return True
    except:
        return False

# Set up GPU if available
def setup_gpu():
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available. Using GPU acceleration.")
        # Set memory growth to avoid hogging all GPU memory
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except:
            print("Error setting memory growth")
    else:
        print("No GPU found. Using CPU for training.")

# Load the dataset
def load_data():
    # If in Colab, check if the file exists, if not prompt to upload
    if is_colab():
        from google.colab import files
        import io
        
        if not os.path.exists('heart_disease.csv'):
            print("Please upload the heart_disease.csv file")
            uploaded = files.upload()
            
            for fn in uploaded.keys():
                print(f'User uploaded file {fn} with length {len(uploaded[fn])} bytes')
                
                # Save the file to the local runtime
                with open('heart_disease.csv', 'wb') as f:
                    f.write(uploaded[fn])
    
    df = pd.read_csv('heart_disease.csv')
    # Check for missing values and handle them
    df = df.replace('', np.nan)
    df = df.fillna(df.median(numeric_only=True))
    return df

# Prepare the data
def prepare_data(df):
    # Target variable
    y = df['Heart Disease Status'].map({'Yes': 1, 'No': 0})
    
    # Features
    X = df.drop('Heart Disease Status', axis=1)
    
    # Split categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
    
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X, y, X_train, X_test, y_train, y_test, preprocessor

# Create a ResNet model for tabular data
def create_resnet_model(input_dim, trial):
    inputs = Input(shape=(input_dim,))
    
    # Number of residual blocks
    n_blocks = trial.suggest_int('n_blocks', 1, 3)
    
    # Initial dense layer
    x = Dense(trial.suggest_int('initial_units', 64, 256))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Residual blocks
    for i in range(n_blocks):
        # Shortcut connection
        shortcut = x
        
        # First layer in block
        x = Dense(trial.suggest_int(f'block_{i}_units_1', 32, 256))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Second layer in block
        x = Dense(trial.suggest_int(f'block_{i}_units_2', 32, 256))(x)
        x = BatchNormalization()(x)
        
        # Dropout
        dropout_rate = trial.suggest_float(f'dropout_{i}', 0.1, 0.5)
        x = Dropout(dropout_rate)(x)
        
        # Add shortcut
        x = tf.keras.layers.add([x, shortcut])
        x = Activation('relu')(x)
    
    # Output layer
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    # Compile model
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

# Create a DenseNet model for tabular data
def create_densenet_model(input_dim, trial):
    inputs = Input(shape=(input_dim,))
    
    # Number of dense blocks
    n_blocks = trial.suggest_int('n_blocks', 1, 3)
    
    # Initial dense layer
    x = Dense(trial.suggest_int('initial_units', 64, 256))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # DenseNet blocks
    for i in range(n_blocks):
        # Number of layers in dense block
        n_layers = trial.suggest_int(f'block_{i}_layers', 2, 4)
        
        # Dense block
        block_input = x
        concat_list = [block_input]
        
        for j in range(n_layers):
            # Dense layer
            units = trial.suggest_int(f'block_{i}_layer_{j}_units', 32, 128)
            layer_out = Dense(units)(x)
            layer_out = BatchNormalization()(layer_out)
            layer_out = Activation('relu')(layer_out)
            
            # Add to concatenation list
            concat_list.append(layer_out)
            
            # Update x for next layer (concatenate all previous outputs)
            x = tf.keras.layers.concatenate(concat_list)
        
        # Transition layer
        transition_units = trial.suggest_int(f'transition_{i}_units', 32, 128)
        x = Dense(transition_units)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Dropout
        dropout_rate = trial.suggest_float(f'dropout_{i}', 0.1, 0.5)
        x = Dropout(dropout_rate)(x)
    
    # Output layer
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    # Compile model
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

# Traditional ML models objective function for Optuna
def objective_ml(trial, X, y, preprocessor):
    # Define the model type
    classifier_name = trial.suggest_categorical('classifier', ['RandomForest', 'GradientBoosting', 'SVC'])
    
    if classifier_name == 'RandomForest':
        classifier = RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 5, 30),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
    elif classifier_name == 'GradientBoosting':
        classifier = GradientBoostingClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            random_state=42
        )
    else:
        classifier = SVC(
            C=trial.suggest_float('C', 0.1, 100.0, log=True),
            kernel=trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            gamma=trial.suggest_categorical('gamma', ['scale', 'auto']),
            probability=True,
            random_state=42
        )
    
    # Build the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Calculate the cross-validation score
    scores = cross_val_score(pipeline, X, y, cv=3, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Deep Learning models objective function for Optuna
def objective_dl(trial, X_train, X_test, y_train, y_test, preprocessor, model_type='ResNet'):
    # Process the data
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get the input dimension
    input_dim = X_train_processed.shape[1]
    
    # Create the model
    if model_type == 'ResNet':
        model = create_resnet_model(input_dim, trial)
    else:  # DenseNet
        model = create_densenet_model(input_dim, trial)
    
    # Early stopping and learning rate reduction
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
    
    # Train the model
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    history = model.fit(
        X_train_processed, y_train,
        epochs=100,
        batch_size=batch_size,
        validation_data=(X_test_processed, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    # Evaluate the model
    y_pred_proba = model.predict(X_test_processed)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Return the accuracy as the objective
    return accuracy

# Train the best model
def train_best_model(X, y, X_train, X_test, y_train, y_test, best_params, preprocessor, model_type='ml'):
    if model_type == 'ml':
        classifier_name = best_params['classifier']
        
        if classifier_name == 'RandomForest':
            classifier = RandomForestClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                bootstrap=best_params['bootstrap'],
                random_state=42,
                n_jobs=-1
            )
        elif classifier_name == 'GradientBoosting':
            classifier = GradientBoostingClassifier(
                n_estimators=best_params['n_estimators'],
                learning_rate=best_params['learning_rate'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                subsample=best_params['subsample'],
                random_state=42
            )
        else:
            classifier = SVC(
                C=best_params['C'],
                kernel=best_params['kernel'],
                gamma=best_params['gamma'],
                probability=True,
                random_state=42
            )
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Best model: {classifier_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)
        
        # Save the trained model
        with open('heart_disease_model.pkl', 'wb') as model_file:
            pickle.dump(pipeline, model_file)
        
        # Save feature importance if available
        feature_importance = None
        feature_names = None
        
        if hasattr(classifier, 'feature_importances_'):
            # For tree-based models
            preprocessor.fit(X)
            feature_names = get_feature_names(preprocessor, X.columns)
            feature_importance = classifier.feature_importances_
            
            # Save feature importance
            feature_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)
            
            feature_imp_df.to_csv('feature_importance.csv', index=False)
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            plt.barh(feature_imp_df['Feature'][:15], feature_imp_df['Importance'][:15])
            plt.xlabel('Importance')
            plt.title('Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            if is_colab():
                plt.show()
        
        return pipeline, accuracy, auc, feature_importance, feature_names
    
    else:  # Deep learning model (ResNet or DenseNet)
        # Process the data
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get the input dimension
        input_dim = X_train_processed.shape[1]
        
        # Create the model
        if model_type == 'ResNet':
            model = create_resnet_model(input_dim, best_params)
        else:  # DenseNet
            model = create_densenet_model(input_dim, best_params)
        
        # Early stopping and learning rate reduction
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
        
        # Train the model
        batch_size = best_params['batch_size']
        
        history = model.fit(
            X_train_processed, y_train,
            epochs=100,
            batch_size=batch_size,
            validation_data=(X_test_processed, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        if is_colab():
            plt.show()
        
        # Evaluate the model
        y_pred_proba = model.predict(X_test_processed)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Best model: {model_type}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)
        
        # Save the trained model
        save_model(model, 'heart_disease_model_dl')
        
        # Save the preprocessor
        with open('preprocessor.pkl', 'wb') as preprocessor_file:
            pickle.dump(preprocessor, preprocessor_file)
        
        # Create a feature importance proxy (using permutation importance or other techniques)
        # For simplicity, we'll use a simple correlation-based approach here
        feature_imp_df = compute_feature_correlation(X, y)
        feature_imp_df.to_csv('feature_importance.csv', index=False)
        
        return model, accuracy, auc, None, None

def compute_feature_correlation(X, y):
    """Compute correlation between features and target as a proxy for importance"""
    # Convert to numeric
    X_numeric = X.copy()
    for col in X_numeric.columns:
        if X_numeric[col].dtype == 'object':
            # One-hot encode categorical features
            dummies = pd.get_dummies(X_numeric[col], prefix=col, drop_first=True)
            X_numeric = pd.concat([X_numeric.drop(col, axis=1), dummies], axis=1)
    
    # Compute correlation with target
    X_numeric['target'] = y
    corr = X_numeric.corr()['target'].sort_values(ascending=False)
    corr = corr[corr.index != 'target'].abs()
    
    # Create feature importance dataframe
    feature_imp_df = pd.DataFrame({
        'Feature': corr.index,
        'Importance': corr.values
    }).sort_values(by='Importance', ascending=False)
    
    return feature_imp_df

def get_feature_names(column_transformer, original_feature_names):
    """Get feature names from all transformers."""
    output_features = []
    
    for name, pipe, features in column_transformer.transformers_:
        if name != 'remainder':
            if hasattr(pipe, 'get_feature_names_out'):
                feature_names = pipe.get_feature_names_out()
            else:
                # If the transformer doesn't have get_feature_names_out, use the original feature names
                feature_names = np.array(features)
            
            output_features.extend(feature_names)
    
    return np.array(output_features)

def main():
    # Setup GPU if available
    setup_gpu()
    
    # Display information about environment
    if is_colab():
        print("Running in Google Colab")
        # Mount Google Drive for saving models if needed
        from google.colab import drive
        mount_drive = input("Do you want to mount Google Drive to save models? (y/n): ").lower() == 'y'
        if mount_drive:
            drive.mount('/content/drive')
            save_path = '/content/drive/My Drive/HeartDiseaseModel/'
            os.makedirs(save_path, exist_ok=True)
            print(f"Models will be saved to {save_path}")
    
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_data()
    X, y, X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    
    # Ask user which model type to optimize
    if is_colab():
        model_choices = ['ml', 'ResNet', 'DenseNet']
        for i, model in enumerate(model_choices):
            print(f"{i+1}. {model}")
        
        choice = int(input("Select model type to optimize (1-3): "))
        model_choice = model_choices[choice-1]
    else:
        model_choice = input("Select model type to optimize (ml/ResNet/DenseNet): ").strip()
    
    # Number of trials
    n_trials = int(input("Enter number of Optuna trials (recommended: 20-50): "))
    
    # Start timing
    start_time = time.time()
    
    # Create a study object and optimize the objective function
    print(f"\nStarting Optuna optimization for {model_choice} model with {n_trials} trials...")
    study = optuna.create_study(direction='maximize')
    
    if model_choice == 'ml':
        study.optimize(lambda trial: objective_ml(trial, X, y, preprocessor), n_trials=n_trials)
    else:  # Deep learning models
        study.optimize(
            lambda trial: objective_dl(trial, X_train, X_test, y_train, y_test, preprocessor, model_type=model_choice),
            n_trials=n_trials
        )
    
    # Print the best parameters
    print("\nOptimization completed!")
    print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Train the best model
    print("\nTraining the best model...")
    best_model, accuracy, auc, feature_importance, feature_names = train_best_model(
        X, y, X_train, X_test, y_train, y_test, trial.params, preprocessor, model_type=model_choice
    )
    
    # Save the Optuna study
    with open('optuna_study.pkl', 'wb') as study_file:
        pickle.dump(study, study_file)
    
    print(f"\nModel saved with accuracy: {accuracy:.4f} and AUC: {auc:.4f}")
    
    # If in Colab, provide download link for the model
    if is_colab():
        from google.colab import files
        
        print("\nDownloading files to your computer...")
        if model_choice == 'ml':
            files.download('heart_disease_model.pkl')
        else:
            # Create a zip file of the model directory
            import shutil
            shutil.make_archive('heart_disease_model_dl', 'zip', 'heart_disease_model_dl')
            files.download('heart_disease_model_dl.zip')
        
        files.download('feature_importance.csv')
        files.download('feature_importance.png')
        files.download('optuna_study.pkl')
        if model_choice != 'ml':
            files.download('training_history.png')
            files.download('preprocessor.pkl')

if __name__ == '__main__':
    main() 