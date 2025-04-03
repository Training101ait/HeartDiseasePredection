import os
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import sklearn.metrics

# Add the current directory to sys.path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Set memory growth to avoid OOM errors - must be done before any other TF operations
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memory growth enabled")
    except:
        print("Memory growth already enabled or device already initialized")

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna

# Import the check_gpu function
try:
    from check_gpu import check_gpu
except ImportError:
    # Define a fallback check_gpu function if import fails
    def check_gpu():
        print("TensorFlow version:", tf.__version__)
        print("\nGPU Available:", tf.config.list_physical_devices('GPU'))
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print("\nGPU Device Name:", tf.test.gpu_device_name())
        else:
            print("\nNo GPU devices found. TensorFlow is using CPU only.")

# First check if GPU is available
print("Checking GPU availability:")
check_gpu()

# Load the dataset
def load_data(file_path=None):
    if file_path is None:
        # Try to find the dataset in common locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)
        
        possible_locations = [
            os.path.join(script_dir, "heart_disease.csv"),
            os.path.join(base_dir, "heart_disease.csv"),
            os.path.join(base_dir, "heart_disease", "heart_disease.csv"),
            "heart_disease.csv"
        ]
    else:
        possible_locations = [file_path]
    
    data = None
    for loc in possible_locations:
        try:
            print(f"Trying to load data from: {loc}")
            if os.path.exists(loc):
                data = pd.read_csv(loc)
                print(f"Successfully loaded data from: {loc}")
                break
            else:
                print(f"File not found at: {loc}")
        except Exception as e:
            print(f"Error loading from {loc}: {str(e)}")
            continue
    
    if data is None:
        raise FileNotFoundError(f"Could not find the dataset file in any of the expected locations: {possible_locations}")
    
    print(f"Dataset shape: {data.shape}")
    print(f"First few rows:\n{data.head()}")
    print(f"Column names: {data.columns.tolist()}")
    
    # Handle missing values
    missing_values = data.isnull().sum()
    print(f"Missing values before handling:\n{missing_values[missing_values > 0]}")
    data = data.dropna()
    print(f"Dataset shape after dropping missing values: {data.shape}")
    
    # Extract features and target
    X = data.drop("Heart Disease Status", axis=1)
    y = data["Heart Disease Status"].map({"Yes": 1, "No": 0})
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Calculate class weights for imbalanced dataset
    class_counts = y.value_counts()
    total_samples = len(y)
    class_weights = {
        0: total_samples / (2 * class_counts[0]),
        1: total_samples / (2 * class_counts[1])
    }
    print(f"Class weights: {class_weights}")
    
    # Split the data with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Training class distribution: {y_train.value_counts().to_dict()}")
    print(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test, X.columns, class_weights

# Preprocess the data
def preprocess_data(X_train, X_test, feature_columns):
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    
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
    
    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # If categorical data was one-hot encoded, it'll be a sparse matrix
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()
    
    return X_train_processed, X_test_processed, preprocessor

# Define DenseNet block
def dense_block(x, filters, kernel_size=3, growth_rate=32, num_layers=4, activation='relu', l2_reg=0.001):
    concat_feat = x
    
    for i in range(num_layers):
        x = layers.Dense(growth_rate, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(concat_feat)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        
        # Concatenate with previous features
        concat_feat = layers.Concatenate()([concat_feat, x])
    
    return concat_feat

# Create the DenseNet model
def create_densenet_model(input_shape, n_blocks=3, initial_filters=64, growth_rate=32, 
                          layers_per_block=4, activation='relu', dropout_rate=0.5, l2_reg=0.001):
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(initial_filters, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    # Add dense blocks
    for i in range(n_blocks):
        x = dense_block(x, initial_filters * (2 ** min(i, 2)), growth_rate=growth_rate, 
                        num_layers=layers_per_block, activation=activation, l2_reg=l2_reg)
        
        # Add transition layer except after the last block
        if i < n_blocks - 1:
            x = layers.Dense(x.shape[-1] // 2, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
    
    # Global average pooling and output layer
    x = layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    
    model = models.Model(inputs, outputs)
    return model

# Optuna objective function for DenseNet
def objective_densenet(trial, X_train, y_train, X_val, y_val, input_shape, class_weights):
    # Define hyperparameters to optimize
    n_blocks = trial.suggest_int('n_blocks', 2, 5)
    initial_filters = trial.suggest_categorical('initial_filters', [32, 64, 128])
    growth_rate = trial.suggest_categorical('growth_rate', [16, 32, 64])
    layers_per_block = trial.suggest_int('layers_per_block', 2, 6)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)
    l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
    
    # Create and compile the model
    model = create_densenet_model(
        input_shape, 
        n_blocks=n_blocks, 
        initial_filters=initial_filters,
        growth_rate=growth_rate,
        layers_per_block=layers_per_block,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    
    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=50,  # Reduced for faster trials
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        class_weight=class_weights,
        verbose=0
    )
    
    # Evaluate the model - optimize for ROC AUC instead of accuracy
    val_metrics = model.evaluate(X_val, y_val, verbose=0)
    val_auc = val_metrics[2]  # AUC is the third metric
    return val_auc

# Main function
def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_columns, class_weights = load_data()
    
    # Split train data into train and validation sets with stratification
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # Preprocess the data
    X_train_processed, X_val_processed, preprocessor = preprocess_data(X_train, X_val, feature_columns)
    X_test_processed = preprocessor.transform(X_test)
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()
    
    print(f"Training data shape: {X_train_processed.shape}")
    input_shape = X_train_processed.shape[1]
    
    # Create directory for model saving
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models will be saved to: {models_dir}")
    
    # Optimize DenseNet model with Optuna
    print("\nOptimizing DenseNet model...")
    densenet_study = optuna.create_study(direction='maximize')
    densenet_study.optimize(
        lambda trial: objective_densenet(trial, X_train_processed, y_train, X_val_processed, y_val, input_shape, class_weights),
        n_trials=20  # Increased number of trials since we're only training one model type
    )
    
    print("Best DenseNet trial:")
    print(f"  Value (AUC): {densenet_study.best_trial.value:.4f}")
    print("  Params:")
    for key, value in densenet_study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Create and train the best DenseNet model
    best_densenet = create_densenet_model(
        input_shape,
        n_blocks=densenet_study.best_trial.params['n_blocks'],
        initial_filters=densenet_study.best_trial.params['initial_filters'],
        growth_rate=densenet_study.best_trial.params['growth_rate'],
        layers_per_block=densenet_study.best_trial.params['layers_per_block'],
        dropout_rate=densenet_study.best_trial.params['dropout_rate'],
        l2_reg=densenet_study.best_trial.params['l2_reg']
    )
    best_densenet.compile(
        optimizer=optimizers.Adam(learning_rate=densenet_study.best_trial.params['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    
    checkpoint = ModelCheckpoint(
        os.path.join(models_dir, 'best_densenet_model.h5'),
        save_best_only=True,
        monitor='val_auc',
        mode='max'
    )
    
    densenet_history = best_densenet.fit(
        X_train_processed, y_train,
        epochs=150,  # Increased epochs since we're only training one model
        batch_size=densenet_study.best_trial.params['batch_size'],
        validation_data=(X_val_processed, y_val),
        callbacks=[EarlyStopping(patience=20, restore_best_weights=True, monitor='val_auc', mode='max'), checkpoint],
        class_weight=class_weights,
        verbose=1
    )
    
    # Find best classification threshold
    print("\nFinding optimal classification threshold...")
    
    # For DenseNet
    densenet_val_probs = best_densenet.predict(X_val_processed)
    best_densenet_threshold = 0.5
    best_densenet_f1 = 0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        densenet_val_preds = (densenet_val_probs > threshold).astype(int).flatten()
        f1 = sklearn.metrics.f1_score(y_val, densenet_val_preds)
        if f1 > best_densenet_f1:
            best_densenet_f1 = f1
            best_densenet_threshold = threshold
    
    print(f"Best DenseNet threshold: {best_densenet_threshold:.2f} (F1: {best_densenet_f1:.4f})")
    
    # Evaluate DenseNet model on test set
    print("\nEvaluating DenseNet model on test set:")
    
    densenet_pred_proba = best_densenet.predict(X_test_processed)
    densenet_pred = (densenet_pred_proba > best_densenet_threshold).astype(int).flatten()
    densenet_accuracy = accuracy_score(y_test, densenet_pred)
    densenet_auc = sklearn.metrics.roc_auc_score(y_test, densenet_pred_proba)
    densenet_f1 = sklearn.metrics.f1_score(y_test, densenet_pred)
    
    print(f"\nDenseNet Test Metrics:")
    print(f"  Accuracy: {densenet_accuracy:.4f}")
    print(f"  AUC: {densenet_auc:.4f}")
    print(f"  F1 Score: {densenet_f1:.4f}")
    print("\nDenseNet Classification Report:")
    print(classification_report(y_test, densenet_pred))
    
    # Save the final model
    print("\nSaving DenseNet model...")
    best_densenet.save(os.path.join(models_dir, 'best_densenet_final_model.h5'))
    
    # Save the threshold and preprocessor for inference
    with open(os.path.join(models_dir, 'best_densenet_metadata.pkl'), 'wb') as f:
        import pickle
        pickle.dump({
            'threshold': best_densenet_threshold,
            'metrics': {
                "accuracy": densenet_accuracy,
                "auc": densenet_auc,
                "f1": densenet_f1
            },
            'class_weights': class_weights
        }, f)
    
    return best_densenet, preprocessor, best_densenet_threshold

if __name__ == "__main__":
    main() 