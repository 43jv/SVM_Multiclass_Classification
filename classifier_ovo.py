import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from itertools import combinations
import argparse

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(df, fit_scaler=False, preprocessor=None):
    # Separate features and target variable
    X = df.drop('Segmentation', axis=1, errors='ignore')
    y = df['Segmentation'] if 'Segmentation' in df.columns else None
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Preprocessing pipeline for numerical and categorical columns
    if preprocessor is None:
        # Impute missing values, scale numerical, encode categorical
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        X = preprocessor.fit_transform(X)
    else:
        X = preprocessor.transform(X)
    
    # Check for NaN values
    if np.isnan(X).any():
        raise ValueError("There are still NaN values in the data after preprocessing!")
    
    return X, y, preprocessor

def save_predictions(predictions, output_file):
    output_df = pd.DataFrame(predictions, columns=['predicted'])
    output_df.to_csv(output_file, index=False)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def train_ovo_svm(X_train, y_train):
    classes = np.unique(y_train)
    classifiers = {}
    
    for (class1, class2) in combinations(classes, 2):
        mask = np.logical_or(y_train == class1, y_train == class2)
        X_pair = X_train[mask]
        y_pair = y_train[mask]
        
        # Convert class labels to binary encoding
        y_pair = np.where(y_pair == class1, 0, 1)
        clf = SVC(kernel='linear', probability=True)
        clf.fit(X_pair, y_pair)
        
        classifiers[(class1, class2)] = clf
        
    return classifiers

# Predict using One-vs-One SVM
def predict_ovo(classifiers, X_test):
    votes = np.zeros((X_test.shape[0], len(classifiers)), dtype=object)  # Use object type for string labels
    class_labels = list(classifiers.keys())
    
    for i, (class1, class2) in enumerate(classifiers):
        clf = classifiers[(class1, class2)]
        predictions = clf.predict(X_test)
        class_predictions = np.where(predictions == 0, class1, class2)  # Revert binary encoding
        votes[:, i] = class_predictions
    
    # Get final predictions based on majority vote
    final_predictions = []
    for i in range(votes.shape[0]):
        unique, counts = np.unique(votes[i], return_counts=True)
        majority_vote = unique[np.argmax(counts)]  # Get the class with the most votes
        final_predictions.append(majority_vote)
    
    return np.array(final_predictions)

def main(train_path, test_path):
    # Load and preprocess the data
    train_df, test_df = load_data(train_path, test_path)
    X, y, preprocessor = preprocess_data(train_df, fit_scaler=True)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers_ovo = train_ovo_svm(X_train, y_train)

    # Make predictions
    predictions_ovo = predict_ovo(classifiers_ovo, X_val)

    # Calculate metrics and print classification reports
    print("One-vs-One Classifier Report:")
    print(classification_report(y_val, predictions_ovo, zero_division=0))

    # Plot confusion matrices
    plot_confusion_matrix(y_val, predictions_ovo, np.unique(y))

    # Save predictions for test data
    X_test, _, _ = preprocess_data(test_df, fit_scaler=False, preprocessor=preprocessor)
    test_predictions_ovo = predict_ovo(classifiers_ovo, X_test)

    save_predictions(test_predictions_ovo, 'ovo.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and predict using One-vs-One SVM classifier.')
    parser.add_argument('test_path', type=str, help='Path to the test CSV file')
    args = parser.parse_args()

    train_path = 'Customer_train.csv'  # Replace with actual path to the training file
    main(train_path, args.test_path)