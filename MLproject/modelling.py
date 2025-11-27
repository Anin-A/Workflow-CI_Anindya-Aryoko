import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import sys
import warnings

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Ambil argumen
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data_preprocessing/loan_approval_dataset_preprocessing.csv"
    )

    # Load dataset
    df = pd.read_csv(file_path)
    X = df.drop(columns=['loan_approved'])
    y = df['loan_approved']

    # Oversampling dengan SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("Distribusi target setelah SMOTE:")
    print(y_resampled.value_counts(normalize=True))

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=0.2,
        random_state=42,
        stratify=y_resampled
    )

    print("\nDistribusi target di train set:")
    print(y_train.value_counts(normalize=True))

    print("\nDistribusi target di test set:")
    print(y_test.value_counts(normalize=True))

    input_example = X_train.iloc[0:5]

    # Autolog MLflow
    mlflow.sklearn.autolog()

    # Mulai run MLflow
    with mlflow.start_run(run_name="GradientBoosting"):
        # Train model
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        # Log metric akurasi
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy:", acc)
        mlflow.log_metric("accuracy", acc)
