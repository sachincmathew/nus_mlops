import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def main():
    # Load data
    data = load_breast_cancer()
    X = data.data
    y = data.target
    # Split data into training and test sets
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
    # Create and train the model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    # Predict and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()