from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from execution_profiler import profile_execution

@profile_execution
def logistic_regression():
    # Real dataset: Cancer detection (benign vs malignant)
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    logistic_regression()