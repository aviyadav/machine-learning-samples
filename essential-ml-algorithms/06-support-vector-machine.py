from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from execution_profiler import profile_execution

@profile_execution
def test_svm():
    # Handwriting digit recognition
    data = load_digits()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model.fit(X_train_scaled, y_train)

    print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test_scaled)):.2%}")


if __name__ == "__main__":
    test_svm()