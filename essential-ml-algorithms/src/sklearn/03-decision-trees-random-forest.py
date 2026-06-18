from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

from execution_profiler import profile_execution

@profile_execution
def test_random_forest():
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")

    # See which features matter most
    feature_importance = pd.Series(model.feature_importances_, index=data.feature_names)
    print("\nFeature Importance:")
    print(feature_importance.sort_values(ascending=False))


if __name__ == "__main__":
    test_random_forest()