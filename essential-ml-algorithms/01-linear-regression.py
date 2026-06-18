from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from execution_profiler import profile_execution


@profile_execution
def test_linear_regression():
    # Sample: Predicting house prices
    X = np.array([[750], [1000], [1200], [1500], [1800], [2200]])  # Square footage
    y = np.array([150000, 200000, 230000, 290000, 340000, 410000])  # House prices

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Predictions: {predictions}")
    print(f"MSE: {mean_squared_error(y_test, predictions):.2f}")    


if __name__ == "__main__":
    test_linear_regression()