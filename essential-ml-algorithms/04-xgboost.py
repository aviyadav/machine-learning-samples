import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from xgboost import data

from execution_profiler import profile_execution

@profile_execution
def test_xgboost():
    data = load_diabetes()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"RMSE: {rmse:.2f}")


if __name__ == "__main__":
    test_xgboost()