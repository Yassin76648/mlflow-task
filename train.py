import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
EXPERIMENT_NAME = "california-housing-gbr"

CONFIGS = [
    {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
]


def load_and_split():
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


def main():
    X_train, X_val, y_train, y_val = load_and_split()
    print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    results = []

    for cfg in CONFIGS:
        with mlflow.start_run(run_name=cfg["run_name"]):
            model = GradientBoostingRegressor(
                max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"],
                random_state=RANDOM_SEED,
            )
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
            mae = float(mean_absolute_error(y_val, preds))
            r2 = float(r2_score(y_val, preds))

            mlflow.log_param("max_depth", cfg["max_depth"])
            mlflow.log_param("learning_rate", cfg["learning_rate"])

            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)

            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )

            results.append({**cfg, "RMSE": rmse, "MAE": mae, "R2": r2})
            print(f"{cfg['run_name']}: depth={cfg['max_depth']} lr={cfg['learning_rate']} "
                  f"RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")

    print("\n| Run | Max Depth | Learning Rate | RMSE | MAE | R2 |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['run_name']} | {r['max_depth']} | {r['learning_rate']} | "
              f"{r['RMSE']:.4f} | {r['MAE']:.4f} | {r['R2']:.4f} |")

    best = min(results, key=lambda r: r["RMSE"])
    print(f"\nBest model: {best['run_name']} "
          f"(max_depth={best['max_depth']}, learning_rate={best['learning_rate']}, "
          f"RMSE={best['RMSE']:.4f})")


if __name__ == "__main__":
    main()
