import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabpfn import TabPFNRegressor
import joblib
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import logging
from datetime import datetime
import torch
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
np.random.seed(42)

# =================== Configuration ====================
BASE_PATH = Path(r"\final")  # Replace with your data directory
DATA_FILE = BASE_PATH / "scaler_features_model_train.csv"
MODEL_DIR = BASE_PATH / "models"
PLOT_DIR = BASE_PATH / "plots"
N_FOLDS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


# =================== Logging Setup ====================
def setup_logging(base_path):
    """Set up logging with a timestamped log file."""
    log_filename = base_path / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Starting TabPFN training script")
    logging.info(f"Using device: {DEVICE}")


# =================== Utility Functions ====================
def eval_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def load_data(file_path):
    """Load and validate dataset."""
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at {file_path}")
    data = pd.read_csv(file_path, encoding='unicode_escape')
    missing_cols = data.columns[data.isna().any()].tolist()
    if missing_cols:
        logging.warning(f"Columns with missing values: {missing_cols}")
        data = data.fillna(data.mean(numeric_only=True))  # Mean imputation
    return data


def plot_shap_summary(shap_values, features, plot_path, title="SHAP Summary Plot"):
    """Create and save SHAP summary plot."""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, features, plot_type="dot", show=False)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=600)
    plt.close()
    logging.info(f"SHAP summary plot saved to {plot_path}")


def plot_feature_importance(shap_values, features, plot_path, title="Feature Importance Visualization"):
    """Create and save SHAP feature importance bar plot."""
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'SHAP Value': mean_shap,
        'Percentage': (mean_shap / mean_shap.sum() * 100).round(2)
    }, index=features.columns).sort_values(by='SHAP Value', ascending=False)

    # Save to CSV
    importance_df.to_csv(plot_path.with_name("feature_importance.csv"))
    logging.info(f"Feature importance saved to {plot_path.with_name('feature_importance.csv')}")

    # Plot
    cmap = plt.get_cmap('coolwarm')
    plt.figure(figsize=(12, 8))
    bars = plt.barh(importance_df.index, importance_df['SHAP Value'],
                    color=cmap(importance_df['SHAP Value'] / importance_df['SHAP Value'].max()))
    for i, (_, row) in enumerate(importance_df.iterrows()):
        plt.text(row['SHAP Value'], i, f"{row['Percentage']:.2f}%", va='center', ha='left', fontsize=10)
    plt.xlabel('mean(|SHAP value|) (average impact on model output)')
    plt.ylabel('Features')
    plt.title(title)
    plt.gca().invert_yaxis()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, importance_df['SHAP Value'].max()))
    plt.colorbar(sm, label='SHAP Value')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=600)
    plt.close()
    logging.info(f"Feature importance plot saved to {plot_path}")


# =================== Main Script ====================
def main():
    """Run nested cross-validation with TabPFN and SHAP analysis."""
    setup_logging(BASE_PATH)

    # Load data
    data = load_data(DATA_FILE)
    features = data.iloc[:, 1:-1]
    target = data['Output_diss_fraction']
    groups = data['formulation_id']
    logging.info(f"Loaded dataset with {len(data)} samples, {features.shape[1]} features")

    # Initialize storage
    performance_list = []
    shap_values_all = []
    predictions_all = []
    X_test_all = []
    y_test_all = []

    # Cross-validation
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(features, target, groups)):
        logging.info(f"Processing Fold {fold_idx + 1}")
        X_train = features.iloc[train_idx]
        X_test = features.iloc[test_idx]
        y_train = target.iloc[train_idx]
        y_test = target.iloc[test_idx]

        # Train TabPFN model
        model = TabPFNRegressor(device=DEVICE)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        metrics = eval_metrics(y_test, y_pred)
        performance_list.append(metrics)
        logging.info(f"Fold {fold_idx + 1} Performance: " +
                     ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

        # SHAP analysis
        explainer = shap.Explainer(model.predict, X_test)
        shap_values = explainer(X_test)
        shap_values_all.append(shap_values.values)
        predictions_all.append(y_pred)
        X_test_all.append(X_test)
        y_test_all.append(y_test)

        # Clear GPU memory
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    # Print performance
    performance_df = pd.DataFrame(performance_list)
    print("\n=== 5-Fold Cross-Validation Performance ===")
    for metric in performance_df.columns:
        mean_val = performance_df[metric].mean()
        std_val = performance_df[metric].std()
        print(f"{metric}: {mean_val:.3f}±{std_val:.3f}")
        logging.info(f"{metric}: {mean_val:.3f}±{std_val:.3f}")

    # Save performance metrics
    performance_df.to_csv(MODEL_DIR / "performance_metrics.csv", index=False)
    logging.info("Performance metrics saved to performance_metrics.csv")

    # Train and save final model
    final_model = TabPFNRegressor(device=DEVICE)
    final_model.fit(features, target)
    model_path = MODEL_DIR / "tabpfn.pkl"
    joblib.dump(final_model, model_path)
    logging.info(f"Final model saved to {model_path}")

    # Combine SHAP results
    shap_values_all = np.concatenate(shap_values_all, axis=0)
    predictions_all = np.concatenate(predictions_all, axis=0)
    X_test_all = pd.concat(X_test_all, axis=0)
    y_test_all = np.concatenate(y_test_all, axis=0)

    # Save SHAP values
    shap_dict = {
        'predicted_value': predictions_all,
        'true_value': y_test_all,
        **{f'shap_{col}': shap_values_all[:, i] for i, col in enumerate(X_test_all.columns)},
        **{col: X_test_all[col].values for col in X_test_all.columns}
    }
    shap_values_df = pd.DataFrame(shap_dict)
    shap_values_df.to_csv(MODEL_DIR / "shap_values.csv", index=False)
    logging.info("SHAP values saved to shap_values.csv")

    # Generate plots
    plot_shap_summary(shap_values_all, X_test_all, PLOT_DIR / "shap_summary_plot.png")
    plot_feature_importance(shap_values_all, X_test_all, PLOT_DIR / "shap_feature_importance.png")

    logging.info("Script completed successfully!")


if __name__ == "__main__":
    main()
