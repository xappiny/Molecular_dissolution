import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import joblib
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import warnings

warnings.filterwarnings('ignore')

# =================== Configuration ====================
BASE_PATH = Path(r"C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\external_val\final")  # Replace with your data directory
DATA_FILE = BASE_PATH / "scaler_features_model_train.csv"
MODEL_DIR = BASE_PATH / "models"
PLOT_DIR = BASE_PATH / "plots"
EPOCHS = 200
BATCH_SIZE = 16
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
MAX_EVALS = 100

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


# =================== Utility Functions ====================
def eval_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def adjust_hyperparameters(model_class, params, n_features):
    """Adjust hyperparameters for specific model types."""
    params = params.copy()
    if model_class in (RandomForestRegressor, DecisionTreeRegressor):
        if 'max_depth' in params:
            params['max_depth'] = params['max_depth'] + 1
        if 'max_features' in params:
            params['max_features'] = min(max(params['max_features'] + 1, 1), n_features)
    elif model_class == lgb.LGBMRegressor:
        if 'max_depth' in params:
            params['max_depth'] = params['max_depth'] + 3
        if 'num_leaves' in params:
            params['num_leaves'] = params['num_leaves'] + 5
        if 'n_estimators' in params:
            params['n_estimators'] = params['n_estimators'] + 5
        params['verbose'] = -1  # Suppress LightGBM output
    elif model_class == XGBRegressor:
        if 'max_depth' in params:
            params['max_depth'] = params['max_depth'] + 3
        if 'n_estimators' in params:
            params['n_estimators'] = params['n_estimators'] + 5
    return params


def train_and_save_model(model_class, params, X_train, y_train, model_path, n_features):
    """Train and save a model with adjusted hyperparameters."""
    params = adjust_hyperparameters(model_class, params, n_features)
    model = model_class(**params)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model


def optimize_model(model_class, hyper_space, groups, X_train, y_train, max_evals=MAX_EVALS):
    """Optimize model hyperparameters using Hyperopt."""

    def objective(hyperparams):
        params = adjust_hyperparameters(model_class, hyperparams, X_train.shape[1])
        model = model_class(**params)
        gkf = GroupKFold(n_splits=N_INNER_FOLDS)
        val_rmse, val_mae, val_r2 = [], [], []

        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
            train_x, train_y = X_train.iloc[train_idx], y_train.iloc[train_idx]
            val_x, val_y = X_train.iloc[val_idx], y_train.iloc[val_idx]
            model.fit(train_x, train_y)
            preds = model.predict(val_x)
            rmse, mae, r2 = eval_metrics(val_y, preds)
            val_rmse.append(rmse)
            val_mae.append(mae)
            val_r2.append(r2)

        return {
            'loss': np.mean(val_rmse),
            'status': STATUS_OK,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2
        }

    trials = Trials()
    best = fmin(fn=objective, space=hyper_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    inner_performance = {
        'rmse': trials.best_trial['result']['val_rmse'],
        'mae': trials.best_trial['result']['val_mae'],
        'r2': trials.best_trial['result']['val_r2']
    }
    return best, trials, inner_performance


def plot_predictions(model_name, y_true, y_pred, performance, plot_path):
    """Create a hexbin scatter plot with histograms for model predictions."""
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(4, 4)

    # Clip data to [0, 1]
    y_true = np.clip(y_true, 0, 1)
    y_pred = np.clip(y_pred, 0, 1)

    # Main hexbin plot
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    slope, intercept, r_value, _, _ = stats.linregress(y_true, y_pred)
    line_x = np.linspace(0, 1, 100)
    line_y = line_x
    hb = ax_main.hexbin(y_true, y_pred, gridsize=50, cmap='Blues', mincnt=1, norm=LogNorm())
    ax_main.plot(line_x, line_y, color='k', linewidth=1, alpha=0.8, linestyle='--', label='y=x')

    # Add metrics text
    ax_main.text(0.05, 0.95, f'N = {len(y_true):,}', transform=ax_main.transAxes, fontsize=10)
    ax_main.text(0.05, 0.88, f'y = {slope:.2f}x + {intercept:.2f}', transform=ax_main.transAxes, fontsize=10)
    ax_main.text(0.05, 0.81, f'R² = {np.mean(performance["r2"]):.3f}', transform=ax_main.transAxes, fontsize=10)
    ax_main.text(0.05, 0.74, f'Pearson = {r_value:.3f}', transform=ax_main.transAxes, fontsize=10)
    ax_main.text(0.05, 0.67, f'RMSE = {np.mean(performance["rmse"]):.3f}', transform=ax_main.transAxes, fontsize=10)
    ax_main.set_xlabel('Experimental Value', fontsize=12)
    ax_main.set_ylabel('Predicted Value', fontsize=12)
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)

    # X-axis histogram
    ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    n, bins, patches = ax_histx.hist(y_true, bins=50, range=(0, 1))
    cm = plt.get_cmap('Blues')
    norm = Normalize(vmin=min(n), vmax=max(n))
    for c, p in zip(n, patches):
        plt.setp(p, 'facecolor', cm(norm(c)))
    ax_histx.set_yticks([])
    for spine in ax_histx.spines.values():
        spine.set_visible(False)

    # Y-axis histogram
    ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    n, bins, patches = ax_histy.hist(y_pred, bins=50, range=(0, 1), orientation='horizontal')
    cm = plt.get_cmap('Blues')
    norm = Normalize(vmin=min(n), vmax=max(n))
    for c, p in zip(n, patches):
        plt.setp(p, 'facecolor', cm(norm(c)))
    ax_histy.set_xticks([])
    for spine in ax_histy.spines.values():
        spine.set_visible(False)

    # Add colorbar and save
    fig.colorbar(hb, ax=ax_main, label='Count', fraction=0.046, pad=0.04)
    plt.suptitle(f'{model_name.upper()} Test Set Predictions', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_path, dpi=600)
    plt.close()


# =================== Main Script ====================
def main():
    """Run nested cross-validation for multiple models and generate predictions."""
    # Load and validate data
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, encoding='unicode_escape')
    missing_cols = df.columns[df.isna().any()].tolist()
    if missing_cols:
        print(f"Warning: Columns with missing values: {missing_cols}")
        df = df.fillna(df.mean(numeric_only=True))  # Simple imputation

    features = df.iloc[:, 1:-1]
    target = df['Output_diss_fraction']
    groups = df['formulation_id']

    # Define models and hyperparameter spaces
    models = {
        'lgb': (lgb.LGBMRegressor, {
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.2),
            'max_depth': hp.choice('max_depth', range(3, 15)),
            'num_leaves': hp.choice('num_leaves', range(5, 256)),
            'subsample': hp.uniform('subsample', 0.6, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
            'n_estimators': hp.choice('n_estimators', range(5, 2000)),
            'reg_alpha': hp.uniform('reg_alpha', 0, 100),
            'reg_lambda': hp.uniform('reg_lambda', 0, 100)
        }),
        'rf': (RandomForestRegressor, {
            'max_depth': hp.choice('max_depth', range(1, 100)),
            'max_features': hp.choice('max_features', range(1, features.shape[1])),
            'n_estimators': hp.choice('n_estimators', range(100, 1000))
        }),
        'xgb': (XGBRegressor, {
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.2),
            'max_depth': hp.choice('max_depth', range(3, 20)),
            'subsample': hp.uniform('subsample', 0.6, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
            'n_estimators': hp.choice('n_estimators', range(5, 1000)),
            'reg_alpha': hp.uniform('reg_alpha', 0, 100),
            'reg_lambda': hp.uniform('reg_lambda', 0, 100),
            'min_child_weight': hp.randint('min_child_weight', 6),
            'gamma': hp.uniform('gamma', 0, 1)
        }),
        'mlp': (MLPRegressor, {
            'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)]),
            'alpha': hp.loguniform('alpha', -6, 0),
            'learning_rate_init': hp.loguniform('learning_rate_init', -5, -1),
            'max_iter': hp.choice('max_iter', range(200, 1000))
        }),
        'dt': (DecisionTreeRegressor, {
            'max_depth': hp.choice('max_depth', range(1, 100)),
            'max_features': hp.choice('max_features', range(1, features.shape[1]))
        })
    }

    # Initialize storage for performance and predictions
    best_models_info = {
        name: {'model': None, 'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf'),
               'fold_idx': -1, 'params': None}
        for name in models
    }
    outer_performance = {name: {'rmse': [], 'mae': [], 'r2': []} for name in models}
    inner_performance_all = {name: {'rmse': [], 'mae': [], 'r2': []} for name in models}
    all_test_preds = {name: [] for name in models}
    all_test_y = {name: [] for name in models}

    # Nested cross-validation
    outer_cv = GroupKFold(n_splits=N_OUTER_FOLDS)
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(features, target, groups=groups)):
        print(f"\nOuter Fold {fold_idx + 1}")
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        for model_name, (model_class, hyper_space) in models.items():
            print(f"Optimizing {model_name.upper()} in Fold {fold_idx + 1}...")
            best_params, _, inner_perf = optimize_model(model_class, hyper_space, groups_train, X_train, y_train)

            # Train and evaluate model
            model = train_and_save_model(
                model_class, best_params, X_train, y_train,
                MODEL_DIR / f"best_{model_name}_fold{fold_idx + 1}.pkl",
                X_train.shape[1]
            )

            # Record inner loop performance
            for metric in ['rmse', 'mae', 'r2']:
                inner_performance_all[model_name][metric].extend(inner_perf[metric])

            # Test set predictions
            y_pred = model.predict(X_test)
            all_test_preds[model_name].extend(y_pred)
            all_test_y[model_name].extend(y_test)

            # Outer loop performance
            rmse, mae, r2 = eval_metrics(y_test, y_pred)
            outer_performance[model_name]['rmse'].append(rmse)
            outer_performance[model_name]['mae'].append(mae)
            outer_performance[model_name]['r2'].append(r2)

            # Update best model
            if rmse < best_models_info[model_name]['rmse']:
                best_models_info[model_name].update({
                    'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2,
                    'fold_idx': fold_idx + 1, 'params': best_params
                })

    # Save best models and print results
    print("\n=== Global Best Models ===")
    for model_name, info in best_models_info.items():
        if info['model'] is not None:
            model_path = MODEL_DIR / f"best_{model_name}_fold{info['fold_idx']}_global.pkl"
            joblib.dump(info['model'], model_path)
            print(f"\nBest {model_name.upper()} model saved to {model_path}")
            print(f"Best performance (Fold {info['fold_idx']}): RMSE = {info['rmse']:.3f}, "
                  f"MAE = {info['mae']:.3f}, R² = {info['r2']:.3f}")
            print(f"Best hyperparameters: {info['params']}")

    # Print performance statistics
    print("\n=== Performance Statistics ===")
    for model_name in models:
        print(f"\nResults for {model_name.upper()}:")
        print("Inner Loop Performance (across all inner folds):")
        for metric in ['rmse', 'mae', 'r2']:
            mean_val = np.mean(inner_performance_all[model_name][metric])
            std_val = np.std(inner_performance_all[model_name][metric])
            print(f"  {metric.upper()}: {mean_val:.3f}±{std_val:.3f}")
        print("Outer Loop Performance (across 5 outer folds):")
        for metric in ['rmse', 'mae', 'r2']:
            mean_val = np.mean(outer_performance[model_name][metric])
            std_val = np.std(outer_performance[model_name][metric])
            print(f"  {metric.upper()}: {mean_val:.3f}±{std_val:.3f}")

    # Generate plots
    for model_name in models:
        plot_predictions(
            model_name, all_test_y[model_name], all_test_preds[model_name],
            outer_performance[model_name], PLOT_DIR / f"{model_name}_test_set_scatter_hist.png"
        )

    print("\nAll model evaluations and plots completed!")


if __name__ == "__main__":
    main()