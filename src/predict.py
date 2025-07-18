from sklearn.preprocessing import StandardScaler
import pandas as pd
from tabpfn import TabPFNRegressor
import joblib
import numpy as np
from pathlib import Path

# Define constants for file paths
DATA_PATH = r'\descriptor_molecular_v2.csv'
SCALER_PATH = r'\scaler.pkl'
TABPFN_MODEL_PATH = r'\tabpfn.pkl'
RF_MODEL_PATH = r'\rf.pkl'
LGBM_MODEL_PATH = r'\lgb.pkl'
OUTPUT_PATH = r'\external_prediction_molecular.csv'


# Function to load data and models with error handling
def load_data_and_models():
    try:
        # Load pre-trained scaler
        scaler = joblib.load(SCALER_PATH)
        # Load pre-trained models
        tabpfn_model = joblib.load(TABPFN_MODEL_PATH)
        rf_model = joblib.load(RF_MODEL_PATH)
        lgbm_model = joblib.load(LGBM_MODEL_PATH)
        # Load new dataset
        data_new = pd.read_csv(DATA_PATH, encoding='unicode_escape')
        return scaler, tabpfn_model, rf_model, lgbm_model, data_new
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        raise
    except Exception as e:
        print(f"Error loading data or models: {e}")
        raise


# Function to preprocess data
def preprocess_data(data_new, scaler):
    try:
        # Verify required columns
        if 'formulation_id' not in data_new.columns:
            raise ValueError("Missing 'formulation_id' column in dataset")

        # Assume features start from the third column (adjust if needed)
        X_new = data_new.iloc[:, 1:-1]
        if X_new.empty:
            raise ValueError("No features found in dataset (check column selection)")

        # Standardize features using pre-trained scaler
        X_new_scaled = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns)
        return X_new_scaled
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise


# Function to make predictions
def make_predictions(X_new_scaled, tabpfn_model, rf_model, lgbm_model):
    try:
        predictions = {
            'TabPFN': tabpfn_model.predict(X_new_scaled),
            'RF': rf_model.predict(X_new_scaled),
            'LGBM': lgbm_model.predict(X_new_scaled)
        }
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise


# Main execution
def main():
    # Load data and models
    scaler, tabpfn_model, rf_model, lgbm_model, data_new = load_data_and_models()

    # Preprocess data
    X_new_scaled = preprocess_data(data_new, scaler)

    # Make predictions
    predictions = make_predictions(X_new_scaled, tabpfn_model, rf_model, lgbm_model)

    # Add predictions to original data
    data_new['Predicted_diss_TabPFN'] = predictions['TabPFN']
    data_new['Predicted_diss_RF'] = predictions['RF']
    data_new['Predicted_diss_LGBM'] = predictions['LGBM']

    # Organize predictions by formulation
    has_output_time = 'Output_time' in data_new.columns
    three_drug_predictions = {}

    for formulation_id in data_new['formulation_id'].unique():
        formulation_data = data_new[data_new['formulation_id'] == formulation_id]

        if has_output_time:
            time_points = formulation_data['Output_time'].values
            three_drug_predictions[formulation_id] = {
                'TabPFN': dict(zip(time_points, formulation_data['Predicted_diss_TabPFN'].values)),
                'RF': dict(zip(time_points, formulation_data['Predicted_diss_RF'].values)),
                'LGBM': dict(zip(time_points, formulation_data['Predicted_diss_LGBM'].values))
            }
        else:
            three_drug_predictions[formulation_id] = {
                'TabPFN': formulation_data['Predicted_diss_TabPFN'].values,
                'RF': formulation_data['Predicted_diss_RF'].values,
                'LGBM': formulation_data['Predicted_diss_LGBM'].values
            }

    # Print predictions for ISFG formulations
    print("=== Predicted Dissolution Profiles for ISFG Formulations ===")
    for formulation_id, predictions_dict in three_drug_predictions.items():
        print(f"Formulation ID: {formulation_id}")
        for model_name, model_predictions in predictions_dict.items():
            print(f"  Model: {model_name}")
            if has_output_time:
                for time_point, diss in model_predictions.items():
                    print(f"    Time: {time_point:.1f} hr, Predicted Dissolution: {diss:.3f}%")
            else:
                for idx, diss in enumerate(model_predictions):
                    print(f"    Sample {idx}: Predicted Dissolution: {diss:.3f}%")
        print("-" * 50)

    # Save predictions to CSV
    try:
        predictions_df = pd.DataFrame({
            'formulation_id': data_new['formulation_id'],
            **({'time_point': data_new['Output_time']} if has_output_time else {}),
            'Predicted_diss_TabPFN': predictions['TabPFN'],
            'Predicted_diss_RF': predictions['RF'],
            'Predicted_diss_LGBM': predictions['LGBM']
        })
        predictions_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Predictions saved to '{OUTPUT_PATH}'")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        raise


if __name__ == "__main__":
    main()
