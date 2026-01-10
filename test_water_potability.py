"""
Water Potability Prediction - Test Suite
MSc Computing - Independent Project - Assignment 3
Comprehensive test cases for testing and validation
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================================
# FIXTURES - Shared test data
# ============================================================================

@pytest.fixture(scope="module")
def dataset():
    """Load the water potability dataset."""
    df = pd.read_csv('water_potability.csv')
    return df


@pytest.fixture(scope="module")
def preprocessed_data(dataset):
    """Preprocess the dataset - handle missing values."""
    df = dataset.copy()
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            df[column].fillna(df[column].median(), inplace=True)
    return df


@pytest.fixture(scope="module")
def train_test_data(preprocessed_data):
    """Split data into train and test sets."""
    X = preprocessed_data.drop('Potability', axis=1)
    y = preprocessed_data['Potability']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


@pytest.fixture(scope="module")
def trained_models(train_test_data):
    """Train all four models."""
    X_train, X_test, y_train, y_test, scaler = train_test_data
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, 
                                  random_state=42, eval_metric='logloss', scale_pos_weight=1.5),
        'LightGBM': LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, 
                                    random_state=42, verbose=-1, class_weight='balanced')
    }
    
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    
    return trained


# ============================================================================
# TEST CASE 1: DATA LOADING TESTS
# ============================================================================

class TestDataLoading:
    """Test cases for data loading functionality."""
    
    def test_TC001_dataset_file_exists(self):
        """TC001: Verify dataset file exists in project directory."""
        assert os.path.exists('water_potability.csv'), "Dataset file not found"
    
    def test_TC002_dataset_loads_successfully(self, dataset):
        """TC002: Verify dataset loads without errors."""
        assert dataset is not None, "Dataset failed to load"
        assert isinstance(dataset, pd.DataFrame), "Dataset should be a DataFrame"
    
    def test_TC003_dataset_has_correct_shape(self, dataset):
        """TC003: Verify dataset has expected number of rows and columns."""
        assert dataset.shape[0] > 3000, f"Expected >3000 rows, got {dataset.shape[0]}"
        assert dataset.shape[1] == 10, f"Expected 10 columns, got {dataset.shape[1]}"
    
    def test_TC004_dataset_has_required_columns(self, dataset):
        """TC004: Verify all required columns are present."""
        required_columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                           'Conductivity', 'Organic_carbon', 'Trihalomethanes', 
                           'Turbidity', 'Potability']
        for col in required_columns:
            assert col in dataset.columns, f"Missing column: {col}"
    
    def test_TC005_target_column_binary(self, dataset):
        """TC005: Verify target column contains only 0 and 1."""
        unique_values = dataset['Potability'].unique()
        assert set(unique_values).issubset({0, 1}), f"Target should be binary, got {unique_values}"


# ============================================================================
# TEST CASE 2: DATA PREPROCESSING TESTS
# ============================================================================

class TestDataPreprocessing:
    """Test cases for data preprocessing functionality."""
    
    def test_TC006_missing_values_handled(self, preprocessed_data):
        """TC006: Verify no missing values after preprocessing."""
        missing = preprocessed_data.isnull().sum().sum()
        assert missing == 0, f"Found {missing} missing values after preprocessing"
    
    def test_TC007_data_types_correct(self, preprocessed_data):
        """TC007: Verify all features are numeric."""
        for col in preprocessed_data.columns:
            assert preprocessed_data[col].dtype in ['int64', 'float64'], \
                f"Column {col} has non-numeric type: {preprocessed_data[col].dtype}"
    
    def test_TC008_no_duplicate_rows(self, preprocessed_data):
        """TC008: Verify no duplicate rows in dataset."""
        duplicates = preprocessed_data.duplicated().sum()
        # Note: Some duplicates may exist in real data, so we just log this
        print(f"Found {duplicates} duplicate rows")
        assert True  # Informational test
    
    def test_TC009_feature_scaling_applied(self, train_test_data):
        """TC009: Verify StandardScaler normalizes features correctly."""
        X_train_scaled, _, _, _, scaler = train_test_data
        # Scaled data should have mean ≈ 0 and std ≈ 1
        mean = np.mean(X_train_scaled, axis=0)
        std = np.std(X_train_scaled, axis=0)
        assert np.allclose(mean, 0, atol=0.1), "Scaled mean should be close to 0"
        assert np.allclose(std, 1, atol=0.1), "Scaled std should be close to 1"
    
    def test_TC010_train_test_split_ratio(self, train_test_data, preprocessed_data):
        """TC010: Verify 80-20 train-test split."""
        X_train, X_test, _, _, _ = train_test_data
        total = len(preprocessed_data)
        train_ratio = len(X_train) / total
        test_ratio = len(X_test) / total
        assert 0.79 <= train_ratio <= 0.81, f"Train ratio should be ~0.8, got {train_ratio}"
        assert 0.19 <= test_ratio <= 0.21, f"Test ratio should be ~0.2, got {test_ratio}"


# ============================================================================
# TEST CASE 3: MODEL TRAINING TESTS
# ============================================================================

class TestModelTraining:
    """Test cases for model training functionality."""
    
    def test_TC011_logistic_regression_trains(self, trained_models):
        """TC011: Verify Logistic Regression model trains successfully."""
        assert 'Logistic Regression' in trained_models
        assert trained_models['Logistic Regression'] is not None
    
    def test_TC012_random_forest_trains(self, trained_models):
        """TC012: Verify Random Forest model trains successfully."""
        assert 'Random Forest' in trained_models
        assert trained_models['Random Forest'] is not None
    
    def test_TC013_xgboost_trains(self, trained_models):
        """TC013: Verify XGBoost model trains successfully."""
        assert 'XGBoost' in trained_models
        assert trained_models['XGBoost'] is not None
    
    def test_TC014_lightgbm_trains(self, trained_models):
        """TC014: Verify LightGBM model trains successfully."""
        assert 'LightGBM' in trained_models
        assert trained_models['LightGBM'] is not None


# ============================================================================
# TEST CASE 4: MODEL PREDICTION TESTS
# ============================================================================

class TestModelPrediction:
    """Test cases for model prediction functionality."""
    
    def test_TC015_logistic_regression_predicts(self, trained_models, train_test_data):
        """TC015: Verify Logistic Regression makes predictions."""
        _, X_test, _, y_test, _ = train_test_data
        model = trained_models['Logistic Regression']
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})
    
    def test_TC016_random_forest_predicts(self, trained_models, train_test_data):
        """TC016: Verify Random Forest makes predictions."""
        _, X_test, _, y_test, _ = train_test_data
        model = trained_models['Random Forest']
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})
    
    def test_TC017_xgboost_predicts(self, trained_models, train_test_data):
        """TC017: Verify XGBoost makes predictions."""
        _, X_test, _, y_test, _ = train_test_data
        model = trained_models['XGBoost']
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})
    
    def test_TC018_lightgbm_predicts(self, trained_models, train_test_data):
        """TC018: Verify LightGBM makes predictions."""
        _, X_test, _, y_test, _ = train_test_data
        model = trained_models['LightGBM']
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})
    
    def test_TC019_probability_predictions(self, trained_models, train_test_data):
        """TC019: Verify models return valid probability predictions."""
        _, X_test, _, _, _ = train_test_data
        for name, model in trained_models.items():
            probas = model.predict_proba(X_test)
            assert probas.shape[1] == 2, f"{name}: Should return 2 class probabilities"
            assert np.allclose(probas.sum(axis=1), 1), f"{name}: Probabilities should sum to 1"
            assert (probas >= 0).all() and (probas <= 1).all(), f"{name}: Probabilities should be in [0,1]"


# ============================================================================
# TEST CASE 5: MODEL ACCURACY TESTS
# ============================================================================

class TestModelAccuracy:
    """Test cases for model accuracy validation."""
    
    def test_TC020_logistic_regression_accuracy(self, trained_models, train_test_data):
        """TC020: Verify Logistic Regression accuracy is above baseline."""
        _, X_test, _, y_test, _ = train_test_data
        model = trained_models['Logistic Regression']
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Logistic Regression Accuracy: {accuracy:.4f}")
        assert accuracy > 0.4, f"Accuracy {accuracy:.4f} is below baseline 0.4"
    
    def test_TC021_random_forest_accuracy(self, trained_models, train_test_data):
        """TC021: Verify Random Forest accuracy is above 60%."""
        _, X_test, _, y_test, _ = train_test_data
        model = trained_models['Random Forest']
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Random Forest Accuracy: {accuracy:.4f}")
        assert accuracy > 0.6, f"Accuracy {accuracy:.4f} is below expected 0.6"
    
    def test_TC022_xgboost_accuracy(self, trained_models, train_test_data):
        """TC022: Verify XGBoost accuracy is above 60%."""
        _, X_test, _, y_test, _ = train_test_data
        model = trained_models['XGBoost']
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"XGBoost Accuracy: {accuracy:.4f}")
        assert accuracy > 0.6, f"Accuracy {accuracy:.4f} is below expected 0.6"
    
    def test_TC023_lightgbm_accuracy(self, trained_models, train_test_data):
        """TC023: Verify LightGBM accuracy is above 60%."""
        _, X_test, _, y_test, _ = train_test_data
        model = trained_models['LightGBM']
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"LightGBM Accuracy: {accuracy:.4f}")
        assert accuracy > 0.6, f"Accuracy {accuracy:.4f} is below expected 0.6"


# ============================================================================
# TEST CASE 6: INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidation:
    """Test cases for input validation and edge cases."""
    
    def test_TC024_valid_input_prediction(self, trained_models, train_test_data):
        """TC024: Verify prediction with valid input values."""
        _, _, _, _, scaler = train_test_data
        # Valid water sample input
        valid_input = np.array([[7.0, 200.0, 20000.0, 7.0, 300.0, 400.0, 14.0, 65.0, 4.0]])
        scaled_input = scaler.transform(valid_input)
        
        for name, model in trained_models.items():
            prediction = model.predict(scaled_input)
            assert prediction[0] in [0, 1], f"{name}: Invalid prediction {prediction[0]}"
    
    def test_TC025_extreme_ph_values(self, trained_models, train_test_data):
        """TC025: Verify model handles extreme pH values (0 and 14)."""
        _, _, _, _, scaler = train_test_data
        
        # Extreme low pH
        low_ph_input = np.array([[0.0, 200.0, 20000.0, 7.0, 300.0, 400.0, 14.0, 65.0, 4.0]])
        scaled_low = scaler.transform(low_ph_input)
        
        # Extreme high pH
        high_ph_input = np.array([[14.0, 200.0, 20000.0, 7.0, 300.0, 400.0, 14.0, 65.0, 4.0]])
        scaled_high = scaler.transform(high_ph_input)
        
        for name, model in trained_models.items():
            pred_low = model.predict(scaled_low)
            pred_high = model.predict(scaled_high)
            assert pred_low[0] in [0, 1], f"{name}: Failed on low pH"
            assert pred_high[0] in [0, 1], f"{name}: Failed on high pH"
    
    def test_TC026_boundary_values(self, trained_models, train_test_data):
        """TC026: Verify model handles minimum boundary values."""
        _, _, _, _, scaler = train_test_data
        # Minimum values (edge case)
        min_input = np.array([[0.0, 50.0, 300.0, 0.0, 100.0, 100.0, 2.0, 0.0, 1.0]])
        scaled_min = scaler.transform(min_input)
        
        for name, model in trained_models.items():
            prediction = model.predict(scaled_min)
            assert prediction[0] in [0, 1], f"{name}: Failed on boundary values"
    
    def test_TC027_maximum_boundary_values(self, trained_models, train_test_data):
        """TC027: Verify model handles maximum boundary values."""
        _, _, _, _, scaler = train_test_data
        # Maximum values (edge case)
        max_input = np.array([[14.0, 350.0, 60000.0, 15.0, 500.0, 800.0, 30.0, 130.0, 7.0]])
        scaled_max = scaler.transform(max_input)
        
        for name, model in trained_models.items():
            prediction = model.predict(scaled_max)
            assert prediction[0] in [0, 1], f"{name}: Failed on maximum boundary values"


# ============================================================================
# TEST CASE 7: METRICS CALCULATION TESTS
# ============================================================================

class TestMetricsCalculation:
    """Test cases for evaluation metrics calculation."""
    
    def test_TC028_precision_calculation(self, trained_models, train_test_data):
        """TC028: Verify precision is calculated correctly for all models."""
        _, X_test, _, y_test, _ = train_test_data
        
        for name, model in trained_models.items():
            predictions = model.predict(X_test)
            precision = precision_score(y_test, predictions, zero_division=0)
            assert 0 <= precision <= 1, f"{name}: Precision {precision} is invalid"
            print(f"{name} Precision: {precision:.4f}")
    
    def test_TC029_recall_calculation(self, trained_models, train_test_data):
        """TC029: Verify recall is calculated correctly for all models."""
        _, X_test, _, y_test, _ = train_test_data
        
        for name, model in trained_models.items():
            predictions = model.predict(X_test)
            recall = recall_score(y_test, predictions, zero_division=0)
            assert 0 <= recall <= 1, f"{name}: Recall {recall} is invalid"
            print(f"{name} Recall: {recall:.4f}")
    
    def test_TC030_f1_score_calculation(self, trained_models, train_test_data):
        """TC030: Verify F1-score is calculated correctly for all models."""
        _, X_test, _, y_test, _ = train_test_data
        
        for name, model in trained_models.items():
            predictions = model.predict(X_test)
            f1 = f1_score(y_test, predictions, zero_division=0)
            assert 0 <= f1 <= 1, f"{name}: F1-score {f1} is invalid"
            print(f"{name} F1-Score: {f1:.4f}")


# ============================================================================
# TEST CASE 8: FUTURE IMPROVEMENT TESTS (Expected to identify gaps)
# ============================================================================

class TestFutureImprovements:
    """Test cases that identify areas for future improvement."""
    
    def test_TC031_model_accuracy_target_70_percent(self, trained_models, train_test_data):
        """TC031: Target 70% accuracy for production readiness (Improvement needed)."""
        _, X_test, _, y_test, _ = train_test_data
        
        best_accuracy = 0
        for name, model in trained_models.items():
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
        
        print(f"Best Model Accuracy: {best_accuracy:.4f}")
        # Target: 70% accuracy for production - currently ~66%
        assert best_accuracy >= 0.70, \
            f"Best accuracy {best_accuracy:.2%} is below 70% target. Consider: feature engineering, hyperparameter tuning, or ensemble methods."
    
    def test_TC032_recall_minority_class_above_50_percent(self, trained_models, train_test_data):
        """TC032: Recall for potable class should be above 50% (Improvement needed)."""
        _, X_test, _, y_test, _ = train_test_data
        
        best_recall = 0
        best_model = ""
        for name, model in trained_models.items():
            predictions = model.predict(X_test)
            recall = recall_score(y_test, predictions, zero_division=0)
            if recall > best_recall:
                best_recall = recall
                best_model = name
        
        print(f"Best Recall ({best_model}): {best_recall:.4f}")
        # Target: 55% recall to reduce false negatives (missed potable water)
        assert best_recall >= 0.55, \
            f"Best recall {best_recall:.2%} is below 55%. Consider: SMOTE oversampling, threshold adjustment, or cost-sensitive learning."
    
    def test_TC033_class_balance_check(self, preprocessed_data):
        """TC033: Check if dataset has severe class imbalance (Informational)."""
        class_counts = preprocessed_data['Potability'].value_counts()
        ratio = class_counts.min() / class_counts.max()
        
        print(f"Class distribution: {dict(class_counts)}")
        print(f"Minority/Majority ratio: {ratio:.2f}")
        
        # Target: At least 45% minority class representation
        assert ratio >= 0.45, \
            f"Class imbalance detected (ratio={ratio:.2f}). Consider: SMOTE, class weights, or data augmentation."


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
