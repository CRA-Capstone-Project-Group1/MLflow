# from setuptools import find_packages, setup


# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='Credit Risk Model code structuring',
#     author='Swapnil Kangralkar',
#     license='',
# )

# for tracking experiments, models, and associated metadata.
import mlflow
from src.data.make_dataset import load_and_preprocess_data
#from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance, plot_confusion_matrix
from src.features.build_features import create_dummy_vars

# This comes from src/models/__init__.py file
"""By including these imports in __init__.py, you can import train_decision_tree and evaluate_model 
directly from the src.models package without needing to reference the specific modules."""
from src.models import train_decision_tree
from src.models import evaluate_model

if __name__ == "__main__":
    
    # set up ML-flow experiment
    mlflow.set_experiment("Credit Default Prediction")

    # Start an MLflow run
    with mlflow.start_run(run_name="Initial Model Run"):
    
        # Load and preprocess the data
        data_path = "/Users/swapnilklkar/Documents/Experiment_tracking_with_MLflow/data/raw/credit.csv"
        df = load_and_preprocess_data(data_path)
        
        # Set a tag for this run
        mlflow.set_tag("model_type", "DecisionTree")
        mlflow.set_tag("dataset", "CreditRisk")

        # Create dummy variables and separate features and target
        X, y = create_dummy_vars(df)

        # Train the decision tree classifier model
        model, X_test_scaled, y_test, max_depth, min_samples_split, min_samples_leaf = train_decision_tree(X, y)

        # Evaluate the model        
        results = evaluate_model(model, X_test_scaled, y_test)
        
        # Log scalar metrics
        for metric in ['Accuracy', 'Precision', 'Recall']:
            mlflow.log_metric(metric, results[metric])
            
        # Log the model parameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        print(results)