from .train_model import train_decision_tree
from .predict_model import evaluate_model

## Question - What is the purpose of the __init__.py file in each folder??

""" you can initialize setup like

def setup_environment():

    # Setup a default data path
    os.environ['DATA_PATH'] = '/Users/swapnilklkar/Documents/ML_code_with_Cookiecutter/data/raw/credit.csv'


setup_environment()

# Optional: Check for required dependencies
def check_dependencies():
    try:
        import numpy
        import pandas
        import scikit_learn
    except ImportError as e:
        logger.warning(f"Missing dependency: {e.name}. Some features may not work properly.")

check_dependencies()
    
"""



"""
The presence of __init__.py files ensures that directories are recognized as packages by Python, 
which is essential for importing modules from those directories."""