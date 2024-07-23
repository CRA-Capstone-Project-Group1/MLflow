from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle


# Function to train the model
def train_decision_tree(X, y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the model parameters
    max_depth = 3
    min_samples_split = 2
    min_samples_leaf = 1

    # Train the decision tree classifier model
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf).fit(X_train_scaled, y_train)
    
    # Save the trained model
    with open('models/decision_tree_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model, X_test_scaled, y_test, max_depth, min_samples_split, min_samples_leaf
