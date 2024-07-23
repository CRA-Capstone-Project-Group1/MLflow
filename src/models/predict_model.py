# Import accuracy score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# # Function to predict and evaluate
def evaluate_model(model, X_test_scaled, y_test):
    # Predict the loan eligibility on the testing set
    y_pred = model.predict(X_test_scaled)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate precision
    precision = precision_score(y_test, y_pred)
    
    # Calculate recall
    recall = recall_score(y_test, y_pred)

    # Calculate the confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)

    return {'Accuracy': accuracy, 'Precision':precision, 'Recall':recall, 'Confusion Matrix':confusion_mat}
