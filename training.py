import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle

# Load the Data
file_path = 'software_processes_and_tasks.xlsx'
df = pd.read_excel(file_path)

# Separate features and labels
X = df[['Process']]
y = df['Task Type']

# Encode the features (task descriptions)
preprocessor = ColumnTransformer(
    transformers=[
        ('text', OneHotEncoder(handle_unknown='ignore'), ['Process'])
    ],
    remainder='passthrough'
)

X_transformed = preprocessor.fit_transform(X)

# Encode the labels (task types)
label_encoder = LabelEncoder()
y_transformed = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=0.2, random_state=42)

# Set up the model with class weights
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Randomized search for hyperparameters
param_distributions = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20],
}
randomized_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=3, scoring='accuracy', random_state=42)
randomized_search.fit(X_train, y_train)

# Best model
best_model = randomized_search.best_estimator_

# Evaluate the Model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')

# Get unique classes for labels
labels = np.unique(y_test)

# Print classification report with zero_division parameter
print('Classification Report:')
print(classification_report(y_test, y_pred, labels=labels, target_names=label_encoder.classes_, zero_division=0))

# Save the model, preprocessor, and label encoder
with open('trained_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('preprocessor.pkl', 'wb') as preprocessor_file:
    pickle.dump(preprocessor, preprocessor_file)

with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)

# Function to predict task type
def predict_task_type(process_description):
    # Load the preprocessor, model, and label encoder
    with open('preprocessor.pkl', 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
    
    with open('trained_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('label_encoder.pkl', 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)
    
    # Prepare the input data
    new_process_df = pd.DataFrame({'Process': [process_description]})
    X_new_transformed = preprocessor.transform(new_process_df)
    
    # Predict
    y_new_pred = model.predict(X_new_transformed)
    predicted_label = label_encoder.inverse_transform(y_new_pred)
    
    return predicted_label[0]

# Example usage
if __name__ == "__main__":
    new_process_description = input("Enter a process description: ")
    predicted_task_type = predict_task_type(new_process_description)
    print(f'Process: {new_process_description}\nPredicted Task Type: {predicted_task_type}')
