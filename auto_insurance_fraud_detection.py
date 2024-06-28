# Importing the libraries for the project
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.metrics
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# 1. Loading the CSV file into Pandas DataFrame
# Getting a quick overview of the data (first 5 rows, data types, summary statistics, column names, unique value
df = pd.read_csv("/content/insurance.csv")
df.head(5) # checking the first 5 rows of the dataframe
df.info() # checking the Dtype for the dataframe
df.describe() # checking the count, mean, std, min of the dataframe
df.columns # checking all the columns available in the dataframe
df.nunique() #checking unique values
df.shape # checking the shape of the dataframe

# Producing Heatmap using Seaborn
# To find out how strongly each variable is related to every other variable in the dataframe. 
plt.figure(figsize=(7,6))
dataplot = sns.heatmap(df.corr())



# 2. Data Visualization - to visualize different aspect of the data 

# 2A. Producing Bar Plot for Fraud Reported Count 
# Aim: To see the total fraud reported during the insurance claim

# Set the style and color palette
plt.style.use('fivethirtyeight')
sns.set_palette("Set2")

# Create the count plot
ax = sns.countplot(x='fraud_reported', data=df, hue='fraud_reported')

# Customize the plot
ax.set_title("Fraud Reported Count")
ax.set_xlabel("Fraud Reported")
ax.set_ylabel("Count")

# Add annotations
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 3, '{:.1f}%'.format((height / total) * 100), ha="center")

# Show the plot
plt.show()

df['fraud_reported'].value_counts() # Count number of frauds vs non-frauds



# 2B: Producing Bar Plot for Fraud Reported by Incident State 
# Aim: To see which state has the highest fraud report

# Set the style and color palette
plt.style.use('fivethirtyeight')
sns.set_palette("Set2")

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Group and plot the data
df.groupby('incident_state').fraud_reported.count().plot(kind='bar', ylim=0, ax=ax)

# Customize the plot
ax.set_title("Fraud Reported by Incident State")
ax.set_xlabel("Incident State")
ax.set_ylabel("Fraud Reported")

# Add annotations
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 3, '{:.1f}%'.format((height / total) * 100), ha="center")

# Show the plot
plt.show()




# 2C: Producing Bar Plot for Fraud Reported by Incident Type
# Aim: To see which type of incident has the highest fraud reports

# Set the style and color palette
plt.style.use('fivethirtyeight')
sns.set_palette("Set2")

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Group and plot the data
df.groupby('incident_type').fraud_reported.count().plot(kind='bar', ylim=0, ax=ax)

# Customize the plot
ax.set_title("Fraud Reported by Incident Type")
ax.set_xlabel("Incident Type")
ax.set_ylabel("Fraud Reported")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

# Add annotations
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 3, '{:.1f}%'.format((height / total) * 100), ha="center")

# Show the plot
plt.show()



# 2D: Creating Cluster Bar Plot for Average Vehicle Claim by Insured's Education Level
# Aim: To see which education level class has the highest fraud report in insurance claim

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Group and calculate the average vehicle claim by insured's education level, grouped by fraud reported
average_claim = df.groupby(['insured_education_level', 'fraud_reported'])['total_claim_amount'].mean().unstack()

# Plot the data
average_claim.plot(kind='barh', ax=ax)

# Customize the plot
ax.set_title("Average Vehicle Claim by Insured's Education Level")
ax.set_xlabel("Average Vehicle Claim")
ax.set_ylabel("Insured's Education Level")

# Add legend
ax.legend(title='Fraud Reported')

# Move the legend to the lower right
plt.legend(bbox_to_anchor =(1.1,0), loc='lower right')
plt.tight_layout()

# Show the plot
plt.show()



# 3. Data Preprocessing - Selecting useful column & handling data imbalance

# 3A: Selecting useful column 
# Dropping columns based on result from the graph plotted
df = df.drop(['_c39','policy_csl','incident_location', 'policy_bind_date', 'incident_date', 'auto_model', 'insured_occupation', 'policy_number'], axis=1)

# Viewing the new dataframe with dropped column
df.columns

#Plotting our target values
import matplotlib.pyplot as plt

# Count the occurrences of each class in the target variable
class_counts = df['fraud_reported'].value_counts()

# Create a bar plot to visualize the class distribution
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Fraud Reported')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

#The plot shows that there is class imbalance problem, so lets deal with it



# 3B: Class Imbalance 
# Aim: Address the imbalance issue by oversampling the minority class 
from imblearn.over_sampling import RandomOverSampler

# Separate the features and the target variable
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

# Apply Random Over-Sampling Examples (ROSE) to resample the target variable
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X, y)

# Convert the resampled data back to a DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['fraud_reported'] = y_resampled

df_resampled.head()

#Now lets check the target classes
#Plotting our target values
import matplotlib.pyplot as plt

# Count the occurrences of each class in the target variable
class_counts = df_resampled['fraud_reported'].value_counts()

# Create a bar plot to visualize the class distribution
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Fraud Reported')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

# Plot shows balance class for fraud reported and fraud not reported during the insurance claim




# 4 - Machine Learning Modeling 
# Aim: Applying the Machine Learning Algorithm (Random Forest, KNN, SVM Logistic Regression, Decision Tree)
# Selecting the best algorithm based on evaluation metrics F-1 score & Accuracy that is calculated using k-fold cross-validation

# 4A: Applying the Machine Learning Algorithm (Random Forest, KNN, SVM Logistic Regression, Decision Tree)
# Aim: Selecting the best algorithm based on evaluation metrics F-1 score & Accuracy that is calculated using k-fold cross-validation
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# Prepare the data
X = df_resampled.drop('fraud_reported', axis=1)
y = df_resampled['fraud_reported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the feature engineering pipeline
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessing = [
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('scaling', StandardScaler(), numerical_features)
]
preprocessor = ColumnTransformer(transformers=preprocessing)

# Initialize the classifiers
classifiers = [
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC(random_state=42)),
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42))
]

# Define the custom scoring function for F1-score
f1_scorer = make_scorer(f1_score, pos_label='Y')

# Train and evaluate each classifier using k-fold cross-validation
results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for clf_name, clf in classifiers:
    clf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    f1_scores = cross_val_score(clf_pipeline, X_train, y_train, cv=kf, scoring=f1_scorer)  # Use the custom F1 scorer
    accuracy_scores = cross_val_score(clf_pipeline, X_train, y_train, cv=kf, scoring='accuracy')
    results[clf_name] = {'F1-score': f1_scores, 'Accuracy': accuracy_scores}

# Print the k-fold scores for each classifier
for clf_name, metrics in results.items():
    f1_percentage = [score * 100 for score in metrics['F1-score']]
    accuracy_percentage = [score * 100 for score in metrics['Accuracy']]

    print(f"{clf_name}:")
    print(f"   F1-scores (%): {', '.join([f'{score:.2f}' for score in f1_percentage])}")
    print(f"   Accuracy (%): {', '.join([f'{score:.2f}' for score in accuracy_percentage])}")
    print()

# Find the best algorithm based on average F1-score
best_algorithm = max(results, key=lambda x: np.mean(results[x]['F1-score']))
print(f"The best algorithm based on average F1-score is: {best_algorithm}")



# 4B: Fitting the entire pipeline with the training data 
# Aim: To print the sample predictions and actual result when using the algorithm with the best F-1 and Accuracy score
X_test.columns

# Fit the entire pipeline with the training data
clf_pipeline.fit(X_train, y_train)

# Make predictions for 10 examples
sample_predictions = clf_pipeline.predict(X_test[:10])
actual_results = y_test[:10]

# Print the sample predictions and actual results
print("Sample Predictions vs Actual Results:")
for i in range(10):
    print(f"Example {i+1}:")
    print(f"   Prediction: {sample_predictions[i]}")
    print(f"   Actual Result: {actual_results.iloc[i]}")
    print()





# 5: Feature Engineering - Intergrating the best ML Algorithm with Feature Engineering

# 5A: Using the Heatmap produced to select constant and variable attributes in application of Feature Engineering
# Aim: Applying the best ML algorithm with selected constant and variable attributes
# Same as base model, F-1 and Accuracy score are used as evaluation metrics

# Prepare the data
X = df_resampled.drop('fraud_reported', axis=1)
y = df_resampled['fraud_reported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the custom transformer to handle missing values

class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['collision_type'].fillna('Unknown', inplace=True)
        X_copy['property_damage'].fillna('Unknown', inplace=True)
        X_copy['police_report_available'].fillna('Unknown', inplace=True)
        X_copy['bodily_injuries'].fillna('Unknown', inplace=True)
        X_copy['number_of_vehicles_involved'].fillna('Unknown', inplace=True)
        return X_copy

# Define the feature engineering pipelines with specific attribute groupings

pipeline_1 = make_pipeline(
    CustomTransformer(),
    ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['incident_hour_of_the_day', 'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim']),
            ('num', StandardScaler(), ['incident_hour_of_the_day'])
        ]
    ),
    RandomForestClassifier(random_state=42)
)

pipeline_2 = make_pipeline(
    CustomTransformer(),
    ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['age', 'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim']),
            ('num', StandardScaler(), ['age'])
        ]
    ),
    RandomForestClassifier(random_state=42)
)

pipeline_3 = make_pipeline(
    CustomTransformer(),
    ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['number_of_vehicles_involved', 'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim']),
            ('num', StandardScaler(), ['number_of_vehicles_involved'])
        ]
    ),
    RandomForestClassifier(random_state=42)
)

pipeline_4 = make_pipeline(
    CustomTransformer(),
    ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['months_as_customer', 'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim']),
            ('num', StandardScaler(), ['months_as_customer'])
        ]
    ),
    RandomForestClassifier(random_state=42)
)

pipelines = [
    ('Pipeline 1', pipeline_1),
    ('Pipeline 2', pipeline_2),
    ('Pipeline 3', pipeline_3),
    ('Pipeline 4', pipeline_4)
]


# Define the custom scoring function for F1-score
f1_scorer = make_scorer(f1_score, pos_label='Y')

results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for pipeline_name, pipeline in pipelines:
    f1_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring=f1_scorer)  # Use k-fold cross-validation for F1-score
    accuracy_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='accuracy')  # Use k-fold cross-validation for accuracy
    results[pipeline_name] = {'F1-score': f1_scores, 'Accuracy': accuracy_scores}

# Print the k-fold scores for each pipeline
for pipeline_name, metrics in results.items():
    f1_percentage = [score * 100 for score in metrics['F1-score']]
    accuracy_percentage = [score * 100 for score in metrics['Accuracy']]

    print(f"{pipeline_name}:")
    print(f"   F1-scores (%): {', '.join([f'{score:.2f}' for score in f1_percentage])}")
    print(f"   Accuracy (%): {', '.join([f'{score:.2f}' for score in accuracy_percentage])}")
    print()

# Find the best pipeline based on average F1-score
best_pipeline = max(results, key=lambda x: np.mean(results[x]['F1-score']))
print(f"The best pipeline is: {best_pipeline}")




# 6: Artificial Neural Network - Intergrate the best ML algorithm and FE pipelines with Neural Network
# 6A: Utilizing TensorFlow to build and train a neural network 
# Aim: Applying the best ML algorithm and FE pipelines on the Neural Network 
# Aim #2: To see how much accuracy was improved from the ML base model and FE pipelines. 

# Prepare the data
X = df_resampled.drop('fraud_reported', axis=1)
y = df_resampled['fraud_reported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the target variable to numeric format
y_train = y_train.replace({'Y': 1, 'N': 0})
y_test = y_test.replace({'Y': 1, 'N': 0})

# Define the feature engineering pipeline
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessing = [
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('scaling', StandardScaler(), numerical_features)
]
preprocessor = ColumnTransformer(transformers=preprocessing)

# Preprocess the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Convert the sparse matrices to dense matrices
X_train_preprocessed = X_train_preprocessed.toarray()
X_test_preprocessed = X_test_preprocessed.toarray()

# Build the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_preprocessed.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Define the number of folds for k-fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


# Train and evaluate the model using k-fold cross-validation
f1_scores = []
accuracy_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train_preprocessed)):
    X_train_fold, X_val_fold = X_train_preprocessed[train_index], X_train_preprocessed[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    model.fit(X_train_fold, y_train_fold, epochs=3, batch_size=16, verbose=0)

    y_val_pred_probs = model.predict(X_val_fold)
    y_val_pred_labels = (y_val_pred_probs >= 0.5).astype(int)  # Convert probabilities to 0 or 1

    # Convert the y_test array to contain string labels
    y_pred_labels = ['Y' if prob >= 0.5 else 'N' for prob in y_val_pred_probs]
    y_test_labels = ['Y' if label == 1 else 'N' for label in y_val_pred_labels]

    # Calculates the f1 and accuracy score by fold
    f1_fold = f1_score(y_val_fold, y_val_pred_labels)
    accuracy_fold = accuracy_score(y_val_fold, y_val_pred_labels)

    print(f"Fold {fold+1} - F1-score: {f1_fold*100:.2f}%, Accuracy: {accuracy_fold*100:.2f}%")

    f1_scores.append(f1_fold)
    accuracy_scores.append(accuracy_fold)



# Print the F1 scores and accuracy for each fold
print("F1 Scores for each fold:", [f"{score*100:.2f}%" for score in f1_scores])
print("Accuracy Scores for each fold:", [f"{score*100:.2f}%" for score in accuracy_scores])

# Sample Predictions vs Actual Results
print("Sample Predictions vs Actual Results:")
for i in range(20):
    print(f"Example {i+1}:")
    print(f"   Prediction: {y_pred_labels[i]}")
    print(f"   Actual Result: {y_test_labels[i]}")





# 7. Data Visualization of F-1 and Accuracy score for ML, FE and Neural Network
# 7A: Creating table and bar plot of F-1 & Accuracy result for the best ML lgorithm

# Machine learning F1 score
ML_f1_result = pd.DataFrame(columns = ['K-Fold', 'RF', 'KNN', 'SVM', 'LR', 'DT'])


ml_f1_data = [
    ('Fold_1', 89.06, 68.48, 87.45, 83.95, 88.80, ),
    ('Fold_2', 91.50, 66.67, 85.96, 83.76, 89.52, ),
    ('Fold_3', 94.12, 68.57, 90.52, 84.75, 90.24, ),
    ('Fold_4', 91.57, 66.41, 90.08, 89.26, 90.76, ),
    ('Fold_5', 90.55, 63.49, 86.85, 81.93, 87.90)
]

ML_f1_result = pd.DataFrame.from_records(ml_f1_data, columns=ML_f1_result.columns)

# Rearranging index
ML_f1_result.index = np.arange(1, len(ML_f1_result) + 1)

ML_f1_result

# Create the bar plot
ML_f1_result.plot(kind='bar', figsize=(20, 10))

# Add labels and title
plt.xlabel('K-Fold Count')
plt.ylabel('Scores %')
plt.title('Machine Learning - F1 Score')

# Move the legend to the lower right
plt.legend(bbox_to_anchor =(1.1,0), loc='lower right')
plt.tight_layout()

# Show the plot
plt.show()

# Machine learning Accuracy score
ML_acc_result = pd.DataFrame(columns = ['K-Fold', 'RF', 'KNN', 'SVM', 'LR', 'DT'])


ml_acc_data = [
    ('Fold_1', 88.38, 66.39, 87.14, 83.82, 88.38, ),
    ('Fold_2', 91.29, 68.05, 86.31, 84.23, 89.21, ),
    ('Fold_3', 94.19, 68.05, 90.87, 85.06, 90.04, ),
    ('Fold_4', 91.29, 64.32, 90.04, 89.21, 90.46, ),
    ('Fold_5', 90.00, 61.67, 86.25, 81.25, 87.50)
]

ML_acc_result = pd.DataFrame.from_records(ml_acc_data, columns=ML_acc_result.columns)


# Rearranging index
ML_acc_result.index = np.arange(1, len(ML_acc_result) + 1)

ML_acc_result

# Create the bar plot
ML_acc_result.plot(kind='bar', figsize=(20, 10))

# Add labels and title
plt.xlabel('K-Fold Count')
plt.ylabel('Scores %')
plt.title('Machine Learning - Accuracy Score')

# Move the legend to the lower right
plt.legend(bbox_to_anchor =(1.1,0), loc='lower right')
plt.tight_layout()

# Show the plot
plt.show()

# F-1 Score - Random Forest

ML_f1_result.plot(y=["RF"],
        kind="bar", figsize=(20, 10))

# Add labels and title
plt.xlabel('K-Fold Count')
plt.ylabel('Scores %')
plt.title('Random Forest - F-1 Score')

# Move the legend to the lower right
plt.legend(bbox_to_anchor =(1.1,0), loc='lower right')
plt.tight_layout()

# Show the plot
plt.show()

# Accuracy Score - Random Forest
colour_code = '#FFD39B'

ML_acc_result.plot(y=["RF"],
        kind="bar", figsize=(20, 10), color = colour_code)

# Add labels and title
plt.xlabel('K-Fold Count')
plt.ylabel('Scores %')
plt.title('Random Forest - Accuracy Score')

# Move the legend to the lower right
plt.legend(bbox_to_anchor =(1.1,0), loc='lower right')
plt.tight_layout()

# Show the plot
plt.show()






# 7B: Creating table and bar plot of F-1 & Accuracy result for the Feature Engineering pipelines

# Feature Engineering F1 score
FE_f1_result = pd.DataFrame(columns = ['K-Fold', 'Pipeline_1', 'Pipeline_2', 'Pipeline_3', 'Pipeline_4'])


fe_f1_data = [
    ('Fold_1', 87.61, 87.61, 88.00, 88.00),
    ('Fold_2', 87.39, 88.39, 87.27, 87.39),
    ('Fold_3', 94.50, 94.50, 91.74, 93.64),
    ('Fold_4', 87.22, 88.79, 87.00, 88.79),
    ('Fold_5', 87.22, 87.34, 85.96, 87.22)
]

FE_f1_result = pd.DataFrame.from_records(fe_f1_data, columns=FE_f1_result.columns)

# Rearranging index
FE_f1_result.index = np.arange(1, len(FE_f1_result) + 1)

FE_f1_result

# Create the bar plot
FE_f1_result.plot(kind='bar', figsize=(20, 10))

# Add labels and title
plt.xlabel('K-Fold Count')
plt.ylabel('Scores %')
plt.title('Feature Engineering - F1 Score')

# Move the legend to the lower right
plt.legend(bbox_to_anchor =(1.1,0), loc='lower right')
plt.tight_layout()

# Show the plot
plt.show()

# Feature Engineering Accuracy score
FE_acc_result = pd.DataFrame(columns = ['K-Fold', 'Pipeline_1', 'Pipeline_2', 'Pipeline_3', 'Pipeline_4'])


fe_acc_data = [
    ('Fold_1', 88.38, 88.38, 88.80, 88.80, ),
    ('Fold_2', 88.38, 89.21, 88.38, 88.38, ),
    ('Fold_3', 95.02, 95.02, 92.53, 94.19, ),
    ('Fold_4', 87.97, 89.63, 87.97, 89.63, ),
    ('Fold_5', 87.92, 87.92, 86.67, 87.92)
]

FE_acc_result = pd.DataFrame.from_records(fe_acc_data, columns=FE_acc_result.columns)

# Rearranging index
FE_acc_result.index = np.arange(1, len(FE_acc_result) + 1)

FE_acc_result

# Create the bar plot
FE_acc_result.plot(kind='bar', figsize=(20, 10))

# Add labels and title
plt.xlabel('K-Fold Count')
plt.ylabel('Scores %')
plt.title('Feature Engineering - Accuracy Score')

# Move the legend to the lower right
plt.legend(bbox_to_anchor =(1.1,0), loc='lower right')
plt.tight_layout()

# Show the plot
plt.show()

# Feature Engineering - Pipeline 2
colour_code = "#FF7F24"
FE_f1_result.plot(y=["Pipeline_2"],
        kind="bar", figsize=(20, 10), color=colour_code)

# Add labels and title
plt.xlabel('K-Fold Count')
plt.ylabel('Scores %')
plt.title('F-1 Score - Pipeline 2')

# Move the legend to the lower right
plt.legend(bbox_to_anchor =(1.1,0), loc='lower right')
plt.tight_layout()

# Show the plot
plt.show()

# Feature Engineering - Pipeline 2
colour_code = "#E9967A"
FE_acc_result.plot(y=["Pipeline_2"],
        kind="bar", figsize=(20, 10), color=colour_code)

# Add labels and title
plt.xlabel('K-Fold Count')
plt.ylabel('Scores %')
plt.title('Accuracy Score - Pipeline 2')

# Move the legend to the lower right
plt.legend(bbox_to_anchor =(1.1,0), loc='lower right')
plt.tight_layout()

# Show the plot
plt.show()








# 7C: Creating table and bar plot of F-1 & Accuracy result for the Artificial Neural Network 

# Artificial Neural Network (ANN)
ANN_f1_result = pd.DataFrame(columns = ['K-Fold', 'F1_ANN'])


ANN_f1_data = [
    ('Fold_1', 80.83),
    ('Fold_2', 89.54),
    ('Fold_3', 97.37),
    ('Fold_4', 98.76),
    ('Fold_5', 99.19)
]

ANN_f1_result = pd.DataFrame.from_records(ANN_f1_data, columns=ANN_f1_result.columns)

# Rearranging index
ANN_f1_result.index = np.arange(1, len(ANN_f1_result) + 1)

ANN_f1_result

# Artificial Neural Network (ANN)
ANN_acc_result = pd.DataFrame(columns = ['K-Fold', 'Accuracy_ANN'])


ANN_acc_data = [
    ('Fold_1', 80.91),
    ('Fold_2', 89.63),
    ('Fold_3', 97.51),
    ('Fold_4', 98.76),
    ('Fold_5', 99.17)
]

ANN_acc_result = pd.DataFrame.from_records(ANN_acc_data, columns=ANN_acc_result.columns)

# Rearranging index
ANN_acc_result.index = np.arange(1, len(ANN_acc_result) + 1)

ANN_acc_result

# Create the bar plot
colour_code_f1 = "#FFC0CB"
colour_code_acc = "#D02090"

fig, ax = plt.subplots(figsize=(20, 10))

# Plot F1 scores from ANN_f1_result DataFrame
ANN_f1_result.plot(kind='bar', color=colour_code_f1, ax=ax, position=0, width=0.4)

# Plot Accuracy scores from ANN_acc_result DataFrame
ANN_acc_result.plot(kind='bar', color=colour_code_acc, ax=ax, position=1, width=0.4)

# Add labels and title
plt.xlabel('K-Fold Count')
plt.ylabel('Scores %')
plt.title('F-1 Score and Accuracy Score of Artificial Neural Network')

# Move the legend to the lower right
plt.legend(['F1 Score', 'Accuracy'], bbox_to_anchor=(1.1, 0), loc='lower right')
plt.tight_layout()

# Show the plot
plt.show()



"""# Comparing Machine Learning (Random Forest), Feature Engineering (Pipeline 2) and Artificial Neural Network (ANN)"""

# Define color codes for each dataframe
colour_code_ml_f1 = "#FF8247"
colour_code_ml_acc = "#8B4726"
colour_code_fe_f1 = "#00E5EE"
colour_code_fe_acc = "#00868B"
colour_code_ann_f1 = "#FFC0CB"
colour_code_ann_acc = "#D02090"

# Select specific columns from each dataframe
mlf1_columns = ['K-Fold', 'RF']
mlacc_columns = ['K-Fold', 'RF']
fef1_columns = ['K-Fold', 'Pipeline_2']
feacc_columns = ['K-Fold', 'Pipeline_2']
annf1_columns = ['K-Fold', 'F1_ANN']
annacc_columns = ['K-Fold', 'Accuracy_ANN']


# Create the bar plots for each dataframe with selected columns
fig, ax = plt.subplots(figsize=(20, 10))

ML_f1_result[mlf1_columns].plot(kind='bar', color=colour_code_ml_f1, ax=ax, position=0, width=0.2)
ML_acc_result[mlacc_columns].plot(kind='bar', color=colour_code_ml_acc, ax=ax, position=1, width=0.2)
FE_f1_result[fef1_columns].plot(kind='bar', color=colour_code_fe_f1, ax=ax, position=1, width=0.2)
FE_acc_result[feacc_columns].plot(kind='bar', color=colour_code_fe_acc, ax=ax, position=0, width=0.2)
ANN_f1_result[annf1_columns].plot(kind='bar', color=colour_code_ann_f1, ax=ax, position=1, width=0.2)
ANN_acc_result[annacc_columns].plot(kind='bar', color=colour_code_ann_acc, ax=ax, position=0, width=0.2)


# Add labels and title
plt.xlabel('K-Fold Count')
plt.ylabel('Score %')
plt.title('Comparison of F-1 Score and Accuracy Score of Random Forest, Pipeline 2 and Artificial Neural Network')


# Move the legend to the lower right
plt.legend(['Random Forest F1', 'Random Forest Accuracy', 'Pipeline 2 F-1', 'Pipeline 2 Accuracy', 'ANN F-1', 'ANN Accuracy'], bbox_to_anchor=(1.1, 0), loc='lower right')
plt.tight_layout()

# Show the plot
plt.show()

# Merging the dataframe for numerical comparison
merged_f1 = pd.concat([ML_f1_result, FE_f1_result, ANN_f1_result], axis=1)

# Removing duplicates column
merged_f1 = merged_f1.loc[:, ~merged_f1.columns.duplicated()]

# Saving the excel file into computer
merged_f1.to_csv('insurance_f1.csv', index=False)

merged_f1

# Merging the dataframe for numerical comparison
merged_acc = pd.concat([ML_acc_result, FE_acc_result, ANN_acc_result], axis=1)

# Removing duplicates column
merged_acc = merged_acc.loc[:, ~merged_acc.columns.duplicated()]

# Saving the excel file into computer
merged_acc.to_csv('insurance_acc.csv', index=False)

merged_acc
