#Libraries
import warnings
import pickle
import pandas as pd
from tabulate import tabulate
# Machine Learning - Multi-Classification Prediction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix
, classification_report, precision_score, recall_score, f1_score)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from EDA import  EDA
def generateBarPlot(models, d1, title, filename):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[name for name, _ in models], y=list(d1.values()), palette="viridis")
    for i, v in enumerate(d1.values()):
        plt.text(i, v + 0.01, str(round(v, 4)), ha='center', va='bottom')
    plt.title('Comparison of Models')
    plt.xlabel('Models')
    plt.ylabel(title)
    plt.savefig("images/ML/"+filename+".png")
def chooseBestModelDictValue(d1):
    names = list(d1.keys())
    values = list(d1.values())
    return [names[values.index(max(values))],max(values)]

warnings.filterwarnings('ignore')
def ML():
    df1, genderDict, occupationDict, bmiDict, sleepDisorderDict = EDA()
    # Prepare the data
    X = df1.drop(['Person ID', 'Sleep Disorder'], axis=1)
    y = df1['Sleep Disorder']
    X.drop(['Age_bin'], axis=1, inplace=True)
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    print(tabulate(X_train.head(5), X_train.columns, tablefmt="psql"))
    # Create a pipeline with data preprocessing and classification model
    pipeline = Pipeline([('scaler', StandardScaler()),('clf', RandomForestClassifier())])
    # Define parameter grids for hyperparameter tuning
    param_grid = [
    {
        'clf': [RandomForestClassifier()],
        'clf__n_estimators': [100, 200, 300,400],
        'clf__max_depth': [None, 5, 10,15],
    },
    {
        'clf': [SVC()],
        'clf__kernel': ['linear', 'rbf'],
        'clf__C': [0.01,0.1, 1, 10],
    },
    {
        'clf': [LogisticRegression()],
        'clf__solver': ['liblinear', 'lbfgs'],
        'clf__C': [0.01,0.1, 1, 10],
    },
    {
        'clf': [KNeighborsClassifier()],
        'clf__n_neighbors': [3, 5, 7,9],
    },
    {
        'clf': [GradientBoostingClassifier()],
        'clf__n_estimators': [100, 200, 300,400],
        'clf__learning_rate': [0.01, 0.1, 1],
    },
    {
        'clf': [DecisionTreeClassifier()],
        'clf__max_depth': [None, 5, 10,15],
    }
]
    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    # Get the best model
    best_model = grid_search.best_estimator_
    # Calculate accuracy scores for each model
    models = [
            ('Random Forest', RandomForestClassifier()),
            ('SVM', SVC()),
            ('Logistic Regression', LogisticRegression()),
            ('KNN', KNeighborsClassifier()),
            ('Gradient Boosting', GradientBoostingClassifier()),
            ('Decision Tree', DecisionTreeClassifier())
        ]
    accuracy_scores = {}
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}
    for name, model in models:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="micro")
        recall = recall_score(y_test, y_pred, average="micro")
        f1 = f1_score(y_test, y_pred, average="micro")

        accuracy_scores[name] = accuracy
        precision_scores[name] = precision
        recall_scores[name] = recall
        f1_scores[name] = f1
    generateBarPlot(models, d1=accuracy_scores, title="Accuracy Score", filename="AccuracyScore")
    print("Best model based on accuracy: ", chooseBestModelDictValue(accuracy_scores)[0],", ",chooseBestModelDictValue(accuracy_scores)[1])
    generateBarPlot(models, d1=precision_scores, title="Precision Score", filename="PrecisionScore")
    print("Best model based on precision: ", chooseBestModelDictValue(precision_scores)[0],", ",chooseBestModelDictValue(precision_scores)[1])
    generateBarPlot(models, recall_scores, title="Recall Score", filename="RecallScore")
    print("Best model based on recall: ", chooseBestModelDictValue(recall_scores)[0],", ",chooseBestModelDictValue(recall_scores)[1])
    generateBarPlot(models, d1=f1_scores, title="F1 Score", filename="F1Score")
    print("Best model based on F1: ", chooseBestModelDictValue(f1_scores)[0],", ",chooseBestModelDictValue(f1_scores)[1])
    # Confusion Matrix and Class Distribution Visualization
    # Fit the best model on the entire dataset
    best_model.fit(X, y)

    # Predictions on the test set
    y_pred_test = best_model.predict(X_test)

    # Get unique classes from the dataset
    unique_classes = df1['Sleep Disorder'].unique()
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred_test, labels=unique_classes)

    # Display the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("images/ML/confusionmatrix.png")

    # Check for bias in the dataset
    class_distribution = df1['Sleep Disorder'].value_counts(normalize=True)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_distribution.index, y=class_distribution.values)
    plt.title("Class Distribution")
    plt.xlabel("Sleep Disorder")
    plt.ylabel("Percentage")
    plt.savefig("images/ML/classdistribution.png")

    with open ('Model/MLModel.pkl','wb') as file:
        pickle.dump(best_model, file)

    with pd.ExcelWriter("DataDictionary/Dictionary.xlsx", engine='xlsxwriter') as writer:
        genderDictdf = pd.DataFrame({'index': genderDict.keys(), 'value': genderDict.values()})
        genderDictdf.to_excel(writer, sheet_name="gender", index=False)

        occupationDictdf = pd.DataFrame.from_dict({'index': occupationDict.keys(), 'value': occupationDict.values()})
        occupationDictdf.to_excel(writer, sheet_name="occupation", index=False)

        bmiDictdf = pd.DataFrame.from_dict({'index': bmiDict.keys(), 'value': bmiDict.values()})
        bmiDictdf.to_excel(writer, sheet_name="bmi", index=False)

        sleepDisorderDictdf = pd.DataFrame.from_dict({'index': sleepDisorderDict.keys(), 'value': sleepDisorderDict.values()})
        sleepDisorderDictdf .to_excel(writer, sheet_name="sleepDisorder", index=False)
#ML()

