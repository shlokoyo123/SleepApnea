import pandas as pd
import pickle
from tabulate import tabulate
from EDA import dictConverter as DC
# User Input and Prediction
# Obtain model inputs
# Step1: Read in dictionaries
# if userinputval is 0 then single user input
# if userinputval is 1 then it is a batch input
import warnings
warnings.filterwarnings("ignore")

def convertInputFile(df, gd, fork):
    if fork == 0:
        df['Gender'] = df[['Gender']].apply(lambda x: gd["gender"][x[0]], axis=1)
        df['Occupation'] = df[['Occupation']].apply(lambda x: gd["occupation"][x[0]], axis=1)
        df['BMI Category'] = df[['BMI Category']].apply(lambda x: gd["bmi"][x[0]], axis=1)
        return  df
    elif fork == 1:
        revSleepDict  = {v: k for k, v in gd["sleepDisorder"].items()}
        #print(revSleepDict)
        df['SleepPredicted'] = df[['SleepPredicted']].apply(lambda x: revSleepDict[x[0]], axis=1)
        return df
def UserInput(userinputval):
    dictNames = pd.ExcelFile("DataDictionary//Dictionary.xlsx").sheet_names
    globalDict = {}
    for dict in dictNames:
        df = pd.read_excel("DataDictionary//Dictionary.xlsx", sheet_name=dict)
        globalDict[dict] = DC(df, 'index', 'value')
    #print(globalDict)
    #Step2: Read best model file
    # rb stands for read binary
    revSleepDisorderDict = {v:k for k, v in globalDict["sleepDisorder"].items()}
    with open("Model//MLModel.pkl",'rb') as file:
        model = pickle.load(file)
        #model.predict()

    # Obtain user inputs
    # Get user input
    if userinputval == 0:
        gender = input("Enter Gender (Male/Female): ")
        age = int(input("Enter Age: "))
        occupation = input("Enter Occupation: ")
        sleep_duration = float(input("Enter Sleep Duration: "))
        quality_of_sleep = float(input("Enter Quality of Sleep: "))
        physical_activity_level = int(input("Enter Physical Activity Level: "))
        stress_level = int(input("Enter Stress Level: "))
        bmi_category = input("Enter BMI Category: ")
        heart_rate = int(input("Enter Heart Rate: "))
        daily_steps = int(input("Enter Daily Steps: "))
        upper_bp = float(input("Enter Upper Blood Pressure: "))
        lower_bp = float(input("Enter Lower Blood Pressure: "))

        gender_mapping = globalDict["gender"].get(gender.capitalize(), -1)
        occupation_mapping = globalDict["occupation"].get(occupation.capitalize(), -1)
        bmi_category_mapping = globalDict["bmi"].get(bmi_category.capitalize(), -1)


        # Create a DataFrame with the user input
        user_data = pd.DataFrame({
            'Age': [age],
            'Sleep Duration': [sleep_duration],
            'Quality of Sleep': [quality_of_sleep],
            'Physical Activity Level': [physical_activity_level],
            'Stress Level': [stress_level],
            'Heart Rate': [heart_rate],
            'Daily Steps': [daily_steps],
            'Systolic (Blood Pressure)': [upper_bp],
            'Diastolic (Blood Pressure)': [lower_bp],
            'Gender': [gender_mapping],
            'Occupation': [occupation_mapping],
            'BMI Category': [bmi_category_mapping]
        })
        user_pred = model.predict(user_data)
        print("Predicted Sleep Disorder:", revSleepDisorderDict[int(user_pred[0])])
    elif userinputval == 1:
        file = input("File Name: \n")
        df = pd.read_csv("BatchInput//"+file)
        cleanDF = convertInputFile(df, globalDict, 0)
        cleanDF.drop(columns=["Sleep Disorder"], inplace=True)
        predF = pd.DataFrame({"SleepPredicted":model.predict(cleanDF)})
        predF = convertInputFile(predF, globalDict, 1)
        df = pd.read_csv("BatchInput//" + file)
        newDF = df.join(predF)
        newDF.to_csv("BatchOutput//" + file.split(".")[0] + "_predicted.csv", index=False)
        del df
        del cleanDF
        del predF
        del newDF

        #print(tabulate(newDF.head(5), newDF.columns, tablefmt="sql"))

processingType = input("Enter '0' for single user input and '1' for batch file processing: \n")
UserInput(userinputval=int(processingType))
