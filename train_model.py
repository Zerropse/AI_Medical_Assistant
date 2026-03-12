import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("dataset/training.csv")

df.fillna("", inplace=True)

symptom_columns = df.columns[1:]

symptoms = set()

for col in symptom_columns:
    for item in df[col].unique():
        item = str(item).strip().lower()
        if item != "":
            symptoms.add(item)

symptoms = sorted(list(symptoms))

X = []
y = []

for index, row in df.iterrows():

    symptom_vector = [0] * len(symptoms)

    row_symptoms = row[1:].values

    for symptom in row_symptoms:

        symptom = str(symptom).strip().lower()

        if symptom in symptoms:
            symptom_vector[symptoms.index(symptom)] = 1

    X.append(symptom_vector)
    y.append(row["Disease"])

model = RandomForestClassifier()

model.fit(X, y)

joblib.dump(model, "model/disease_model.pkl")
joblib.dump(symptoms, "model/symptoms.pkl")

print("Model trained successfully.")