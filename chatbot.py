import joblib

# Load model and symptoms
model = joblib.load("model/disease_model.pkl")
symptoms_list = joblib.load("model/symptoms.pkl")


def suggest_symptoms(user_input):

    suggestions = []

    for symptom in symptoms_list:
        if user_input in symptom:
            suggestions.append(symptom)

    return suggestions[:5]


def get_user_symptoms():

    user_symptoms = []

    print("\nEnter symptoms one by one.")
    print("Type 'done' when finished.\n")

    while True:

        symptom = input("Symptom: ").strip().lower().replace(" ", "_")

        if symptom == "done":
            break

        if symptom in symptoms_list:

            user_symptoms.append(symptom)

        else:

            suggestions = suggest_symptoms(symptom)

            if len(suggestions) > 0:

                print("\nDid you mean:")

                for i, s in enumerate(suggestions):
                    print(f"{i+1}. {s}")

                choice = input("Select number or press enter to skip: ")

                if choice.isdigit():

                    index = int(choice) - 1

                    if index < len(suggestions):
                        user_symptoms.append(suggestions[index])

            else:

                print("Symptom not recognized.")

    return user_symptoms


def predict_disease(user_symptoms):

    input_vector = [0] * len(symptoms_list)

    for symptom in user_symptoms:

        if symptom in symptoms_list:

            index = symptoms_list.index(symptom)
            input_vector[index] = 1

    prediction = model.predict([input_vector])

    return prediction[0]


def chatbot():

    print("\n==========================")
    print("   MEDICAL AI CHATBOT")
    print("==========================\n")

    symptoms = get_user_symptoms()

    if len(symptoms) == 0:

        print("No symptoms entered.")
        return

    disease = predict_disease(symptoms)

    print("\nPredicted Disease:", disease)


if __name__ == "__main__":
    chatbot()