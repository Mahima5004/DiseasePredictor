import streamlit as st
import pandas as pd
from urllib.error import URLError
import pandas as pd
import csv
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




file = open('final_dataset.csv')
csv_file = csv.reader(file)

symptom_file = open('archive/Symptom-severity.csv')
symptom_csv = csv.reader(symptom_file)

precautions = open('archive/symptom_precaution.csv')
prec_csv = csv.reader(precautions)

header = next(csv_file)
symptoms_header = next(symptom_csv)


symptom_hasher = {}
count = 0

for symptom in symptom_csv:
    symptom_hasher[symptom[0]] = count #building hash table for symptoms using symptom severity file
    count += 1

rows = []  

for row in csv_file:
    rows.append(row) #stroring the final dataset


disease_symptoms = []

for row in rows:
    x = []
    x.append(row[0]) #storing all the diseases from the final csv file

    for i in range(17): #as final csv file contains maximum of 17 symptom columns
        if row[i + 1] != '':
            a = row[i + 1].strip() 
            b = a.split(" ") 
            c = ''.join(b)
            x.append(symptom_hasher[c]) #after each disease storing the hash value of their symptoms

    disease_symptoms.append(x) #list of rows with the disease value on first position preceding by symptoms hash value using symptoms


final_data = []

for disease in disease_symptoms:
  symptom_indices = disease[1:]
  i = len(symptom_indices)
  zeros = [0] * 133
  for j in range(i):
    zeros[symptom_indices[j]] = 1
  vector = [disease[0]] + zeros
  final_data.append(vector)



df = pd.DataFrame(final_data,columns = ['Disease'] + [i for i in range(133)])



clean_data = df.drop(117,axis='columns') #beacuse of ambiguity

inputs = clean_data.drop("Disease",axis='columns')
target = clean_data["Disease"]


le_disease = LabelEncoder()

target['Disease_final'] = le_disease.fit_transform(target)


final_target = pd.DataFrame(target["Disease_final"],columns=["Disease"])



X_train,X_test,Y_train,Y_test = train_test_split(inputs,final_target, test_size= 0.85)

model = tree.DecisionTreeClassifier()
model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)

accuracy_score=accuracy_score(Y_test, Y_pred)




st.set_page_config(page_title='Disease Predictor', page_icon = "👩‍⚕️")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)


def predict_disease(inp):
    return le_disease.inverse_transform(model.predict([inp]))

def get_symptoms():
    df_simp = pd.DataFrame(list(symptom_hasher.keys()),columns=['Symptoms'])
    return df_simp

try:
    df = get_symptoms()
    st.title("Let us help you for Better Treatment🩺")
    input_Simps = st.multiselect(
        "Choose Symptoms which are causing you problem", list(df['Symptoms'])
    )
    if not input_Simps:
        st.success("There seems no symptoms? That means You're healthy 👍😊❤️‍🩹")
    else:
        inp = [0]*132
        for simp in input_Simps:
            inp[symptom_hasher[simp]] = 1

        x = predict_disease(inp)[0]
        st.write(f'I think you are suffering from {x}.')

        message = ""

        for disease in prec_csv:
            if disease[0].lower() == x.lower():
                out = "For now you should follow these precautions : \n"
                for precaution in disease[1:]:
                    if precaution != "":
                        out = out + precaution.capitalize() + ", \n"

                message = out+"Afterwords you should book an appointment via MediHelp."
                break

        st.info(message)
        #st.write(accuracy_score)
        



except URLError as e:
    st.error(
        """
        **internet access require.**

        Connection error: %s
    """
        % e.reason
    )
