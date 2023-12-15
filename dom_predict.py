import streamlit as st
import pickle
import pandas as pd


def load_data():
    with open('HR_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_data()

model = data['model']
le_salary = data['le_salary']

def show_prediction():
    st.title("MACHINE LEARNING PREDICTION : LOGISTIC REGRESSION")
    Satisfaction = st.slider("Satisfaction Level : ", 0.0, 1.0, 0.70)
    Hours = st.slider("Monthly Hours (Average) : ", 90, 350, 100)
    Promotion = st.selectbox("Promotion (5 years ago) : ", (0,1))
    Salary = st.selectbox("Salary", ('low', 'medium', 'high'))
    butAct = st.button("Predict Status")

    if butAct:
        predictors = pd.DataFrame(
        {'satisfaction_level' : [Satisfaction],
        'average_montly_hours' : [Hours],
        'promotion_last_5years' : [Promotion],
        'salary' : [Salary]
        })

        st.dataframe(predictors)

        predictors.salary = le_salary.transform(predictors.salary)

        pred = model.predict(predictors)

        st.write(f"Employee might {'stay.' if pred == 0 else 'leave'}")