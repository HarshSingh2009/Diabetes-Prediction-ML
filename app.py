from ML_Pipeline import MlPipeline
import streamlit as st

st.title('Diabetes Prediction Using ML')
st.subheader('Please fill the form to know if you have diabetes or not : ')

with st.sidebar:
    st.subheader('Chose your ML model: ')
    choices = st.selectbox('Select your model', ['Logistic Regression', 'K-Nearest Neighbors', 'SVM'])


with st.form(key='form1'):
    age = st.slider('Pick your Age : ', 0, 130)
    pregnancies = st.text_input('Pregnancies : ')
    glucose = st.text_input('Glucose : ')
    blood_pressure = st.text_input('Blood Pressure : ')
    skin_thickness = st.text_input('Skin Thickness : ')
    insulin = st.text_input('Insulin : ')
    bmi = st.text_input('Body Mass Index (BMI) : ')
    dpf = st.text_input('Diabetes Pedigree Function : ')

    if pregnancies:
        pregnancies = int(pregnancies)
    if glucose:
        glucose = int(glucose)
    if blood_pressure:
        blood_pressure =  int(blood_pressure)
    if skin_thickness:
        skin_thickness = int(skin_thickness)
    if insulin:
        insulin = int(insulin)
    if bmi:
        bmi = float(bmi)
    if dpf:
        dpf = float(dpf)

    data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

    predict_button = st.form_submit_button(label='Predict')
    if predict_button:
        ml_pipeline = MlPipeline(data=data)
        prediction = ml_pipeline.predict(choices=choices)
        if prediction[0] == 0:
            st.success(f'{choices} : You don\'t have diabetes.')
        else:
            st.text(f'{choices} : You have Diabetes')
