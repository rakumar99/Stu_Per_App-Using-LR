import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder

def load_model():
    with open("Student-Performance.pkl","rb") as file:
        model,scaler,le=pickle.load(file)
    return model,scaler,le

def preprocessing_input_data(data,scaler,le):
    data["Extracurricular Activities"]=le.transform([data["Extracurricular Activities"]])[0]
    df=pd.DataFrame([data])
    df_transformed=scaler.transform(df)
    return df_transformed



def predict_data(data):
    model,scaler,le=load_model()
    processed_data=preprocessing_input_data(data,scaler,le)
    prediction=model.predict(processed_data)
    return prediction

def main():
    st.title("student performance prediction")
    st.write("enter your data to get a prediction for your performance")

    hours_studied=st.number_input("Hours Studied", min_value=1,max_value=10,value=5)
    previous_score=st.number_input("Previous Scores",min_value=40,max_value=100,value=70)
    Extra=st.selectbox("Extracurricular Activities", ['Yes','No'])
    sleep_hours=st.number_input("Sleep Hours",min_value=4,max_value=10,value=7)
    Paper_solved=st.number_input("Sample Question Papers Practiced",min_value=0,max_value=10,value=5)

    if st.button("predict_your_performance"):
        user_data={
          "Hours Studied": hours_studied,
          "Previous Scores":previous_score,
          "Extracurricular Activities":Extra,
          "Sleep Hours": sleep_hours,
          "Sample Question Papers Practiced":Paper_solved
        }
        prediction=predict_data(user_data)
        st.success(f"your prediction result is {prediction}")


                    		
if __name__ == "__main__":
    main()



    
    




