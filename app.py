# import numpy as nm
# import matplotlib.pyplot as mtp
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data (replace with your actual data path or upload functionality)
def load_data():
    try:
        data_set = pd.read_csv("C:/Users/ATHUMAN/Downloads/salary_data.csv")
        return data_set
    except FileNotFoundError:
        st.error("Salary data file 'salary_data.csv' not found. Please upload the file or adjust the path.")
        return None  # Indicate data loading failure

data = load_data()

# Check if data loading succeeded
if data is not None:
    st.title("Salary Prediction App by Mkamba")

    # Split data (optional, can be pre-calculated and loaded)
    def split_data(data):
        x = data.iloc[:, :-1].values
        y = data.iloc[:, 1].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)
        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = split_data(data.copy())

    # Train the model (optional, can be pre-trained and loaded)
    def train_model(x_train, y_train):
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)
        return regressor

    model = train_model(x_train, y_train)

    # Prediction (optional, can be combined with input form)
    def predict_salary(model, x):
        y_pred = model.predict([x])
        return y_pred[0]  # Extract the first element

    # User input for prediction (optional)
    if st.sidebar.checkbox("Predict Salary"):
        years_of_experience = st.sidebar.number_input("Enter Years of Experience")
        if years_of_experience:
            predicted_salary = predict_salary(model, [[years_of_experience]])
            st.write(f"Predicted Salary for {years_of_experience} years of experience: ${predicted_salary:.2f}")

    # Visualizations (optional, pre-trained or loaded model can be used)
    import matplotlib.pyplot as plt

    def plot_data(x_train, y_train, x_pred, y_pred):
        plt.figure(figsize=(10, 6))
        plt.scatter(x_train, y_train, color='green', label='Training Data')
        plt.plot(x_train, x_pred, color='red', label='Predicted Salary')
        plt.title('Salary vs Years of Experience (Training Dataset) Trained by Mkamba')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary (In Dollars)')
        plt.legend()
        st.pyplot(plt)  # Display the plot

    plot_data(x_train, y_train, x_train, model.predict(x_train))  # Show training data plot












#
# data_set = pd.read_csv("C:/Users/ATHUMAN/Downloads/salary_data.csv")
#
# x = data_set.iloc[:,:-1].values
# y = data_set.iloc[:,1].values
#
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3, random_state=0)
#
# regressor = LinearRegression()
# regressor.fit(x_train, y_train)
# y_pred = regressor.predict(x_test)
# x_pred = regressor.predict(x_train)
#
# mtp.scatter(x_train, y_train, color='green')
# mtp.plot(x_train, x_pred, color='red')
# mtp.title('Salary vs Years of Experience(Training Dataset) Trained by Mkamba')
# mtp.xlabel('Years of Experience')
# mtp.xlabel('Salary(In Dollars)')
# mtp.show()
