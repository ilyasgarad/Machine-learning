import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

@st.cache_data 
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

model = RandomForestClassifier()
model.fit(df.iloc[:,:-1], df['species'])

st.title("Iris Species Prediction App")
st.markdown("""
This app predicts the species of an Iris flower based on its physical features (sepal length, sepal width, petal length, and petal width).
Adjust the sliders to input the flower's measurements, and the model will predict its species.
""")

st.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/Iris_versicolor_2.jpg", caption="Iris Flower", use_container_width=True)

st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

st.subheader("Prediction Result")
st.write(f"The predicted species of the Iris flower is: **{predicted_species}**")

st.markdown("""
### What is the Iris Dataset?
The **Iris dataset** is a famous dataset in machine learning, containing measurements of 150 Iris flowers. It includes 4 features: 
- Sepal length
- Sepal width
- Petal length
- Petal width

These features are used to classify the flowers into one of three species:
- Setosa
- Versicolor
- Virginica
""")

if st.button("Show Dataset"):
    st.write(df.head())
