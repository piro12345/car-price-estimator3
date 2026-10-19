import pickle
from pathlib import Path

import streamlit as st
from Orange.data import Domain, Table, ContinuousVariable, DiscreteVariable

st.set_page_config(page_title="Second-Hand Car Price Estimator", page_icon="🚗", layout="centered")
MODEL_PATH = Path(__file__).parent / "random_forest.pkcls"

@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)

st.title("car price estimator")
st.write(
    "estimate a used car's selling price using the random forest model saved from orange data mining."
)
st.caption("")

try:
    model = load_model()
except Exception as exc:
    st.error("The saved Orange model could not be loaded. Check the installation and Orange3 version.")
    st.exception(exc)
    st.stop()

# IMPORTANT: Supply the original features from Select Columns, not the model's
# internally transformed/one-hot attributes. Orange can then apply its own
# saved transformations to these original variables.
fuel_values = ["Diesel", "Petrol", "CNG"]
seller_values = ["Dealer", "Individual"]
transmission_values = ["Automatic", "Manual"]

input_domain = Domain([
    ContinuousVariable("Year"),
    ContinuousVariable("Present_Price"),
    ContinuousVariable("Kms_Driven"),
    DiscreteVariable("Fuel_Type", values=fuel_values),
    DiscreteVariable("Seller_Type", values=seller_values),
    DiscreteVariable("Transmission", values=transmission_values),
    ContinuousVariable("Owner"),
])

st.subheader("Enter car details")
with st.form("car_inputs"):
    left, right = st.columns(2)
    with left:
        year = st.number_input("year of purchase", min_value=1990, max_value=2035, value=2017, step=1)
        present_price = st.number_input("current price (₹ lakhs)", min_value=0.1, max_value=500.0, value=8.0, step=0.1)
        kms_driven = st.number_input("kilometres driven", min_value=0, max_value=1_000_000, value=30000, step=1000)
        owner = st.selectbox("no. of Previous owners", options=[0, 1, 2, 3])
    with right:
        fuel_type = st.selectbox("fuel type", options=fuel_values)
        seller_type = st.selectbox("seller type", options=seller_values)
        transmission = st.selectbox("transmission", options=transmission_values)
    submitted = st.form_submit_button("predict selling price", type="primary", use_container_width=True)

if submitted:
    try:
        row = [[
            float(year),
            float(present_price),
            float(kms_driven),
            float(fuel_values.index(fuel_type)),
            float(seller_values.index(seller_type)),
            float(transmission_values.index(transmission)),
            float(owner),
        ]]
        data = Table.from_list(input_domain, row)
        prediction = model(data)
        price = float(prediction[0])
        if price != price:
            st.error("The model returned an invalid prediction. Please check the inputs.")
        else:
            st.success(f"Estimated selling price: ₹{price:.2f} lakh")
            st.caption("Prediction made by Orange Random Forest model.")
    except Exception as exc:
        st.error("Prediction still failed. Please send this error so we can check the saved model's transformation.")
        st.exception(exc)

