import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import re
from models import get_model

from sklearn.metrics import mean_squared_error
from data import DataSet

# SET UP INPUT
st.title("ETS Model Equation")

dataset_option = st.sidebar.selectbox("dataset", ("australian_tourists", "S&P500"))
dataset = DataSet(dataset_option)
test_start = st.sidebar.selectbox("train_test_cutoff", dataset.test_start_options)

model_options = (
    "Prophet",
    "ARIMA(p=1, d=0, q=0)",
    "ARIMA(p=0, d=1, q=0) -- random walk",
    "ARIMA(p=0, d=1, q=1) -- simple exponential smoothing with growth",
    "ARIMA(p=10, d=0, q=0)",
    "ARIMA(p=1, d=1, q=0)",
    "ARIMA(p=10, d=1, q=0)",
    "ARIMA(p=1, d=1, q=1)",
    "ARIMA(p=10, d=1, q=1)",
    "ETS: additive trend, additive seasonality, 4 seasons",
    "ETS: additive trend, additive seasonality, 5 seasons",
    "ETS: additive trend, multiplicative seasonality, 4 seasons",
    "ETS: additive trend, multiplicative seasonality, 5 seasons",
)
model_option_input = st.selectbox("Model option", model_options)


def parse_model_options_box(model_option_input: str):
    if model_option_input.startswith("ETS"):
        groups = re.findall(
            r"^([A-z]+)\: ([A-z ]+)\, ([A-z ]+), ([A-z ,0-9]+)", model_option_input
        )[0]
        model_type = "ETS"
    if model_option_input.startswith("ARIMA"):
        groups = re.findall(
            r"p\=([0-9]+), d\=([0-9]+), q\=([0-9]+)", model_option_input
        )[0]
        model_type = "ARIMA"

    if model_option_input.startswith("Prophet"):
        groups = "empty"
        model_type = "Prophet"

    return model_type, groups


# GET MODEL AND FORECASTS
model_type, model_option_parsed = parse_model_options_box(model_option_input)

data = dataset.data
data_train = data.loc[data.index < test_start]
data_test = data.loc[data.index >= test_start]
model = get_model(model_type, model_option_parsed, data_train)
fit = model.fit()

st.latex(model.equation)

pred = fit.forecast(steps=(data_test.index.size))

mse_in_sample = mean_squared_error(data_train, fit.fittedvalues)
mse_out_of_sample = mean_squared_error(data_test, pred)


# CREATE PLOT
df_plot = pd.concat(
    [
        data.rename("actuals"),
        pred.rename("predictions"),
        fit.fittedvalues.rename("fitted_values"),
    ],
    axis=1,
)

fig, ax_eval = plt.subplots()


df_plot["actuals"].plot(label="actuals", legend=True, ax=ax_eval, alpha=0.5)
df_plot.loc[df_plot.index < test_start]["fitted_values"].plot(
    label="mean_in_sample", legend=True, ax=ax_eval
)
df_plot.loc[df_plot.index >= test_start]["predictions"].plot(
    label="mean_out_of_sample", legend=True, ax=ax_eval
)

st.title("Model Performance")
st.text(f"In-sample RMSE: {round(mse_in_sample, 2)}")
st.text(f"Out-of-sample RMSE: {round(mse_out_of_sample, 2)}")


st.pyplot(fig)

