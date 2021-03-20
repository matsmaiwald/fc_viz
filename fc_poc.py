import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from models import get_model

from sklearn.metrics import mean_squared_error
from data import DataSet
import datetime


st.title("ETS Model Equation")

dataset_option = st.sidebar.selectbox("dataset", ("australian_tourists", "S&P500"))
dataset = DataSet(dataset_option)
test_start = st.sidebar.selectbox("train_test_cutoff", dataset.test_start_options)
# test_start = datetime.datetime.strptime(test_start, "%Y")
model_options = (
    "ETS: additive trend, additive seasonality",
    "ETS: additive trend, multiplicative seasonality",
)
model_option_input = st.selectbox("Model option", model_options)


def parse_model_options_box(model_option_input: str):
    groups = re.findall(r"^([A-z]+)\: ([A-z ]+)\, ([A-z ]+)", model_option_input)[0]
    return groups


model_option_parsed = parse_model_options_box(model_option_input)


data = dataset.data
data_train = data.loc[data.index < test_start]
data_test = data.loc[data.index >= test_start]
model = get_model(model_option_parsed, data_train)
fit = model.fit()

st.latex(model.equation)
# pred = fit.get_prediction(start=test_start.date(), end=max(data.index)).summary_frame(
#     alpha=0.05
# )
pred = fit.forecast(steps=(data_test.index.size))

mse_in_sample = mean_squared_error(data_train, fit.fittedvalues)
mse_out_of_sample = mean_squared_error(data_test, pred)

df_plot = pd.concat(
    [
        data.rename("actuals"),
        pred.rename("predictions"),
        fit.fittedvalues.rename("fitted_values"),
    ],
    axis=1,
)

fig, ax_eval = plt.subplots()

# df_plot.loc[df_plot.index < test_start]["actuals"].plot(
#     label="actuals", legend=True, ax=ax_eval, alpha=0.5
# )

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

