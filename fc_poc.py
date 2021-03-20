import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from models import get_model

from sklearn.metrics import mean_squared_error
from data import (
    get_australian_tourist_data,
    get_fred_data,
    get_train_test_split_options,
)
import datetime


st.title("ETS Model Equation")

dataset_option = st.sidebar.selectbox("dataset", ("australian_tourists", "S&P500"))
test_start = st.sidebar.selectbox(
    "train_test_cutoff", (get_train_test_split_options(dataset_option))
)
# test_start = datetime.datetime.strptime(test_start, "%Y")
model_options = (
    "ETS: additive trend, additive seasonality",
    "ETS: additive trend, multiplicative seasonality",
)
model_option_input = st.sidebar.selectbox("Model option", model_options)


def parse_model_options_input(model_option_input: str):
    groups = re.findall(r"^([A-z]+)\: ([A-z ]+)\, ([A-z ]+)", model_option_input)[0]
    return groups


model_option_parsed = parse_model_options_input(model_option_input)

# def get_equation(trend_opt, seasonality_opt, damped_opt, error_opt) -> str:
#     add_block = "l_{t-1}"
#     mult_block = ""
#     error_block = r"+ \epsilon_t"
#     if error_option == "mul":
#         error_block = r"*(1+\epsilon_t)"
#     if trend_opt == "add":
#         add_block += "+ b_{t-1}"

#     if seasonality_opt == "add":
#         add_block += "+ s_{t-m}"
#     if seasonality_opt == "mul":
#         mult_block = "* s_{t-m}"
#     eqn = f"y_t = ({add_block}) {mult_block} {error_block}"
#     return eqn


# st.latex(
#     get_equation(
#         trend_opt=trend_option,
#         seasonality_opt=seasonality_option,
#         damped_opt=damped_trend,
#         error_opt=error_option,
#     )
# )

# if dataset_option == "australian_tourists":
#     data, data_plot = get_australian_tourist_data()
# if dataset_option == "S&P500":
data = get_fred_data()
data_train = data.loc[data.index < test_start]
data_test = data.loc[data.index >= test_start]
model = get_model(model_option_parsed, data_train)
fit = model.fit()

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

# sns.lineplot(data=df_plot, ax=ax_eval)
st.pyplot(fig)

# st.subheader("Sunpsots")
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
# st.bar_chart(hist_values)

# # Some number in the range 0-23
# hour_to_filter = st.slider("hour", 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

# st.subheader("Map of all pickups at %s:00" % hour_to_filter)
# st.map(filtered_data)

