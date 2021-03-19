import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from data import get_australian_tourist_data


st.title("ETS Model Equation")

dataset_option = st.sidebar.selectbox("dataset", ("australian_tourists", "S&P500"))
test_start = st.sidebar.selectbox("train_test_cutoff", ("2010", "2012", "2014"))
seasonality_option = st.sidebar.selectbox("seasonality", ("None", "add", "mul"))
trend_option = st.sidebar.selectbox("trend", ("None", "add"))
if trend_option == "add":
    damped_trend = st.sidebar.selectbox("damped_trend", ("False", "True"))
else:
    damped_trend = False
error_option = st.sidebar.selectbox("error", ("add", "mul"))
damped_trend_option = True if damped_trend == "True" else False

if seasonality_option == "None":
    seasonality_option = None
if trend_option == "None":
    trend_option = None


def get_equation(trend_opt, seasonality_opt, damped_opt, error_opt) -> str:
    add_block = "l_{t-1}"
    mult_block = ""
    error_block = r"+ \epsilon_t"
    if error_option == "mul":
        error_block = r"*(1+\epsilon_t)"
    if trend_opt == "add":
        add_block += "+ b_{t-1}"

    if seasonality_opt == "add":
        add_block += "+ s_{t-m}"
    if seasonality_opt == "mul":
        mult_block = "* s_{t-m}"
    eqn = f"y_t = ({add_block}) {mult_block} {error_block}"
    return eqn


st.latex(
    get_equation(
        trend_opt=trend_option,
        seasonality_opt=seasonality_option,
        damped_opt=damped_trend,
        error_opt=error_option,
    )
)

if dataset_option == "australian_tourists":
    data, data_plot = get_australian_tourist_data()
model = ETSModel(
    data.loc[data.index < test_start],
    error=error_option,
    trend=trend_option,
    seasonal=seasonality_option,
    damped_trend=damped_trend_option,
    seasonal_periods=4,
)
fit = model.fit()

pred = fit.get_prediction(start=0, end=len(data) - 1).summary_frame(alpha=0.05)

mse_in_sample = mean_squared_error(
    data.loc[data.index < test_start], pred.loc[pred.index < test_start]["mean"]
)
mse_out_of_sample = mean_squared_error(
    data.loc[data.index >= test_start], pred.loc[pred.index >= test_start]["mean"]
)

df_plot = pd.concat([data.rename({"0": "actuals"}), pred], axis=1)

fig, ax_eval = plt.subplots()

# df_plot.loc[df_plot.index < test_start]["actuals"].plot(
#     label="actuals", legend=True, ax=ax_eval, alpha=0.5
# )
df_plot["actuals"].plot(label="actuals", legend=True, ax=ax_eval, alpha=0.5)
df_plot.loc[df_plot.index < test_start]["mean"].plot(
    label="mean_in_sample", legend=True, ax=ax_eval
)
df_plot.loc[df_plot.index >= test_start]["mean"].plot(
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

