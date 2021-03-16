import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from data import austourists_data

st.title("Forecasting Australian Tourist Data")


t = pd.date_range("1999-03-01", "2015-12-01", freq="3MS")
data_plot = pd.DataFrame({"# Tourists": austourists_data, "x": t})
data = pd.Series(austourists_data, index=t)
data.name = "actuals"


print(data)


fig_data, ax_data = plt.subplots()
sns.lineplot(data=data_plot, x="x", y="# Tourists", ax=ax_data)
st.pyplot(fig_data)

st.title("ETS Model Parameters")

test_start = st.selectbox("train_test_cutoff", ("2010", "2012", "2014"))
seasonality_option = st.selectbox("seasonality", ("None", "add", "mul"))
trend_option = st.selectbox("trend", ("None", "add", "mul"))
damped_trend = st.selectbox("damped_trend", ("True", "False"))
error_option = st.selectbox("error", ("add", "mul"))
damped_trend_option = True if damped_trend == "True" else False

if seasonality_option == "None":
    seasonality_option = None
if trend_option == "None":
    trend_option = None

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

st.text(f"In-sample RMSE: {mse_in_sample}")
st.text(f"Out-of-sample RMSE: {mse_out_of_sample}")

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

