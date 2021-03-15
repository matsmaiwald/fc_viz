import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import seaborn as sns
import matplotlib.pyplot as plt


st.title("Forecasting Australian Tourist Data")


austourists_data = [
    30.05251300,
    19.14849600,
    25.31769200,
    27.59143700,
    32.07645600,
    23.48796100,
    28.47594000,
    35.12375300,
    36.83848500,
    25.00701700,
    30.72223000,
    28.69375900,
    36.64098600,
    23.82460900,
    29.31168300,
    31.77030900,
    35.17787700,
    19.77524400,
    29.60175000,
    34.53884200,
    41.27359900,
    26.65586200,
    28.27985900,
    35.19115300,
    42.20566386,
    24.64917133,
    32.66733514,
    37.25735401,
    45.24246027,
    29.35048127,
    36.34420728,
    41.78208136,
    49.27659843,
    31.27540139,
    37.85062549,
    38.83704413,
    51.23690034,
    31.83855162,
    41.32342126,
    42.79900337,
    55.70835836,
    33.40714492,
    42.31663797,
    45.15712257,
    59.57607996,
    34.83733016,
    44.84168072,
    46.97124960,
    60.01903094,
    38.37117851,
    46.97586413,
    50.73379646,
    61.64687319,
    39.29956937,
    52.67120908,
    54.33231689,
    66.83435838,
    40.87118847,
    51.82853579,
    57.49190993,
    65.25146985,
    43.06120822,
    54.76075713,
    59.83447494,
    73.25702747,
    47.69662373,
    61.09776802,
    66.05576122,
]
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
seasonality_option = st.selectbox("seasonality", ("add", "mul"))
trend_option = st.selectbox("trend", ("add", "mul"))

model = ETSModel(
    data.loc[data.index < test_start],
    error="add",
    trend=trend_option,
    seasonal=seasonality_option,
    damped_trend=True,
    seasonal_periods=4,
)
fit = model.fit()

pred = fit.get_prediction(start=0, end=len(data) - 1).summary_frame(alpha=0.05)

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

