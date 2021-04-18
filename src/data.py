import pandas as pd
from numpy import NaN
from typing import List, Tuple
import sktime.datasets.base as skdata

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


def get_australian_tourist_data():
    t = pd.date_range("1999-03-01", "2015-12-01", freq="3MS")
    data = pd.Series(austourists_data, index=t)
    data.name = "actuals"
    return data
    # return data, data.index[-5:-2]


def get_fred_data() -> pd.DataFrame:
    data = pd.read_csv("/src/SP500.csv", na_values=".", parse_dates=True)
    data["DATE"] = pd.to_datetime(data["DATE"])
    data = data.set_index("DATE")["SP500"]
    data = data.reindex(
        pd.bdate_range(min(data.index), max(data.index)), fill_value=NaN
    )
    test_start_options = (
        "2020-01-01",
        "2020-03-01",
        "2020-06-01",
        "2020-09-01",
        "2021-01-01",
    )

    return data.interpolate()


dataset_name_mapping = {
    "S&P500": get_fred_data(),
    "australian_tourists": get_australian_tourist_data(),
    "airline_passengers": skdata.load_airline(),
    "shampoo_sales": skdata.load_shampoo_sales(),
    "lynx_population": skdata.load_lynx(),
}


class DataSet:
    data: pd.Series
    split_options: List[str]

    @staticmethod
    def _get_train_test_split_options(data: pd.DataFrame):
        split_options_ix = list(
            map(lambda x: int(x * data.index.size), [0.5, 0.6, 0.7, 0.8, 0.9])
        )
        split_options = data.index[split_options_ix]
        return split_options

    def __init__(self, raw_name: str):
        assert (
            raw_name in dataset_name_mapping.keys()
        ), f"Could not locate dataset: {dataset_name_mapping}"
        self.data = dataset_name_mapping[raw_name]

        split_options = self._get_train_test_split_options(self.data)
        self.split_options = list(map(lambda x: x.strftime("%Y-%m-%d"), split_options))


if __name__ == "__main__":
    print(DataSet("australian_tourists").split_options)
    print(DataSet("airline").split_options)
