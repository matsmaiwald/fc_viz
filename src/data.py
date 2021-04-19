import pandas as pd
from numpy import NaN
from typing import List
import sktime.datasets.base as skdata


def get_australian_tourist_data():
    data = pd.read_csv("/src/data_raw/australian_tourists.csv", index_col=False)

    t = pd.date_range("1999-03-01", "2015-12-01", freq="3MS")
    data = pd.Series(data.australian_tourists.values, index=t)
    data.index = pd.PeriodIndex(data.index, freq="3M")

    # data.index = pd.PeriodIndex(data.index, freq="3MS")
    data.name = "actuals"
    return data
    # return data, data.index[-5:-2]


def get_fred_data() -> pd.DataFrame:
    data = pd.read_csv("/src/data_raw/SP500.csv", na_values=".", parse_dates=True)
    data["DATE"] = pd.to_datetime(data["DATE"])
    data = data.set_index("DATE")["SP500"]
    data = data.reindex(
        pd.bdate_range(min(data.index), max(data.index)), fill_value=NaN
    )

    return data.interpolate()


def get_airline_data():
    return skdata.load_airline()


def get_lynx_population():
    return skdata.load_lynx()


dataset_name_mapping = {
    "S&P500": get_fred_data(),
    "australian_tourists": get_australian_tourist_data(),
    "airline_passengers": get_airline_data(),
    "shampoo_sales": skdata.load_shampoo_sales(),
    "lynx_population": get_lynx_population(),
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
        ), f"Could not locate dataset: {raw_name}"
        self.data = dataset_name_mapping[raw_name]

        split_options = self._get_train_test_split_options(self.data)
        self.split_options = list(map(lambda x: x.strftime("%Y-%m-%d"), split_options))


if __name__ == "__main__":
    print(DataSet("australian_tourists").split_options)
    print(DataSet("airline_passengers").split_options)
    print(DataSet("australian_tourists").data)
    import pdb

    pdb.set_trace()
    print(DataSet("airline_passengers").data)
