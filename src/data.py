import pandas as pd
from numpy import NaN
from typing import List
import sktime.datasets.base as skdata


def get_fred_data() -> pd.DataFrame:
    data = pd.read_csv("/src/data_raw/SP500.csv", na_values=".", parse_dates=True)
    data["DATE"] = pd.to_datetime(data["DATE"])
    data = data.set_index("DATE")["SP500"]
    data = data.reindex(
        pd.bdate_range(min(data.index), max(data.index)), fill_value=NaN
    )
    data = data.interpolate()
    data.index = pd.PeriodIndex(data.index, freq="B")

    return data


def get_airline_data():
    return skdata.load_airline()


def get_lynx_population():
    return skdata.load_lynx()


dataset_name_mapping = {
    "S&P500": get_fred_data(),
    "airline_passengers": get_airline_data(),
    "shampoo_sales": skdata.load_shampoo_sales(),
    "lynx_population": get_lynx_population(),
}


class DataSet:
    data: pd.Series
    split_options: List[str]
    freq: str

    @staticmethod
    def _get_train_test_split_options(data: pd.DataFrame):
        split_options_ix = list(
            map(lambda x: int(x * data.index.size), [0.5, 0.6, 0.7, 0.8, 0.9])
        )
        split_options = data.index[split_options_ix]
        return split_options

    def __init__(self, data: pd.Series):
        self.data = data
        self.freq = data.index.freq.name
        split_options = self._get_train_test_split_options(self.data)
        self.split_options = list(map(lambda x: x.strftime("%Y-%m-%d"), split_options))

    def get_data_as_DateTimeIndex(self):
        return self.data.to_timestamp(freq=self.freq)

    def get_data(self):
        return self.data


def get_dataset(name):
    assert name in dataset_name_mapping.keys(), f"Could not locate dataset: {name}"
    data = dataset_name_mapping[name]
    return DataSet(data)


if __name__ == "__main__":
    data = get_fred_data()
    import pdb

    pdb.set_trace()
