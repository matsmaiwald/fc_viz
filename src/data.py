import pandas as pd
from numpy import NaN
from typing import List, Tuple
import sktime.datasets.base as skdata


def get_australian_tourist_data():
    austourists_data = pd.read_csv("/src/data_raw/australian_tourists.csv")
    t = pd.date_range("1999-03-01", "2015-12-01", freq="3MS")
    data = pd.Series(austourists_data, index=t)
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
