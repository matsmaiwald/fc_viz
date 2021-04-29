from typing import List, Tuple, Dict, Union
import pandas as pd
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from collections import namedtuple

# from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.naive import NaiveForecaster
from data import DataSet
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.tbats import TBATS


def get_model(model_type: str, model_option_parsed: Tuple[str]):
    if model_type == "ETS":
        model = ETSContainer(model_option_parsed)
    elif model_type == "ARIMA":
        model = ARIMAContainer(model_option_parsed)
    # elif model_type == "Prophet":
    #     model = ProphetContainer(model_option_parsed)
    elif model_type == "Naive":
        model = NaiveContainer()
    elif model_type == "AutoArima":
        model = AutoArimaContainer()
    elif model_type == "TBATS":
        model = TBATSContainer(model_option_parsed)

    return model


ETSHyperparams = namedtuple(
    "ETSHyperparams", ["trend", "seasonality", "n_seasonal_periods", "error"]
)
ARIMAHyperparams = namedtuple("ARIMAHyperparams", ["p", "d", "q"])


class BaseModel:
    def __init__(self):
        self.model = self._init_model()
        self.equation = self._get_equation()

    @staticmethod
    def prep_data(dataset):
        return dataset.get_data()

    def fit(self, data_train):
        return self.model.fit(data_train)

    def _get_equation(self) -> str:
        return "TODO: add equation"


class NaiveContainer(BaseModel):
    def _init_model(self) -> NaiveForecaster:
        model = NaiveForecaster()
        return model

    # def fit(self, data_train):
    #     return self.model.fit(data_train)

    # @staticmethod
    # def prep_data(dataset):
    #     return dataset.get_data()


class ETSContainer(BaseModel):
    hyperparams: ETSHyperparams

    def __init__(self, model_options_raw: Tuple[str]):

        self.hyperparams = self._parse_hyperparams(model_options_raw)
        self.equation = self._get_equation()
        self.model = self._init_model()

    def _parse_hyperparams(self, model_options_raw: Tuple[str]) -> ETSHyperparams:
        trend_option = "mul" if "multiplicative trend" in model_options_raw else "add"
        season_option = (
            "mul" if "multiplicative seasonality" in model_options_raw else "add"
        )
        n_seasons = 4 if "4 seasons" in model_options_raw else 5
        return ETSHyperparams(
            trend=trend_option,
            seasonality=season_option,
            n_seasonal_periods=n_seasons,
            error="add",
        )

    def _init_model(self) -> ExponentialSmoothing:
        model = ExponentialSmoothing(
            # data_train,
            # error=self.hyperparams.error,
            trend=self.hyperparams.trend,
            seasonal=self.hyperparams.seasonality,
            sp=self.hyperparams.n_seasonal_periods,
        )
        return model

    def _get_equation(self) -> str:
        add_block = "l_{t-1}"
        mult_block = ""
        error_block = r"+ \epsilon_t"
        if self.hyperparams.error == "mul":
            error_block = r"*(1+\epsilon_t)"
        if self.hyperparams.trend == "add":
            add_block += "+ b_{t-1}"

        if self.hyperparams.seasonality == "add":
            add_block += "+ s_{t-m}"
        if self.hyperparams.seasonality == "mul":
            mult_block = "* s_{t-m}"
        eqn = f"y_t = ({add_block}) {mult_block} {error_block}"
        return eqn


class AutoArimaContainer(BaseModel):
    def _init_model(self) -> AutoARIMA:
        model = AutoARIMA(seasonal=True)
        return model


class TBATSContainer(BaseModel):
    def __init__(self, seasonality_periods: List[float]):
        self.hyperparams = {"sp": [seasonality_periods]}
        self.model = self._init_model()
        self.equation = self._get_equation()

    def _init_model(self) -> TBATS:
        print(self.hyperparams["sp"])
        model = TBATS(sp=self.hyperparams["sp"])
        return model


class ARIMAContainer(BaseModel):
    def __init__(self, model_options_raw: Tuple[str]):
        self.hyperparams = self._parse_hyperparams(model_options_raw)
        self.model = self._init_model()
        self.equation = self._get_equation()

    def _parse_hyperparams(self, model_options_raw: Tuple[str]) -> ARIMAHyperparams:
        return ARIMAHyperparams(
            p=int(model_options_raw[0]),
            d=int(model_options_raw[1]),
            q=int(model_options_raw[2]),
        )

    def _init_model(self) -> ARIMA:
        model = ARIMA(
            order=(self.hyperparams.p, self.hyperparams.d, self.hyperparams.q),
        )
        return model

    # def fit(self) -> ARIMAResults:
    #     return self.model.fit()

    def _get_equation(self) -> str:
        return "TODO: add equation"


# class ProphetContainer(BaseModel):
#     def __init__(self, model_options_raw: Tuple[str]):
#         # self.hyperparams = self._parse_hyperparams(model_options_raw)
#         self.model = self._init_model()
#         self.equation = self._get_equation()

#     def _parse_hyperparams(self, model_options_raw: Tuple[str]) -> ARIMAHyperparams:
#         pass

#     def _init_model(self) -> Prophet:
#         model = Prophet()
#         return model

#     @staticmethod
#     def prep_data(dataset: DataSet):
#         data_prep = dataset.get_data_as_DateTimeIndex()
#         # try:
#         #     data_prep = data_prep.as_DateTimeIndex
#         # except AttributeError as e:
#         #     pass
#         # data_prep = data_prep.reset_index().rename(columns={"index": "ds"})
#         return data_prep

# def fit(self, data_train) -> ARIMAResults:
#     # data_prep = self._prep_data(self.data_train)
#     model_trained = self.model.fit(data_train)
#     return model_trained

