from typing import List, Tuple, Dict, Union
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel, ETSResults
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from collections import namedtuple


def get_model(
    model_type: str, model_option_parsed: Tuple[str], data_train: pd.DataFrame
):
    if model_type == "ETS":
        model = ETSContainer(model_option_parsed, data_train)
    elif model_type == "ARIMA":
        model = ARIMAContainer(model_option_parsed, data_train)
    return model


ETSHyperparams = namedtuple(
    "ETSHyperparams", ["trend", "seasonality", "n_seasonal_periods", "error"]
)
ARIMAHyperparams = namedtuple("ARIMAHyperparams", ["p", "d", "q"])


class ETSContainer:
    hyperparams: ETSHyperparams

    def __init__(self, model_options_raw: Tuple[str], data_train: pd.Series):

        self.hyperparams = self._parse_hyperparams(model_options_raw)
        self.equation = self._get_equation()
        self.model = self._init_model(data_train)

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

    def _init_model(self, data_train) -> ETSModel:
        model = ETSModel(
            data_train,
            error=self.hyperparams.error,
            trend=self.hyperparams.trend,
            seasonal=self.hyperparams.seasonality,
            seasonal_periods=self.hyperparams.n_seasonal_periods,
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

    def fit(self) -> ETSResults:
        return self.model.fit()


class ARIMAContainer:
    def __init__(self, model_options_raw: Tuple[str], data_train: pd.Series):
        self.hyperparams = self._parse_hyperparams(model_options_raw)
        self.model = self._init_model(data_train)
        self.equation = self._get_equation()

    def _parse_hyperparams(self, model_options_raw: Tuple[str]) -> ARIMAHyperparams:
        return ARIMAHyperparams(
            p=int(model_options_raw[0]),
            d=int(model_options_raw[1]),
            q=int(model_options_raw[2]),
        )

    def _init_model(self, data_train) -> ARIMA:
        model = ARIMA(
            data_train,
            order=(self.hyperparams.p, self.hyperparams.d, self.hyperparams.q),
        )
        return model

    def fit(self) -> ARIMAResults:
        return self.model.fit()

    def _get_equation(self) -> str:
        return "TODO: add equation"
