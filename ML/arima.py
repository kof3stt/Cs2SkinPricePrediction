from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import sys
import os
import json
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import create_session
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from prophet import Prophet


class Arima:
    def __init__(self):
        self._online_model = None
        self._online_forecast_cache = {}

    def price_history_by_id(
        self,
        item_id,
        provider="buff163",
        start_date=None,
        end_date=None,
        limit_last_n=None,
    ):
        session = create_session()

        sql = """
            SELECT timestamp, price
            FROM price_time_series
            WHERE item_id = :item_id
              AND provider = :provider
        """

        params = {
            "item_id": item_id,
            "provider": provider,
        }

        if start_date:
            sql += " AND timestamp >= :start_date"
            params["start_date"] = start_date

        if end_date:
            sql += " AND timestamp <= :end_date"
            params["end_date"] = end_date

        sql += " ORDER BY timestamp DESC"

        if limit_last_n:
            sql += f" LIMIT {limit_last_n}"

        result = session.execute(text(sql), params)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df = df.set_index("timestamp")

        df["price"] = pd.to_numeric(df["price"], errors="coerce")

        df = df.resample("D").ffill()

        return df

    def forecast_arima(self, df, steps=30):
        model = ARIMA(df, order=(1, 1, 2))
        model_fit = model.fit()

        pred = model_fit.get_forecast(steps=steps)
        forecast_values = pred.predicted_mean
        conf_int = pred.conf_int()

        result = pd.DataFrame(
            {
                "forecast": forecast_values,
                "lower_ci": conf_int.iloc[:, 0],
                "upper_ci": conf_int.iloc[:, 1],
            }
        )

        return result

    def forecast_sarimax(self, df, steps=30):
        y = df["price"].copy()
        y = y.asfreq("D")
        y = y.ffill()

        model = SARIMAX(
            y,
            order=(3, 1, 1),  # (1, 1, 2)
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        model_fit = model.fit(disp=False)

        pred = model_fit.get_forecast(steps=steps)
        forecast_values = pred.predicted_mean
        conf_int = pred.conf_int()

        result = pd.DataFrame(
            {
                "forecast": forecast_values,
                "lower_ci": conf_int.iloc[:, 0],
                "upper_ci": conf_int.iloc[:, 1],
            }
        )

        return result

    def forecast_auto(self, df, steps=30, seasonal=True):
        y = df["price"].copy()
        y = y.asfreq("D")
        y = y.ffill()

        model = auto_arima(
            y,
            seasonal=seasonal,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=True,
        )

        forecast_values, conf_int = model.predict(n_periods=steps, return_conf_int=True)

        forecast_index = pd.date_range(
            start=y.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D"
        )

        result = pd.DataFrame(
            {
                "forecast": forecast_values,
                "lower_ci": conf_int[:, 0],
                "upper_ci": conf_int[:, 1],
            },
            index=forecast_index,
        )

        return result

    def forecast_with_exog(
        self,
        df,
        steps=30,
        order=(3, 1, 1),
        seasonal_order=(1, 0, 0, 30),
        exog_columns=[],
        player_json_path="scrapers/cs_online.json",
    ):
        """
        Прогнозирование с использованием модели SARIMAX с экзогенными переменными.
        Включает улучшенный прогноз онлайн-игроков с помощью модели Prophet.
        """
        # Нормализуем временные метки до даты
        df_norm = df.copy()
        df_norm.index = df_norm.index.normalize()
        
        # Создаем DataFrame с экзогенными переменными (без playerCount)
        exog = df_norm[exog_columns].copy() if exog_columns else pd.DataFrame(index=df_norm.index)
        
        # Загружаем и подготавливаем исторические данные по онлайн-игрокам
        with open(player_json_path, "r", encoding="utf-8") as f:
            player_data = json.load(f)
        player_df = pd.DataFrame(player_data)
        player_df["timestamp"] = pd.to_datetime(player_df["timestamp"]).dt.normalize()
        player_df = player_df.set_index("timestamp")
        
        # Объединяем экзогенные переменные с историческими данными об онлайн-игроках
        exog = exog.join(player_df, how="left")
        
        # Заполняем пропуски и приводим типы данных
        exog = exog.ffill().bfill()
        if "playerCount" in exog.columns:
            exog["playerCount"] = exog["playerCount"].astype(int)
        
        # Целевая переменная - цена
        y = df_norm["price"].copy()
        
        # Убеждаемся, что индексы y и exog совпадают
        exog = exog.reindex(y.index)
        exog = exog.ffill().bfill()
        
        # Строим прогноз онлайн-игроков на период steps
        online_forecast = self.forecast_online(steps)
        
        # Создаем DataFrame для будущих экзогенных переменных
        future_index = pd.date_range(
            start=y.index[-1] + pd.Timedelta(days=1), 
            periods=steps, 
            freq="D"
        )
        
        # Создаем exog_future с прогнозируемыми значениями
        exog_future_list = []
        
        for i in range(steps):
            future_row = {}
            
            # Для каждой экзогенной переменной (кроме playerCount) используем последнее известное значение
            for col in exog_columns:
                if col in exog.columns:
                    future_row[col] = exog[col].iloc[-1]
            
            # Добавляем прогнозируемое значение онлайн-игроков
            if i < len(online_forecast):
                future_row["playerCount"] = online_forecast.iloc[i]["playerCount"]
            else:
                # Если прогноз недостаточно длинный, используем последнее прогнозируемое значение
                future_row["playerCount"] = online_forecast.iloc[-1]["playerCount"]
            
            exog_future_list.append(future_row)
        
        exog_future = pd.DataFrame(exog_future_list, index=future_index)
        
        # Убеждаемся, что порядок столбцов совпадает с exog
        if "playerCount" in exog.columns and "playerCount" not in exog_future.columns:
            exog_future["playerCount"] = online_forecast["playerCount"].values[:steps]
        
        # Сортируем столбцы в том же порядке, что и в exog
        exog_future = exog_future[exog.columns]
        
        # Создаем и обучаем модель SARIMAX с экзогенными переменными
        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            exog=exog,
            trend="t"
        )
        
        try:
            model_fit = model.fit(disp=False, maxiter=1000)
        except Exception as e:
            print(f"Ошибка при обучении модели SARIMAX: {e}")
            # Пробуем упрощенную модель в случае ошибки
            model = SARIMAX(
                y,
                order=(1, 1, 0),
                seasonal_order=(0, 0, 0, 0),
                exog=exog,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            model_fit = model.fit(disp=False)
        
        # Делаем прогноз с использованием будущих экзогенных переменных
        pred = model_fit.get_forecast(steps=steps, exog=exog_future)
        forecast_values = pred.predicted_mean
        conf_int = pred.conf_int()
        
        # Формируем результат
        result = pd.DataFrame(
            {
                "forecast": forecast_values,
                "lower_ci": conf_int.iloc[:, 0],
                "upper_ci": conf_int.iloc[:, 1],
            },
            index=future_index,
        )
        
        # Дополнительно: сохраняем использованные экзогенные переменные для анализа
        result["playerCount"] = exog_future["playerCount"].values
        
        return result

    def plot_forecast(self, item_id, df, forecast, min_date=None, save=True):
        session = create_session()

        sql = "SELECT hash_name FROM items WHERE item_id = :item_id LIMIT 1"
        row = session.execute(text(sql), {"item_id": item_id}).fetchone()

        if row and row[0]:
            hash_name = row[0]
        else:
            hash_name = f"item_{item_id}"

        out_dir = "forecasts_arima"
        os.makedirs(out_dir, exist_ok=True)

        if min_date:
            min_date = (
                pd.to_datetime(min_date).tz_localize(df.index.tz)
                if df.index.tz
                else pd.to_datetime(min_date)
            )
            df_plot = df[df.index >= min_date]
        else:
            df_plot = df

        plt.figure(figsize=(14, 6))
        plt.plot(df_plot.index, df_plot["price"], label="Historical Price", linewidth=2)
        plt.plot(forecast.index, forecast["forecast"], label="Forecast", linewidth=2)

        plt.fill_between(
            forecast.index,
            forecast["lower_ci"],
            forecast["upper_ci"],
            alpha=0.2,
            label="Confidence Interval",
        )

        plt.title(f"Price Forecast — {hash_name}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)

        if save:
            file_path = os.path.join(
                out_dir, f"{re.sub(r'[<>:"/\\|?*]', '_', hash_name)}_forecast.png"
            )
            plt.savefig(file_path, dpi=200, bbox_inches="tight")
            print(f"Saved forecast plot to: {file_path}")

        plt.close()

    def verify_model(
        self, item_ids, steps=30, exog_columns=[], min_date_to_plot=None
    ):
        """
        Верификация модели для списка предметов.
        """
        results = []

        save_dir = "verification_arima"
        os.makedirs(save_dir, exist_ok=True)

        session = create_session()

        for item_id in item_ids:
            sql = "SELECT hash_name FROM items WHERE item_id = :item_id LIMIT 1"
            row = session.execute(text(sql), {"item_id": item_id}).fetchone()
            hash_name = row[0] if row and row[0] else f"item_{item_id}"

            df = self.price_history_by_id(item_id, end_date="2025-09-01")

            if df.empty:
                continue

            train = df.iloc[:-steps].copy()
            test = df.iloc[-steps:].copy()

            forecast = self.forecast_with_exog(
                train, steps=steps, exog_columns=exog_columns
            )
            # forecast = self.forecast_sarimax(train, steps=30)

            test.index = test.index.normalize()

            forecast = forecast.reindex(test.index)

            y_true = test["price"]
            y_pred = forecast["forecast"]

            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            results.append(
                {
                    "item_id": item_id,
                    "item_name": hash_name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE (%)": mape,
                }
            )

            print(
                f"Verified {item_id}, {hash_name} — MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%"
            )

            if min_date_to_plot:
                min_date_to_plot = pd.to_datetime(min_date_to_plot)
                if train.index.tz is not None:
                    if min_date_to_plot.tz is None:
                        min_date_to_plot = min_date_to_plot.tz_localize(train.index.tz)
                    else:
                        min_date_to_plot = min_date_to_plot.tz_convert(train.index.tz)
                train = train[train.index >= min_date_to_plot]

            plt.figure(figsize=(10, 5))
            plt.plot(
                train.index - pd.Timedelta(hours=3),
                train["price"],
                label="Train",
                color="blue",
            )
            plt.plot(test.index, y_true, label="Test (real)", color="green")
            plt.plot(forecast.index, y_pred, label="Forecast", color="red")
            plt.fill_between(
                forecast.index,
                forecast["lower_ci"],
                forecast["upper_ci"],
                color="red",
                alpha=0.2,
                label="Confidence Interval",
            )
            plt.title(f"SARIMAX: Forecast vs Real — {hash_name}")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)

            filename = f"{save_dir}/item{item_id}_{re.sub(r'[<>:\"/\\\\|?*]', '_', hash_name)}.png"
            if mape < 5:
                plt.savefig(filename)
            plt.show()

        return pd.DataFrame(results)
    
    def forecast_online(self, steps=120):
        if steps in self._online_forecast_cache:
            return self._online_forecast_cache[steps]

        model = self.load_or_fit_online_model()
        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)[["ds", "yhat"]].tail(steps)
        forecast = forecast.rename(columns={"yhat": "playerCount"})

        self._online_forecast_cache[steps] = forecast
        return forecast
    
    def load_player_online(self, json_path="scrapers/cs_online.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df["ds"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
        df = df[["ds", "playerCount"]]

        # Daily freq
        df = df.set_index("ds").asfreq("D").ffill().reset_index()

        return df
    
    def load_or_fit_online_model(self):
        if self._online_model is not None:
            return self._online_model

        df_online = self.load_player_online()
        df_prophet = df_online.rename(columns={"playerCount": "y"}).copy()

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode="multiplicative",
        )
        model.add_seasonality(name="monthly", period=30.5, fourier_order=6)

        model.fit(df_prophet)
        self._online_model = model
        return model


# arima = Arima()

# item_id = 19356
# df = arima.price_history_by_id(item_id)
# print(df)
# forecast = arima.forecast_auto(df, steps=7, seasonal=True)
# forecast = arima.forecast_sarimax(df, steps=30)
# forecast = arima.forecast_with_exog(df, steps=30)
# print(forecast)
# arima.plot_forecast(item_id, df, forecast, min_date="2025-08-18")

# ids_to_verify = [25355, 25395, 3153, 26954, 19010, 6317, 25201, 11773, 19356]
# ids_to_verify = [23482]
# results = arima.verify_model(ids_to_verify, steps=30, min_date_to_plot='2024-01-01')
# results.to_csv("verification_arima/metrics.csv", index=False, float_format='%.2f')
