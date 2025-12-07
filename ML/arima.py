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


class Arima:
    def __init__(self):
        pass

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
            SELECT timestamp, price, listings
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

        df = df.asfreq('D')
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df['listings'] = pd.to_numeric(df['listings'], errors='coerce')

        df = df.interpolate()

        df['listings'] = df['listings'].astype(int)

        return df

    def forecast_arima(self, df, steps=30):
        model = ARIMA(df, order=(1, 1, 2))
        model_fit = model.fit()
        
        pred = model_fit.get_forecast(steps=steps)
        forecast_values = pred.predicted_mean
        conf_int = pred.conf_int()

        result = pd.DataFrame({
            "forecast": forecast_values,
            "lower_ci": conf_int.iloc[:, 0],
            "upper_ci": conf_int.iloc[:, 1],
        })

        return result
    
    def forecast_sarimax(self, df, steps=30):
        y = df['price'].copy()
        y = y.asfreq('D')
        y = y.ffill()

        model = SARIMAX(
            y,
            order=(3, 1, 1), # (1, 1, 2)
            seasonal_order=(1, 0, 0, 30),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        model_fit = model.fit(disp=False)

        pred = model_fit.get_forecast(steps=steps)
        forecast_values = pred.predicted_mean
        conf_int = pred.conf_int()

        result = pd.DataFrame({
            "forecast": forecast_values,
            "lower_ci": conf_int.iloc[:, 0],
            "upper_ci": conf_int.iloc[:, 1],
        })

        return result
    
    def forecast_auto(self, df, steps=30, seasonal=True):
        y = df['price'].copy()
        y = y.asfreq('D')
        y = y.ffill()

        model = auto_arima(
            y,
            seasonal=seasonal,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=True
        )

        forecast_values, conf_int = model.predict(n_periods=steps, return_conf_int=True)

        forecast_index = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')

        result = pd.DataFrame({
            "forecast": forecast_values,
            "lower_ci": conf_int[:, 0],
            "upper_ci": conf_int[:, 1],
        }, index=forecast_index)

        return result
    
    def forecast_with_exog(
            self, df, steps=30,
            order=(3,1,1), seasonal_order=(1,0,0,30),
            exog_columns=['listings'], player_json_path="scrapers/cs_online.json"
        ):
            # Нормализуем временные метки до даты
            df_norm = df.copy()
            df_norm.index = df_norm.index.normalize()
            exog = df_norm[exog_columns].copy()

            # Загружаем данные по онлайн игрокам
            with open(player_json_path, "r", encoding="utf-8") as f:
                player_data = json.load(f)
            player_df = pd.DataFrame(player_data)
            player_df["timestamp"] = pd.to_datetime(player_df["timestamp"]).dt.normalize()
            player_df = player_df.set_index("timestamp")

            # Объединяем exog с playerCount
            exog = exog.join(player_df, how="left")

            # Заполняем пропуски и приводим к целым числам
            exog = exog.ffill().bfill()
            exog['playerCount'] = exog['playerCount'].astype(int)

            y = df_norm['price'].copy()

            # Создаем SARIMAX модель с экзогенными переменными
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                exog=exog,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)

            # Берем последнюю строку exog для будущих значений
            last_exog = exog.iloc[-1:]
            future_index = pd.date_range(
                start=y.index[-1] + pd.Timedelta(days=1),
                periods=steps,
                freq='D'
            )
            exog_future = pd.DataFrame(
                np.tile(last_exog.values, (steps, 1)),
                index=future_index,
                columns=exog.columns
            )

            # Делаем прогноз
            pred = model_fit.get_forecast(steps=steps, exog=exog_future)
            forecast_values = pred.predicted_mean
            conf_int = pred.conf_int()

            result = pd.DataFrame({
                "forecast": forecast_values,
                "lower_ci": conf_int.iloc[:, 0],
                "upper_ci": conf_int.iloc[:, 1],
            }, index=future_index)

            return result
    
    def plot_forecast(self, item_id, df, forecast, min_date=None, save = True):
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
            min_date = pd.to_datetime(min_date).tz_localize(df.index.tz) if df.index.tz else pd.to_datetime(min_date)
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
            label="Confidence Interval"
        )

        plt.title(f"Price Forecast — {hash_name}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)

        if save:
            file_path = os.path.join(out_dir, f"{re.sub(r'[<>:"/\\|?*]', '_', hash_name)}_forecast.png")
            plt.savefig(file_path, dpi=200, bbox_inches='tight')
            print(f"Saved forecast plot to: {file_path}")

        plt.close()

    def verify_model(self, item_ids, steps=30, exog_columns=['listings'], min_date_to_plot = None):
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

            train = df.iloc[:-steps].copy()
            test = df.iloc[-steps:].copy()

            forecast = self.forecast_with_exog(
                train, 
                steps=steps, 
                exog_columns=exog_columns
            )
            # forecast = self.forecast_sarimax(train, steps=30)

            test.index = test.index - pd.Timedelta(hours=3)

            forecast = forecast.reindex(test.index)

            y_true = test['price']
            y_pred = forecast['forecast']

            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            results.append({
                "item_id": item_id,
                "item_name": hash_name,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE (%)": mape
            })

            print(f"Verified {item_id} — MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%")

            if min_date_to_plot:
                min_date_to_plot = pd.to_datetime(min_date_to_plot)
                if train.index.tz is not None:
                    if min_date_to_plot.tz is None:
                        min_date_to_plot = min_date_to_plot.tz_localize(train.index.tz)
                    else:
                        min_date_to_plot = min_date_to_plot.tz_convert(train.index.tz)
                train = train[train.index >= min_date_to_plot]

            plt.figure(figsize=(10,5))
            plt.plot(train.index - pd.Timedelta(hours=3), train['price'], label='Train', color='blue')
            plt.plot(test.index, y_true, label='Test (real)', color='green')
            plt.plot(forecast.index, y_pred, label='Forecast', color='red')
            plt.fill_between(forecast.index, forecast['lower_ci'], forecast['upper_ci'], color='red', alpha=0.2, label='Confidence Interval')
            plt.title(f"Forecast vs Real — {hash_name}")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)

            filename = f"{save_dir}/{re.sub(r'[<>:"/\\|?*]', '_', hash_name)}.png"
            plt.savefig(filename, bbox_inches='tight')

        return pd.DataFrame(results)


arima = Arima()

# item_id = 19356
# df = arima.price_history_by_id(item_id)
# print(df)
# forecast = arima.forecast_auto(df, steps=7, seasonal=True)
# forecast = arima.forecast_sarimax(df, steps=30)
# forecast = arima.forecast_with_exog(df, steps=30)
# print(forecast)
# arima.plot_forecast(item_id, df, forecast, min_date="2025-08-18")

ids_to_verify = [25355, 25395, 3153, 26954, 19010, 6317, 25201, 11773, 19356]
results = arima.verify_model(ids_to_verify, steps=30, min_date_to_plot='2024-01-01')

results.to_csv("forecasts_arima/metrics.csv", index=False, float_format='%.2f')
