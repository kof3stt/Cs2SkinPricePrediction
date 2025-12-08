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
from prophet import Prophet


class ProphetModel:
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

        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.sort_values("timestamp")
        df = df.rename(columns={"timestamp": "ds", "price": "y"})

        df = df[["ds", "y"]]

        df = df.set_index("ds")

        df = df.asfreq("D")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.interpolate()

        df = df.reset_index()

        return df

    def forecast_prophet_with_online(self, df, steps=30):
        model = Prophet(daily_seasonality=True)
        model.add_regressor("playerCount")

        model.fit(df)

        future = model.make_future_dataframe(periods=steps)

        last_online = df["playerCount"].iloc[-1]
        df_future = df[["playerCount"]].copy()
        future_exog = pd.DataFrame({"playerCount": [last_online] * steps})
        full_exog = pd.concat([df_future, future_exog], ignore_index=True)

        future["playerCount"] = full_exog["playerCount"].values

        forecast = model.predict(future)
        return forecast

    def price_history_with_online(
        self,
        item_id,
        provider="buff163",
        start_date=None,
        end_date=None,
        limit_last_n=None,
    ):
        df_price = self.price_history_by_id(
            item_id, provider, start_date, end_date, limit_last_n
        )
        df_online = self.load_player_online()

        df_price["ds"] = pd.to_datetime(df_price["ds"]).dt.normalize()
        df_online["ds"] = pd.to_datetime(df_online["ds"]).dt.tz_localize(None)

        df = df_price.merge(df_online, on="ds", how="left")
        df["playerCount"] = df["playerCount"].ffill().bfill()

        return df

    def load_player_online(self, json_path="scrapers/cs_online.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        df = df.rename(columns={"timestamp": "ds", "playerCount": "playerCount"})
        return df.set_index("ds").asfreq("D").ffill().reset_index()

    def forecast_prophet(self, df, steps=30):
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=steps, freq="D")
        forecast = model.predict(future)

        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def get_item_name_by_id(self, item_id):
        session = create_session()

        sql = "SELECT hash_name FROM items WHERE item_id = :item_id LIMIT 1"
        row = session.execute(text(sql), {"item_id": item_id}).fetchone()

        return row[0] if row and row[0] else f"item_{item_id}"

    def plot_forecast(self, df, forecast, hash_name, steps, min_date=None, save=True):
        if min_date:
            min_date = pd.to_datetime(min_date)
            df_plot = df[df["ds"] >= min_date]
            forecast_plot = forecast[forecast["ds"] >= min_date]
        else:
            df_plot = df
            forecast_plot = forecast

        plt.figure(figsize=(12, 6))

        plt.plot(df_plot["ds"], df_plot["y"], label="Historical Price", alpha=0.6)

        plt.plot(forecast_plot["ds"], forecast_plot["yhat"], label="Forecast")

        plt.fill_between(
            forecast_plot["ds"],
            forecast_plot["yhat_lower"],
            forecast_plot["yhat_upper"],
            alpha=0.2,
            label="Confidence Interval",
        )

        plt.title(f"Price Forecast — {hash_name}, {steps} days")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save:
            save_dir = "forecasts_prophet"
            os.makedirs(save_dir, exist_ok=True)

            filename = f"{save_dir}/{re.sub(r'[<>:\"/\\\\|?*]', '_', hash_name)}.png"
            plt.savefig(filename, dpi=200, bbox_inches="tight")
            print(f"Saved forecast plot to: {filename}")

        plt.show()

    def verify_model(self, item_ids, steps=30, min_date_to_plot=None):
        results = []

        save_dir = "verification_prophet"
        os.makedirs(save_dir, exist_ok=True)

        df_online = self.load_player_online()
        df_online["ds"] = pd.to_datetime(df_online["ds"]).dt.tz_localize(None)
        df_online = df_online.set_index("ds").asfreq("D").ffill().reset_index()

        online_model = Prophet(daily_seasonality=True)
        online_model.fit(
            df_online[["ds", "playerCount"]].rename(columns={"playerCount": "y"})
        )
        future_online = online_model.make_future_dataframe(periods=steps)
        forecast_online = online_model.predict(future_online)
        playerCount_forecast = forecast_online[["ds", "yhat"]].set_index("ds")

        for item_id in item_ids:
            hash_name = self.get_item_name_by_id(item_id)
            # df = self.price_history_by_id(item_id, end_date="2025-09-01").copy()
            df = self.price_history_with_online(item_id, end_date="2025-09-01").copy()

            if len(df) <= steps * 2:
                print(f"Skipping: {hash_name}")
                continue

            df["weekday"] = df["ds"].dt.weekday

            train_df = df.iloc[:-steps].copy()
            test_df = df.iloc[-steps:].copy()

            model = Prophet(daily_seasonality=True)

            model.add_regressor("weekday")
            model.add_regressor("playerCount")
            model.fit(train_df)

            future = model.make_future_dataframe(periods=steps)

            future = future.set_index("ds")
            future["playerCount"] = (
                playerCount_forecast["yhat"].reindex(future.index).ffill()
            )
            future["weekday"] = future.index.weekday
            future = future.reset_index()

            forecast = model.predict(future)

            y_pred = forecast.iloc[-steps:]["yhat"].values
            y_true = test_df["y"].values

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

            plt.figure(figsize=(12, 5))

            if min_date_to_plot:
                train_plot = train_df[train_df["ds"] >= min_date_to_plot]
            else:
                train_plot = train_df

            plt.plot(train_plot["ds"], train_plot["y"], label="Train", linewidth=1)

            plt.plot(test_df["ds"], test_df["y"], label="Test", linewidth=2)

            plt.plot(test_df["ds"], y_pred, label="Forecast", linewidth=2)

            plt.title(
                f"Forecast vs Real — {re.sub(r'([$#%&{}_])', r'\\\1', hash_name)}"
            )
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            if mape < 5:
                plt.savefig(
                    f"{save_dir}/item{item_id}_{re.sub(r'[<>:\"/\\\\|?*]', '_', hash_name)}.png"
                )
            plt.close()

            print(
                f"Verified {item_id}, {re.sub(r'([$#%&{}_])', r'\\\1', hash_name)} — MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%"
            )

        return pd.DataFrame(results)


# prophet = ProphetModel()
# ID = 3153
# STEPS = 120
#
# hash_name = prophet.get_item_name_by_id(ID)
#
# df = prophet.price_history_with_online(ID)
# forecast = prophet.forecast_prophet_with_online(df, steps=STEPS)
# prophet.plot_forecast(df, forecast, hash_name, steps = STEPS)

# forecast = prophet.forecast_prophet(df, steps=STEPS)
# prophet.plot_forecast(df, forecast, hash_name, steps = STEPS)

# verif
# ids_to_verify = [25355, 25395, 3153, 26954, 19010, 6317, 25201, 11773, 19356]
# ids_to_verify = range(9184, 30000)
# results = prophet.verify_model(ids_to_verify, steps=30, min_date_to_plot="2024-01-01")
# results.to_csv("verification_prophet/metrics.csv", index=False, float_format="%.2f")
