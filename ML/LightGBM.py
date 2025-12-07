from sqlalchemy import text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from lightgbm import LGBMRegressor
from prophet import Prophet
import sys
import os
import json
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import create_session


class LightGBMModel:
    def __init__(self):
        pass

    # ============================
    # LOAD PRICE HISTORY
    # ============================
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

        params = {"item_id": item_id, "provider": provider}

        if start_date:
            sql += " AND timestamp >= :start_date"
            params["start_date"] = start_date
        if end_date:
            sql += " AND timestamp <= :end_date"
            params["end_date"] = end_date

        sql += " ORDER BY timestamp DESC"
        if limit_last_n:
            sql += f" LIMIT {limit_last_n}"

        df = pd.DataFrame(session.execute(text(sql), params).fetchall(),
                          columns=["timestamp", "y", "listings"])

        df["ds"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df[["ds", "y"]].sort_values("ds")

        # Daily frequency, interpolate missing values
        df = df.set_index("ds").asfreq("D")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.interpolate().reset_index()

        return df

    # ============================
    # LOAD PLAYER ONLINE
    # ============================
    def load_player_online(self, json_path="scrapers/cs_online.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df["ds"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
        df = df[["ds", "playerCount"]]

        # Daily freq
        df = df.set_index("ds").asfreq("D").ffill().reset_index()

        return df

    # ============================
    # MERGE PRICE + ONLINE
    # ============================
    def price_history_with_online(self, item_id, **kwargs):
        df_price = self.price_history_by_id(item_id, **kwargs)
        df_online = self.load_player_online()

        df_price["ds"] = pd.to_datetime(df_price["ds"]).dt.normalize()
        df_online["ds"] = pd.to_datetime(df_online["ds"]).dt.tz_localize(None)

        df = df_price.merge(df_online, on="ds", how="left")

        df["playerCount"] = df["playerCount"].ffill().bfill()

        return df

    # ============================
    # FEATURE GENERATOR
    # ============================
    def make_features(self, df, lags=[1,7,14], rolling=[7,14], dropna=True):
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").reset_index(drop=True)

        df["weekday"] = df["ds"].dt.weekday

        # Lags
        for L in lags:
            df[f"lag{L}"] = df["y"].shift(L)

        # Rolling windows
        for W in rolling:
            df[f"ma{W}"] = df["y"].rolling(W).mean()
            df[f"std{W}"] = df["y"].rolling(W).std()

        if dropna:
            df = df.dropna().reset_index(drop=True)

        FEATURES = ["weekday", "playerCount"]
        FEATURES += [f"lag{L}" for L in lags]
        FEATURES += [f"ma{W}" for W in rolling]
        FEATURES += [f"std{W}" for W in rolling]

        return df, FEATURES

    # ============================
    # FORECAST
    # ============================
    def forecast_lightgbm(self, item_id, steps=30, exog_online_df=None):

        df = self.price_history_with_online(item_id).copy()

        df["ds"] = pd.to_datetime(df["ds"])

        # Replace online if provided
        if exog_online_df is not None:
            exog_online_df["ds"] = pd.to_datetime(exog_online_df["ds"])
            df = df.merge(exog_online_df, on="ds", how="left", suffixes=("", "_exog"))

            if "playerCount_exog" in df.columns:
                df["playerCount"] = df["playerCount_exog"].combine_first(df["playerCount"])
                df.drop(columns=["playerCount_exog"], inplace=True)

        # Build historical features
        df_feat, FEATURES = self.make_features(df, dropna=True)

        model = LGBMRegressor(n_estimators=800, learning_rate=0.03, verbose=-1)
        model.fit(df_feat[FEATURES], df_feat["y"])

        # Prepare future dates
        last_date = df["ds"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps)
        future = pd.DataFrame({"ds": future_dates})

        online_forecast = self.forecast_online(steps)
        future = future.merge(online_forecast, on="ds", how="left")

        # Build autoregressive sequence
        tail = df_feat.tail(50).copy()
        full = pd.concat([tail, future], ignore_index=True)

        for i in range(len(tail), len(full)):
            full.loc[i, "weekday"] = full.loc[i, "ds"].weekday()

            for L in [1,7,14]:
                full.loc[i, f"lag{L}"] = full.loc[i - L, "y"]

            for W in [7,14]:
                window = full.loc[i - W:i, "y"]
                full.loc[i, f"ma{W}"] = window.mean()
                full.loc[i, f"std{W}"] = window.std()

            X_row = pd.DataFrame([full.loc[i, FEATURES].values], columns=FEATURES)
            full.loc[i, "y"] = model.predict(X_row)[0]

        forecast = full.iloc[-steps:][["ds", "y"]].rename(columns={"y": "yhat"})
        return forecast

    # ============================
    # GET ITEM NAME
    # ============================
    def get_item_name_by_id(self, item_id):
        session = create_session()
        row = session.execute(
            text("SELECT hash_name FROM items WHERE item_id=:id"),
            {"id": item_id}
        ).fetchone()
        return row[0] if row else f"item_{item_id}"

    # ============================
    # VALIDATION
    # ============================
    def verify_model(self, item_ids, steps=30, min_date_to_plot=None):

        results = []
        save_dir = "verification_lightgbm"
        os.makedirs(save_dir, exist_ok=True)

        for item_id in item_ids:
            hash_name = self.get_item_name_by_id(item_id)
            df = self.price_history_with_online(item_id, end_date="2025-09-01")

            if len(df) <= steps * 2:
                print(f"Skipping {hash_name}")
                continue

            df_feat, FEATURES = self.make_features(df, dropna=True)

            if len(df_feat) <= steps + 5:
                print(f"Skipping {hash_name} — insufficient feature rows")
                continue

            train = df_feat.iloc[:-steps]
            test = df_feat.iloc[-steps:]

            model = LGBMRegressor(n_estimators=500, learning_rate=0.03, verbose=-1)
            model.fit(train[FEATURES], train["y"])

            y_pred = model.predict(test[FEATURES])
            y_true = test["y"].values

            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            results.append({
                "item_id": item_id,
                "item_name": hash_name,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE (%)": mape,
            })

            plt.figure(figsize=(12, 5))

            if min_date_to_plot:
                train_plot = train[train["ds"] >= min_date_to_plot]
            else:
                train_plot = train

            plt.plot(train_plot["ds"], train_plot["y"], label="Train", linewidth=1)

            plt.plot(test["ds"], test["y"], label="Test", linewidth=2)

            plt.plot(test["ds"], y_pred, label="Forecast", linewidth=2)

            plt.title(
                f"LightGBM: Forecast vs Real — {re.sub(r'([$#%&{}_])', r'\\\1', hash_name)}"
            )
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            if mape < 5:
                plt.savefig(
                    f"{save_dir}/item{item_id}_{re.sub(r'[<>:\"/\\\\|?*]', '_', hash_name)}.png"
                )
            plt.close()

            print(f"Verified {item_id}, {hash_name} — MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%")

        return pd.DataFrame(results)
    
    def plot_forecast(self, df, forecast, hash_name, steps, min_date=None, save=True):
        """
        df — исторический DataFrame ['ds', 'y', 'playerCount']
        forecast — DataFrame ['ds', 'yhat']
        """

        if min_date:
            min_date = pd.to_datetime(min_date)
            df_plot = df[df["ds"] >= min_date]
            forecast_plot = forecast[forecast["ds"] >= min_date]
        else:
            df_plot = df
            forecast_plot = forecast

        plt.figure(figsize=(12, 6))

        # История
        plt.plot(
            df_plot["ds"],
            df_plot["y"],
            alpha=0.6,
            label="Historical Price"
        )

        # Прогноз
        plt.plot(
            forecast_plot["ds"],
            forecast_plot["yhat"],
            linewidth=1,
            label="Forecast"
        )

        # plt.fill_between(
        #     forecast_plot["ds"],
        #     forecast_plot["yhat_lower"],
        #     forecast_plot["yhat_upper"],
        #     alpha=0.2,
        #     label="Confidence Interval",
        # )

        plt.title(f"Price Forecast — {hash_name}, {steps} days")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Сохранение
        if save:
            save_dir = "forecasts_lightgbm"
            os.makedirs(save_dir, exist_ok=True)

            filename = f"{save_dir}/{re.sub(r'[<>:\"/\\\\|?*]', '_', hash_name)}.png"
            plt.savefig(filename, dpi=200, bbox_inches="tight")
            print(f"Saved forecast plot to: {filename}")

        plt.show()

    def make_features_online(self, df, lags=[1,7,14], rolling=[7,14], dropna=True):
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").reset_index(drop=True)

        for L in lags:
            df[f"lag{L}"] = df["playerCount"].shift(L)

        for W in rolling:
            df[f"ma{W}"] = df["playerCount"].rolling(W).mean()
            df[f"std{W}"] = df["playerCount"].rolling(W).std()

        if dropna:
            df = df.dropna().reset_index(drop=True)

        FEATURES = []
        FEATURES += [f"lag{L}" for L in lags]
        FEATURES += [f"ma{W}" for W in rolling]
        FEATURES += [f"std{W}" for W in rolling]

        return df, FEATURES
    
    def forecast_online(self, steps=120):
        # Load data
        df_online = self.load_player_online()
        df_online = df_online[["ds", "playerCount"]].copy()

        # Prophet requires columns: ds, y
        df_prophet = df_online.rename(columns={"playerCount": "y"})
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

        # Create model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.8,
            seasonality_mode="multiplicative",   # online often multiplicative
        )

        # Add extra seasonalities if needed
        model.add_seasonality(
            name="monthly",
            period=30.5,
            fourier_order=6
        )

        # Fit
        model.fit(df_prophet)

        # Future dataframe
        future = model.make_future_dataframe(periods=steps)

        # Predict
        forecast = model.predict(future)

        # Return in your format
        result = forecast[["ds", "yhat"]].tail(steps).copy()
        result = result.rename(columns={"yhat": "playerCount"})

        return result


# ============================
# USAGE
# ============================
light_gbm_model = LightGBMModel()

ids_to_verify = [25355, 25395, 3153, 26954, 19010, 6317, 25201, 11773, 19356]
ids_to_verify = range(1, 30000)

results = light_gbm_model.verify_model(ids_to_verify, steps=30, min_date_to_plot="2024-01-01")
results.to_csv("verification_lightgbm/metrics.csv", index=False)


# TEST

# light_gbm_model = LightGBMModel()
# ID = 25201
# STEPS = 30
# hash_name = light_gbm_model.get_item_name_by_id(ID)

# # Получаем исторические данные
# df = light_gbm_model.price_history_with_online(ID)

# # Получаем прогноз
# forecast = light_gbm_model.forecast_lightgbm(ID, steps=STEPS)
# print(forecast)

# # Строим график
# light_gbm_model.plot_forecast(df, forecast, hash_name, STEPS)
