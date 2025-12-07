import os
import re
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TimeSeriesDataSet, Baseline, TemporalFusionTransformer
from pytorch_forecasting.metrics import SMAPE, MAE
from pytorch_forecasting.data import GroupNormalizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from database import create_session

seed_everything(42)


class TFTForecaster:
    def __init__(
        self,
        max_encoder_length: int = 90,
        max_prediction_length: int = 30,
        trainer_kwargs: dict = None,
        checkpoint_dir: str = "tft_checkpoints",
        verbosity: int = 1,
    ):
        """
        max_encoder_length: сколько прошлых шагов даём в качестве входа
        max_prediction_length: шаг прогноза по умолчанию
        trainer_kwargs: dict с аргументами для pytorch-lightning Trainer
        """
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.verbosity = verbosity

        # trainer defaults
        default_trainer = {
            "max_epochs": 30,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1 if torch.cuda.is_available() else None,
            "logger": False,
            "enable_progress_bar": False,
        }
        
        if trainer_kwargs:
            default_trainer.update(trainer_kwargs)
        self.trainer_kwargs = default_trainer

        # models placeholders
        self.player_model = None
        self.price_model = None

        # datasets placeholders
        self.player_tsd = None
        self.price_tsd = None

    # ----------------------------------------
    # Utilities for loading from DB
    # ----------------------------------------
    def price_history_by_id(self, item_id, provider="buff163", start_date=None, end_date=None, limit_last_n=None):
        """
        Returns DataFrame columns ['ds','y','listings']
        """

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
        sql += " ORDER BY timestamp ASC"
        if limit_last_n:
            sql += f" LIMIT {limit_last_n}"

        result = session.execute(text(sql), params)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        if df.empty:
            return pd.DataFrame(columns=["ds", "y", "listings"])

        df["ds"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.sort_values("ds").rename(columns={"price": "y"})[["ds", "y", "listings"]]
        # enforce daily freq & interpolate
        df = df.set_index("ds").asfreq("D")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.interpolate().reset_index()
        return df

    def load_player_online(self, json_path="scrapers/cs_online.json"):
        """
        Loads playerCount JSON and returns DataFrame ['ds','playerCount'] daily freq, tz removed.
        JSON record example:
          {"timestamp": "2015-04-19T00:00:00+00:00", "playerCount": 549694}
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = pd.DataFrame(json.load(f))
        if data.empty:
            return pd.DataFrame(columns=["ds", "playerCount"])
        data["ds"] = pd.to_datetime(data["timestamp"]).dt.tz_convert(None).dt.tz_localize(None).dt.normalize()
        df = data[["ds", "playerCount"]].sort_values("ds")
        df = df.set_index("ds").asfreq("D").ffill().reset_index()
        return df

    def load_item_static(self, item_id):
        """
        Returns a dict with static item fields (category_id, weapon_id, collection_id, rarity_id, finish_id, finish_style_id,
        team_id, tournament_id, sticker_slots, doppler_phases, is_stattrak_available, is_souvenir_available, finish_style_catalog).
        """
        session = create_session()
        sql = """
            SELECT category_id, weapon_id, collection_id, rarity_id, finish_id, finish_style_id,
                   team_id, tournament_id, sticker_slots, doppler_phases,
                   is_stattrak_available, is_souvenir_available, finish_style_catalog
            FROM items
            WHERE item_id = :item_id
            LIMIT 1
        """
        row = session.execute(text(sql), {"item_id": item_id}).fetchone()
        if not row:
            return {}
        cols = ["category_id", "weapon_id", "collection_id", "rarity_id", "finish_id", "finish_style_id",
                "team_id", "tournament_id", "sticker_slots", "doppler_phases",
                "is_stattrak_available", "is_souvenir_available", "finish_style_catalog"]
        return dict(zip(cols, row))

    # ----------------------------------------
    # Build global training frame for prices
    # ----------------------------------------
    def build_price_dataframe(self, item_ids, provider="buff163", start_date=None, end_date=None):
        """
        Build a single dataframe with columns:
          group (item_id), ds, y, playerCount, + static categorical + static numeric
        """
        frames = []
        df_online = self.load_player_online()
        if df_online.empty:
            raise ValueError("player online data empty. Place cs_online.json correctly.")
        for item_id in item_ids:
            df_price = self.price_history_by_id(item_id, provider, start_date, end_date)
            if df_price.empty:
                continue
            df_price = df_price[["ds", "y"]].copy()
            df_price["item_id"] = int(item_id)
            # join online
            df_price = df_price.merge(df_online, on="ds", how="left")
            df_price["playerCount"] = df_price["playerCount"].ffill().bfill()
            # add static features from items table
            static = self.load_item_static(item_id)
            # convert booleans to ints
            for k, v in static.items():
                df_price[k] = v
            frames.append(df_price)
        if not frames:
            return pd.DataFrame()
        df_all = pd.concat(frames, ignore_index=True, sort=False)
        # common time_idx: days since global min
        df_all["ds"] = pd.to_datetime(df_all["ds"])
        min_date = df_all["ds"].min()
        df_all["time_idx"] = (df_all["ds"] - min_date).dt.days.astype(int)
        # group id col name: 'item_id'
        df_all = df_all.sort_values(["item_id", "ds"]).reset_index(drop=True)
        return df_all

    # ----------------------------------------
    # make TimeSeriesDataSet for price model
    # ----------------------------------------
    def create_price_tsd(self, df_all, categorical_static, numeric_static, target="y"):
        """
        categorical_static: list of column names to treat as static categorical
        numeric_static: list of static numeric columns
        """
        if df_all.empty:
            raise ValueError("df_all empty")

        # The pytorch-forecasting TimeSeriesDataSet expects specific column names:
        # group_id, time_idx, target, time_varying_known_reals, time_varying_unknown_reals, static_categoricals, static_reals
        df = df_all.copy()
        df["group_id"] = df["item_id"].astype(str)

        # time_varying_known_reals: playerCount and time_idx (time_idx is also known)
        time_varying_known_reals = ["time_idx", "playerCount"]
        # time_varying_unknown_reals: target lags are implicitly encoded by encoder length (we won't put lag columns explicitly)
        time_varying_unknown_reals = [target]

        static_categoricals = categorical_static
        static_reals = numeric_static

        # fillna for static reals
        for c in static_reals:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c].fillna(0.0))

        # Build dataset
        tsd = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=target,
            group_ids=["group_id"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        return tsd

    # ----------------------------------------
    # PlayerCount dataset (single global series)
    # ----------------------------------------
    def create_player_tsd(self, df_online):
        """
        df_online must have ['ds','playerCount'] daily
        We'll add group_id='player', time_idx starting from 0
        """
        df = df_online.copy().rename(columns={"playerCount": "y"})
        df["group_id"] = "player"
        df = df.sort_values("ds").reset_index(drop=True)
        min_date = df["ds"].min()
        df["time_idx"] = (df["ds"] - min_date).dt.days.astype(int)
        tsd = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="y",
            group_ids=["group_id"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["y"],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        return tsd

    # ----------------------------------------
    # Train player model (TFT)
    # ----------------------------------------
    def train_player_model(self, max_epochs=30, fast_dev_run=False):
        df_online = self.load_player_online()
        if df_online.empty:
            raise ValueError("No online data")

        self.player_tsd = self.create_player_tsd(df_online)
        # train/val split (last prediction window as validation)
        validation = self.player_tsd.split_before(0.8)  # small series -> ok
        train_dataloader = self.player_tsd.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_dataloader = self.player_tsd.to_dataloader(train=False, batch_size=64, num_workers=0)

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename="player_tft-{epoch}-{val_loss:.4f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        trainer = Trainer(
            callbacks=[EarlyStopping(monitor="val_loss", patience=5), checkpoint_callback],
            max_epochs=max_epochs,
            **self.trainer_kwargs,
        )
        tft = TemporalFusionTransformer.from_dataset(
            self.player_tsd,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=MAE(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        trainer.fit(
            tft,
            train_dataloader,
            val_dataloader,
        )
        # load best checkpoint
        best_path = checkpoint_callback.best_model_path
        if best_path:
            self.player_model = TemporalFusionTransformer.load_from_checkpoint(best_path).to(self.device)
        else:
            self.player_model = tft.to(self.device)
        if self.verbosity:
            print("Player model trained and loaded:", getattr(self.player_model, "checkpoint_path", best_path))

    # ----------------------------------------
    # Forecast playerCount
    # ----------------------------------------
    def forecast_player(self, steps=120):
        if self.player_model is None:
            raise ValueError("Player model not trained. Call train_player_model() first.")
        # get online history
        df_online = self.load_player_online()
        min_date = df_online["ds"].min()
        last_time_idx = (df_online["ds"].max() - min_date).days
        # we will predict from last_time_idx+1 ... last_time_idx+steps
        # build encoder data (last encoder length)
        encoder_length = min(self.max_encoder_length, len(df_online))
        encoder_data = df_online.tail(encoder_length).copy()
        encoder_data["group_id"] = "player"
        encoder_data["time_idx"] = (encoder_data["ds"] - min_date).dt.days.astype(int)

        # create dataset for prediction
        predict_tsd = TimeSeriesDataSet.from_dataset(
            self.player_tsd, encoder_data, predict=True, stop_randomization=True
        )
        predict_dl = predict_tsd.to_dataloader(train=False, batch_size=1, num_workers=0)

        raw_predictions, x = self.player_model.predict(predict_dl, mode="raw", return_x=True)
        # we get prediction for max_prediction_length steps. If steps > max_prediction_length, iterate
        preds = self.player_model.predict(predict_dl, mode="prediction")
        preds = preds[0]  # array length = max_prediction_length
        # if user asks fewer steps, slice
        preds = preds[:steps]
        future_dates = pd.date_range(df_online["ds"].max() + timedelta(days=1), periods=steps, freq="D")
        df_pred = pd.DataFrame({"ds": future_dates, "playerCount": preds})
        return df_pred

    # ----------------------------------------
    # Train price model (global TFT on many item groups)
    # ----------------------------------------
    def train_price_model(self, item_ids, max_epochs=30, categorical_static=None, numeric_static=None):
        """
        item_ids: list of item ids to include in training
        categorical_static: list of static categorical columns to pass to TFT (e.g. ["weapon_id","rarity_id"...])
        numeric_static: list of static numeric columns (e.g. ["sticker_slots","doppler_phases"...])
        """
        df_all = self.build_price_dataframe(item_ids)
        if df_all.empty:
            raise ValueError("No price data for given item_ids")

        # default statics if not provided
        if categorical_static is None:
            categorical_static = ["category_id", "weapon_id", "collection_id", "rarity_id", "finish_id",
                                  "finish_style_id", "team_id", "tournament_id"]
        if numeric_static is None:
            numeric_static = ["sticker_slots", "doppler_phases", "is_stattrak_available",
                              "is_souvenir_available", "finish_style_catalog"]

        self.price_tsd = self.create_price_tsd(df_all, categorical_static, numeric_static)

        # split
        training_cutoff = self.price_tsd.time_idx.max() - self.max_prediction_length
        train_tsd = self.price_tsd.subset(lambda x: x.time_idx <= training_cutoff)
        val_tsd = self.price_tsd.subset(lambda x: x.time_idx > training_cutoff - self.max_encoder_length)

        train_dl = train_tsd.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_dl = val_tsd.to_dataloader(train=False, batch_size=64, num_workers=0)

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename="price_tft-{epoch}-{val_loss:.4f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        trainer = Trainer(
            callbacks=[EarlyStopping(monitor="val_loss", patience=5), checkpoint_callback],
            max_epochs=max_epochs,
            **self.trainer_kwargs,
        )

        tft = TemporalFusionTransformer.from_dataset(
            self.price_tsd,
            learning_rate=0.03,
            hidden_size=64,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            loss=SMAPE(),
            log_interval=10,
            reduce_on_plateau_patience=4,
            output_size=1,
        )
        trainer.fit(tft, train_dl, val_dl)

        best_path = checkpoint_callback.best_model_path
        if best_path:
            self.price_model = TemporalFusionTransformer.load_from_checkpoint(best_path).to(self.device)
        else:
            self.price_model = tft.to(self.device)
        if self.verbosity:
            print("Price model trained and loaded:", getattr(self.price_model, "checkpoint_path", best_path))

    # ----------------------------------------
    # Forecast price for a single item using global price_model and external player forecast
    # ----------------------------------------
    def forecast_price(self, item_id, steps=30, use_player_forecast=True):
        """
        Returns dataframe ['ds','yhat'] for requested steps
        """
        if self.price_model is None:
            raise ValueError("Price model not trained. Call train_price_model() first.")

        # build single-item df (history)
        df_item = self.price_history_by_id(item_id)
        if df_item.empty:
            raise ValueError("No price history for item")

        # merge static
        static = self.load_item_static(item_id)
        for k, v in static.items():
            df_item[k] = v

        # merge player history + forecast
        df_online = self.load_player_online()
        if use_player_forecast:
            # ensure player model trained
            if self.player_model is None:
                raise ValueError("player model is not trained; call train_player_model()")
            pf = self.forecast_player(steps=steps)
            # combine player history + pf for dates
            df_online_all = pd.concat([df_online, pf], ignore_index=True, sort=False)
        else:
            df_online_all = df_online.copy()

        df_item = df_item.merge(df_online_all, on="ds", how="left")
        df_item["playerCount"] = df_item["playerCount"].ffill().bfill()

        # prepare time_idx relative to global price_tsd min date
        # note: price_tsd was built from training set; extract its min ds
        # we will compute time_idx consistent with price_tsd's min
        min_date = self.price_tsd.base_dataset.dataframe()["ds"].min()
        df_item["time_idx"] = (pd.to_datetime(df_item["ds"]) - pd.to_datetime(min_date)).dt.days.astype(int)

        # build encoder frame: last max_encoder_length rows
        encoder_data = df_item.tail(self.max_encoder_length).copy()
        encoder_data["group_id"] = str(item_id)
        # Build predict dataset
        predict_tsd = TimeSeriesDataSet.from_dataset(self.price_tsd, encoder_data, predict=True, stop_randomization=True)
        predict_dl = predict_tsd.to_dataloader(train=False, batch_size=1, num_workers=0)

        preds = self.price_model.predict(predict_dl, mode="prediction")
        preds = preds[0]  # array length = max_prediction_length
        preds = preds[:steps]
        future_dates = pd.date_range(df_item["ds"].max() + timedelta(days=1), periods=steps, freq="D")
        return pd.DataFrame({"ds": future_dates, "yhat": preds})

    # ----------------------------------------
    # Verify model (per-item)
    # ----------------------------------------
    def verify_model(self, item_ids, steps=30, min_date_to_plot=None, use_player_forecast=True):
        """
        For each item: hide last `steps` rows, forecast, compute metrics, optionally plot.
        """
        results = []
        for item_id in item_ids:
            hash_name = self.get_item_name_by_id(item_id) if hasattr(self, "get_item_name_by_id") else str(item_id)
            try:
                # build per-item dataset including player history (no pf)
                df_item = self.price_history_by_id(item_id)
                if df_item.empty or len(df_item) < (self.max_encoder_length + steps):
                    print(f"Skipping {hash_name} — insufficient history")
                    continue
                df_online = self.load_player_online()
                df_item = df_item.merge(df_online, on="ds", how="left")
                df_item["playerCount"] = df_item["playerCount"].ffill().bfill()

                train_df = df_item.iloc[0:-steps].copy()
                test_df = df_item.iloc[-steps:].copy()

                # build encoder_data from train_df tail
                # merge static
                static = self.load_item_static(item_id)
                for k, v in static.items():
                    train_df[k] = v
                    test_df[k] = v

                # prepare time_idx consistent with price_tsd min
                min_date = self.price_tsd.base_dataset.dataframe()["ds"].min()
                train_df["time_idx"] = (pd.to_datetime(train_df["ds"]) - pd.to_datetime(min_date)).dt.days.astype(int)
                test_df["time_idx"] = (pd.to_datetime(test_df["ds"]) - pd.to_datetime(min_date)).dt.days.astype(int)

                # create encoder dataset from train_df tail
                encoder_data = train_df.tail(self.max_encoder_length).copy()
                encoder_data["group_id"] = str(item_id)
                # for predicting we need to append future rows (with playerCount known or from forecast)
                # create future rows
                future_dates = pd.date_range(test_df["ds"].min(), periods=steps, freq="D")
                future = pd.DataFrame({"ds": future_dates})
                # attach playerCount: if use_player_forecast -> forecast player, else use last known or interpolation
                if use_player_forecast:
                    pf = self.forecast_player(steps=steps)
                    future = future.merge(pf, on="ds", how="left")
                else:
                    # fill with last known
                    last_online = train_df["playerCount"].iloc[-1]
                    future["playerCount"] = last_online
                # combine
                pred_input = pd.concat([encoder_data, future], ignore_index=True, sort=False)
                # align static fields
                for k, v in static.items():
                    pred_input[k] = v
                # compute time_idx relative to price_tsd min
                pred_input["time_idx"] = (pd.to_datetime(pred_input["ds"]) - pd.to_datetime(min_date)).dt.days.astype(int)
                pred_input["group_id"] = str(item_id)

                predict_tsd = TimeSeriesDataSet.from_dataset(self.price_tsd, pred_input, predict=True, stop_randomization=True)
                predict_dl = predict_tsd.to_dataloader(train=False, batch_size=1, num_workers=0)

                preds = self.price_model.predict(predict_dl, mode="prediction")
                preds = preds[0][:steps]

                y_true = test_df["y"].values
                y_pred = preds

                mae = float(np.mean(np.abs(y_true - y_pred)))
                rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

                results.append({"item_id": item_id, "item_name": hash_name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape})

                # plot optionally
                if min_date_to_plot:
                    plot_from = pd.to_datetime(min_date_to_plot)
                else:
                    plot_from = train_df["ds"].min()
                # build plot df: history + forecast
                history_plot = train_df[train_df["ds"] >= plot_from]
                forecast_plot = pd.DataFrame({"ds": test_df["ds"], "yhat": y_pred})
                plt.figure(figsize=(10, 5))
                plt.plot(history_plot["ds"], history_plot["y"], label="Train")
                plt.plot(test_df["ds"], y_true, label="True")
                plt.plot(forecast_plot["ds"], forecast_plot["yhat"], label="Forecast")
                plt.title(f"TFT Forecast — {hash_name}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                save_dir = "verification_tft"
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(f"{save_dir}/item{item_id}_{re.sub('[<>:\"/\\\\|?*]', '_', hash_name)}.png")
                plt.close()
                print(f"Verified {item_id} — MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%")
            except Exception as e:
                print(f"Error verifying {item_id}: {e}")
        return pd.DataFrame(results)

    # ----------------------------------------
    # helper to get item name (same as before)
    # ----------------------------------------
    def get_item_name_by_id(self, item_id):
        session = create_session()
        sql = "SELECT hash_name FROM items WHERE item_id = :item_id LIMIT 1"
        row = session.execute(text(sql), {"item_id": item_id}).fetchone()
        return row[0] if row and row[0] else f"item_{item_id}"

    # ----------------------------------------
    # plotting helper for standalone usage
    # ----------------------------------------
    def plot_forecast(self, df_history, df_forecast, hash_name, steps, min_date=None, save=True):
        if min_date:
            min_date = pd.to_datetime(min_date)
            history = df_history[df_history["ds"] >= min_date]
        else:
            history = df_history
        forecast = df_forecast

        plt.figure(figsize=(12, 6))
        plt.plot(history["ds"], history["y"], label="History")
        plt.plot(forecast["ds"], forecast["yhat"], label="Forecast")
        plt.title(f"Forecast — {hash_name}")
        plt.legend()
        plt.grid(True)
        if save:
            os.makedirs("forecasts_tft", exist_ok=True)
            fname = f"forecasts_tft/{re.sub(r'[<>:\"/\\\\|?*]', '_', hash_name)}.png"
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            print("Saved:", fname)
        plt.show()


print(torch.__version__)
print(torch.version.cuda)