import re
from functools import lru_cache
from typing import Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import Item, create_session
from ML.LightGBM import LightGBMModel
from ML.Prophet import ProphetModel
from ML.arima import Arima


# ----------------------------------------
# Caching helpers
# ----------------------------------------
@st.cache_resource(show_spinner=False)
def get_lightgbm_model() -> LightGBMModel:
    return LightGBMModel()


@st.cache_resource(show_spinner=False)
def get_prophet_model() -> ProphetModel:
    return ProphetModel()


@st.cache_resource(show_spinner=False)
def get_arima_model() -> Arima:
    return Arima()


@lru_cache(maxsize=256)
def cached_item_by_id(item_id: int) -> Dict:
    """Load item metadata once per ID."""
    with create_session() as session:
        item: Item = session.query(Item).filter(Item.item_id == item_id).first()
        if not item:
            return {}

        def safe_rel(obj, attr):
            return getattr(obj, attr) if obj else None

        return {
            "item_id": item.item_id,
            "hash_name": item.hash_name,
            "pricempire_id": item.pricempire_id,
            "steam_url": item.steam_url,
            "pricempire_url": item.pricempire_url,
            "item_image_url": item.item_image_url,
            "category": safe_rel(item.category, "category_name"),
            "weapon": safe_rel(item.weapon, "weapon_name"),
            "rarity": safe_rel(item.rarity, "rarity_name"),
            "collection": safe_rel(item.collection, "collection_name"),
            "finish": safe_rel(item.finish, "finish_name"),
        }


def search_items(query: str, limit: int = 25) -> List[Dict]:
    """Поиск предметов по части названия."""
    pattern = f"%{query}%"
    with create_session() as session:
        rows = (
            session.query(Item)
            .filter(Item.hash_name.ilike(pattern))
            .order_by(Item.hash_name)
            .limit(limit)
            .all()
        )
        return [{"item_id": r.item_id, "hash_name": r.hash_name} for r in rows]


# ----------------------------------------
# Validation helpers (lightweight, per item)
# ----------------------------------------
def evaluate_lightgbm(item_id: int, steps: int) -> Tuple[Dict, pd.DataFrame]:
    model = get_lightgbm_model()
    df = model.price_history_with_online(item_id)
    if len(df) <= steps * 2:
        raise ValueError("Недостаточно данных для валидации.")

    df_feat, features = model.make_features(df, dropna=True)
    train = df_feat.iloc[:-steps]
    test = df_feat.iloc[-steps:]

    from lightgbm import LGBMRegressor

    reg = LGBMRegressor(n_estimators=400, learning_rate=0.03, verbose=-1)
    reg.fit(train[features], train["y"])
    y_pred = reg.predict(test[features])

    metrics = _calc_metrics(test["y"].values, y_pred)
    eval_df = pd.DataFrame({"ds": test["ds"], "y_true": test["y"], "y_pred": y_pred})
    return metrics, eval_df


def evaluate_prophet(item_id: int, steps: int) -> Tuple[Dict, pd.DataFrame]:
    model = get_prophet_model()
    df = model.price_history_with_online(item_id)
    if len(df) <= steps * 2:
        raise ValueError("Недостаточно данных для валидации.")

    train = df.iloc[:-steps].copy()
    test = df.iloc[-steps:].copy()

    from prophet import Prophet

    m = Prophet(daily_seasonality=True)
    m.add_regressor("playerCount")
    m.fit(train)

    future = m.make_future_dataframe(periods=steps)
    future = future.merge(df[["ds", "playerCount"]], on="ds", how="left").ffill()
    forecast = m.predict(future)
    y_pred = forecast.iloc[-steps:]["yhat"].values

    metrics = _calc_metrics(test["y"].values, y_pred)
    eval_df = pd.DataFrame({"ds": test["ds"], "y_true": test["y"], "y_pred": y_pred})
    return metrics, eval_df


def _calc_metrics(y_true, y_pred) -> Dict:
    import numpy as np
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


# ----------------------------------------
# Plot helpers
# ----------------------------------------
def plot_forecast(history: pd.DataFrame, forecast: pd.DataFrame, title: str):
    """
    history: DataFrame with columns ['ds','y']
    forecast: DataFrame with columns ['ds','yhat']
    """
    hist = history.rename(columns={"y": "value"})
    fc = forecast.rename(columns={"yhat": "value"})
    hist["type"] = "history"
    fc["type"] = "forecast"
    combined = pd.concat([hist, fc], ignore_index=True)

    chart = (
        alt.Chart(combined)
        .mark_line()
        .encode(
            x="ds:T",
            y="value:Q",
            color="type:N",
            tooltip=["ds:T", "value:Q", "type:N"],
        )
        .properties(height=360, title=title)
    )
    st.altair_chart(chart, use_container_width=True)


def plot_validation(eval_df: pd.DataFrame, title: str):
    chart = (
        alt.Chart(eval_df)
        .transform_fold(["y_true", "y_pred"], as_=["label", "value"])
        .mark_line()
        .encode(
            x="ds:T",
            y="value:Q",
            color="label:N",
            tooltip=["ds:T", "value:Q", "label:N"],
        )
        .properties(height=320, title=title)
    )
    st.altair_chart(chart, use_container_width=True)


# ----------------------------------------
# Streamlit UI
# ----------------------------------------
st.set_page_config(
    page_title="CS2 Skin Price Prediction",
    layout="wide",
    page_icon=":chart_with_upwards_trend:",
)

st.title("CS2 Skin Price Prediction — Streamlit")
st.markdown(
    "Поиск предмета → выбор модели → прогноз на N шагов. "
    "Доступна лёгкая валидация на последних точках."
)

with st.sidebar:
    st.header("Параметры")
    search_query = st.text_input(
        "Поиск предмета", placeholder="Начните вводить название..."
    )
    model_choice = st.selectbox(
        "Модель",
        ["LightGBM", "Prophet", "ARIMA"],
        help="TFT не добавлен в UI из-за долгого времени обучения.",
    )
    steps = st.slider("Горизонт прогноза (дней)", 7, 120, 30, 1)
    do_validation = st.checkbox("Выполнить валидацию (последние N точек)", value=False)

    st.markdown("---")
    st.markdown("Чтобы запустить: `streamlit run streamlit_app.py`")


# Search + selection
results = []
if search_query:
    results = search_items(search_query, limit=25)

col_list, col_action = st.columns([2, 1])
with col_list:
    st.subheader("Результаты поиска")
    if results:
        options = {
            f"{row['hash_name']} (id={row['item_id']})": row["item_id"]
            for row in results
        }
        selected_label = st.selectbox("Выберите предмет", list(options.keys()))
        selected_item_id = options[selected_label]
    else:
        selected_item_id = None
        st.info("Начните вводить название предмета для поиска.")

with col_action:
    st.subheader("Действия")
    run_forecast = st.button("Построить прогноз", disabled=not results)


def display_item_meta(item_id: int):
    meta = cached_item_by_id(item_id)
    if not meta:
        st.warning("Метаданные не найдены в базе.")
        return
    st.markdown(f"**Hash name:** {meta.get('hash_name')}")
    st.markdown(f"**Item ID:** {meta.get('item_id')}")
    st.markdown(
        f"**Категория:** {meta.get('category') or '—'} | "
        f"Оружие: {meta.get('weapon') or '—'} | "
        f"Редкость: {meta.get('rarity') or '—'}"
    )
    links = []
    if meta.get("steam_url"):
        links.append(f"[Steam]({meta['steam_url']})")
    if meta.get("pricempire_url"):
        links.append(f"[PriceEmpire]({meta['pricempire_url']})")
    if links:
        st.markdown("Ссылки: " + " • ".join(links))
    if meta.get("item_image_url"):
        st.image(meta["item_image_url"], width=200)


def run_model_and_plot(item_id: int, model_name: str, steps: int):
    if model_name == "LightGBM":
        model = get_lightgbm_model()
        history = model.price_history_with_online(item_id)
        forecast = model.forecast_lightgbm(item_id, steps=steps)
        plot_forecast(history[["ds", "y"]], forecast, "LightGBM прогноз")
        return history, forecast

    if model_name == "Prophet":
        model = get_prophet_model()
        history = model.price_history_with_online(item_id)
        forecast = model.forecast_prophet_with_online(history, steps=steps)
        forecast = forecast.rename(columns={"yhat": "yhat"})
        plot_forecast(history[["ds", "y"]], forecast[["ds", "yhat"]], "Prophet прогноз")
        return history, forecast

    if model_name == "ARIMA":
        model = get_arima_model()
        df = model.price_history_by_id(item_id)
        if df.empty:
            st.error("Нет данных для этого предмета.")
            return None, None
        forecast = model.forecast_sarimax(df, steps=steps)
        forecast = forecast.reset_index().rename(
            columns={"index": "ds", "forecast": "yhat"}
        )
        hist = df.reset_index().rename(columns={"timestamp": "ds", "price": "y"})
        plot_forecast(hist[["ds", "y"]], forecast[["ds", "yhat"]], "ARIMA прогноз")
        return hist, forecast

    st.error("Неизвестная модель.")
    return None, None


if run_forecast and selected_item_id:
    with st.spinner("Строим прогноз..."):
        display_item_meta(selected_item_id)
        history, forecast = run_model_and_plot(selected_item_id, model_choice, steps)

        if do_validation and model_choice in ["LightGBM", "Prophet"]:
            st.markdown("### Валидация на последних точках")
            try:
                if model_choice == "LightGBM":
                    metrics, eval_df = evaluate_lightgbm(selected_item_id, steps)
                else:
                    metrics, eval_df = evaluate_prophet(selected_item_id, steps)
                st.write(metrics)
                plot_validation(eval_df, "Прогноз vs Истина")
            except Exception as exc:
                st.warning(f"Не удалось выполнить валидацию: {exc}")


st.markdown("---")
st.caption(
    "Веб-интерфейс: поиск предмета → выбор модели → прогноз. "
    "Серверная часть обращается к БД и вызывает модуль прогнозирования; "
    "результаты визуализируются интерактивно."
)
