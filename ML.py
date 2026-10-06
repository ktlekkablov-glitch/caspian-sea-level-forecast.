import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DB_USER = "postgres"
DB_PASSWORD = "admin123"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "caspian_db"
TABLE = "caspian_data"

TEST_START = "2023-01-01"
FUTURE_STEPS = 24
TREND_YEARS = 10
RANDOM_STATE = 42

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

df = pd.read_sql(f"SELECT * FROM {TABLE} ORDER BY date", engine, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
df["t"] = np.arange(len(df))

print(f"Загружено: {len(df)} строк")
print(f"Период: {df['date'].min().date()} — {df['date'].max().date()}")

cutoff = df["date"].max() - pd.DateOffset(years=TREND_YEARS)
trend_data = df[df["date"] >= cutoff]

coef = np.polyfit(trend_data["t"], trend_data["sea_level"], 1)
trend_fn = np.poly1d(coef)

df["trend"] = trend_fn(df["t"])
df["residual"] = df["sea_level"] - df["trend"]

print(f"Оценка тренда: {coef[0]:.4f} м/месяц")


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()

    d["month"] = d["date"].dt.month
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)

    # лаги остатка
    d["res_lag1"] = d["residual"].shift(1)
    d["res_lag3"] = d["residual"].shift(3)
    d["res_lag12"] = d["residual"].shift(12)

    # факторные лаги
    d["volga_lag1"] = d["volga"].shift(1)

    return d

df_feat = make_features(df).dropna().reset_index(drop=True)

features = [
    "month_sin", "month_cos",
    "res_lag1", "res_lag3", "res_lag12",
    "temperature", "precipitation", "evaporation", "volga_lag1"
]

target = "residual"

mask = df_feat["date"] >= TEST_START

X_train = df_feat.loc[~mask, features]
y_train = df_feat.loc[~mask, target]

X_test = df_feat.loc[mask, features]
y_test = df_feat.loc[mask, target]

trend_test = df_feat.loc[mask, "trend"].values
sea_test = df_feat.loc[mask, "sea_level"].values
dates_test = df_feat.loc[mask, "date"]

print(f"Train: {len(X_train)}")
print(f"Test : {len(X_test)}")


model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=1.5,
    random_state=RANDOM_STATE,
    verbosity=0
)

model.fit(X_train, y_train)

pred_res_test = model.predict(X_test)
pred_sea_test = trend_test + pred_res_test

rmse = np.sqrt(mean_squared_error(sea_test, pred_sea_test))
mae = mean_absolute_error(sea_test, pred_sea_test)
r2 = r2_score(sea_test, pred_sea_test)

print("\nМетрики на тесте 2023–2024:")
print(f"RMSE = {rmse:.4f}")
print(f"MAE  = {mae:.4f}")
print(f"R²   = {r2:.4f}")

results_test = pd.DataFrame({
    "date": dates_test,
    "actual": sea_test,
    "predicted": pred_sea_test
})

print("\nТестовый прогноз:")
print(results_test)

monthly_stats = df.groupby(df["date"].dt.month)[
    ["temperature", "precipitation", "volga", "evaporation"]
].median()

hist = df_feat.copy()
last_date = hist["date"].iloc[-1]
last_t = hist["t"].iloc[-1]

future_rows = []

for step in range(1, FUTURE_STEPS + 1):
    next_date = last_date + pd.DateOffset(months=step)
    m = next_date.month
    next_t = last_t + step

    next_trend = trend_fn(next_t)

    row = {
        "date": next_date,
        "t": next_t,
        "sea_level": np.nan,
        "trend": next_trend,
        "residual": np.nan,
        "temperature": monthly_stats.loc[m, "temperature"],
        "precipitation": monthly_stats.loc[m, "precipitation"],
        "volga": monthly_stats.loc[m, "volga"],
        "evaporation": monthly_stats.loc[m, "evaporation"]
    }

    temp_hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    temp_hist = make_features(temp_hist)

    X_future = temp_hist.iloc[[-1]][features]
    pred_res = model.predict(X_future)[0]
    pred_sea = next_trend + pred_res

    row["residual"] = pred_res
    row["sea_level"] = pred_sea

    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    future_rows.append(row)

forecast_df = pd.DataFrame(future_rows)[["date", "sea_level"]]
forecast_df.columns = ["date", "forecast"]

print("\nПрогноз на 2025–2026:")
print(forecast_df)

importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nВажность признаков:")
print(importance_df)

plt.figure(figsize=(12, 6))
plt.plot(results_test["date"], results_test["actual"], label="Фактический уровень", linewidth=2)
plt.plot(results_test["date"], results_test["predicted"], label="Прогноз XGBoost", linewidth=2)
plt.title("Тестовый период: факт и прогноз XGBoost")
plt.xlabel("Дата")
plt.ylabel("Уровень моря, м")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


last_actual = df.tail(36)

plt.figure(figsize=(12, 6))
plt.plot(last_actual["date"], last_actual["sea_level"], label="Фактический уровень", linewidth=2)
plt.plot(forecast_df["date"], forecast_df["forecast"], label="Прогноз XGBoost", linewidth=2)

for _, row in forecast_df.iloc[::3].iterrows():
    plt.text(
        row["date"],
        row["forecast"],
        f"{row['forecast']:.2f}",
        fontsize=8,
        ha="center",
        va="bottom"
    )

plt.title("Прогноз уровня Каспийского моря на 2025–2026 (XGBoost)")
plt.xlabel("Дата")
plt.ylabel("Уровень моря, м")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
top_imp = importance_df.sort_values("importance", ascending=True)
plt.barh(top_imp["feature"], top_imp["importance"])
plt.title("Важность признаков XGBoost")
plt.xlabel("Важность")
plt.ylabel("Признак")
plt.tight_layout()
plt.show()