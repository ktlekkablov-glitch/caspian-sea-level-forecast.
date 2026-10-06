library(DBI)
library(RPostgres)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(corrplot)
library(forecast)
library(tseries)
library(lmtest)
library(car)
library(purrr)

graphics.off()
par(mfrow = c(1, 1))
par(mar = c(5, 4, 4, 2) + 0.1)

con <- dbConnect(
  RPostgres::Postgres(),
  dbname = "caspian_db",
  host = "localhost",
  port = 5432,
  user = "postgres",
  password = "admin123"
)

df <- dbGetQuery(con, "SELECT * FROM caspian_data ORDER BY date")
dbDisconnect(con)

df <- df %>%
  mutate(
    date = as.Date(date),
    sea_level = as.numeric(sea_level),
    temperature = as.numeric(temperature),
    precipitation = as.numeric(precipitation),
    volga = as.numeric(volga),
    evaporation = as.numeric(evaporation)
  ) %>%
  filter(!is.na(date)) %>%
  arrange(date) %>%
  mutate(
    month_num = month(date),
    month_ru = factor(
      month_num,
      levels = 1:12,
      labels = c("Янв", "Фев", "Мар", "Апр", "Май", "Июн",
                 "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек")
    )
  )

numeric_cols <- c("sea_level", "temperature", "precipitation", "volga", "evaporation")
months_ru <- c("Янв", "Фев", "Мар", "Апр", "Май", "Июн",
               "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек")

str(df)
print(sapply(df, class))
print(c(min(df$date), max(df$date)))
print(colSums(is.na(df)))

summary_table <- df %>%
  select(all_of(numeric_cols)) %>%
  summarise(across(everything(), list(
    mean   = ~ round(mean(., na.rm = TRUE), 3),
    sd     = ~ round(sd(., na.rm = TRUE), 3),
    min    = ~ round(min(., na.rm = TRUE), 3),
    q25    = ~ round(quantile(., 0.25, na.rm = TRUE), 3),
    median = ~ round(median(., na.rm = TRUE), 3),
    q75    = ~ round(quantile(., 0.75, na.rm = TRUE), 3),
    max    = ~ round(max(., na.rm = TRUE), 3)
  ))) %>%
  pivot_longer(
    everything(),
    names_to = c("variable", "stat"),
    names_sep = "_(?=[^_]+$)"
  ) %>%
  pivot_wider(names_from = stat, values_from = value)

print(summary_table)

outlier_table <- lapply(numeric_cols, function(col) {
  q1 <- quantile(df[[col]], 0.25, na.rm = TRUE)
  q3 <- quantile(df[[col]], 0.75, na.rm = TRUE)
  iqr_val <- q3 - q1
  outliers <- sum(
    df[[col]] < (q1 - 1.5 * iqr_val) |
      df[[col]] > (q3 + 1.5 * iqr_val),
    na.rm = TRUE
  )
  data.frame(variable = col, outliers = outliers)
}) %>%
  bind_rows()

print(outlier_table)

p1 <- ggplot(df, aes(date, sea_level)) +
  geom_line(color = "#2471A3", linewidth = 0.7) +
  geom_smooth(method = "loess", span = 0.3, color = "red", se = TRUE, alpha = 0.15) +
  labs(
    title = "Динамика уровня Каспийского моря (2001–2024)",
    x = "Дата",
    y = "Уровень моря, м"
  ) +
  theme_minimal(base_size = 13)
print(p1)

df_long <- df %>%
  pivot_longer(all_of(numeric_cols), names_to = "variable", values_to = "value") %>%
  mutate(variable = factor(
    variable,
    levels = c("sea_level", "volga", "precipitation", "evaporation", "temperature"),
    labels = c(
      "Уровень моря, м",
      "Сток Волги, м³/с",
      "Осадки, мм/мес",
      "Испарение, мм/мес",
      "Температура, °C"
    )
  ))

p2 <- ggplot(df_long, aes(date, value, color = variable)) +
  geom_line(linewidth = 0.6) +
  facet_wrap(~ variable, scales = "free_y", ncol = 1) +
  labs(title = "Динамика исследуемых показателей", x = "Дата", y = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none")
print(p2)

p3 <- ggplot(df, aes(month_ru, sea_level, fill = month_ru)) +
  geom_boxplot(alpha = 0.7, outlier.color = "red") +
  labs(
    title = "Уровень Каспийского моря по месяцам",
    x = "Месяц",
    y = "Уровень моря, м"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")
print(p3)

cor_data <- df %>% select(all_of(numeric_cols))
colnames(cor_data) <- c("Уровень моря", "Температура", "Осадки", "Волга", "Испарение")

cor_matrix <- cor(cor_data, use = "complete.obs", method = "pearson")
print(round(cor_matrix, 3))

corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  addCoef.col = "black",
  tl.col = "black",
  tl.srt = 45,
  number.cex = 0.9,
  title = "Корреляционная матрица",
  mar = c(0, 0, 2, 0)
)

ts_level <- ts(df$sea_level, start = c(2001, 1), frequency = 12)

stl_decomp <- stl(ts_level, s.window = "periodic")
plot(stl_decomp)
title(main = "STL-декомпозиция временного ряда")

seasonal_means <- tapply(stl_decomp$time.series[, "seasonal"], cycle(ts_level), mean)
names(seasonal_means) <- months_ru
print(round(seasonal_means, 2))
cat("Амплитуда сезонных колебаний:", round(max(seasonal_means) - min(seasonal_means), 2), "\n")

df_stl <- df %>%
  mutate(
    trend = as.numeric(stl_decomp$time.series[, "trend"]),
    seasonal = as.numeric(stl_decomp$time.series[, "seasonal"])
  )

p_trend <- ggplot(df_stl, aes(date)) +
  geom_line(aes(y = sea_level, color = "Исходный ряд"), alpha = 0.4, linewidth = 0.6) +
  geom_line(aes(y = trend, color = "Тренд"), linewidth = 1.2, na.rm = TRUE) +
  scale_color_manual(values = c("Исходный ряд" = "steelblue", "Тренд" = "red")) +
  labs(
    title = "Уровень Каспия: исходный ряд и тренд",
    x = "Дата",
    y = "Уровень моря, м",
    color = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top")
print(p_trend)

df_seas <- data.frame(month = 1:12, seasonal = as.numeric(seasonal_means))

p_seasonal <- ggplot(df_seas, aes(month, seasonal)) +
  geom_line(color = "darkgreen", linewidth = 1) +
  geom_point(color = "darkgreen", size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_continuous(breaks = 1:12, labels = months_ru) +
  labs(
    title = "Сезонная компонента уровня Каспия",
    x = "Месяц",
    y = "Отклонение от тренда, м"
  ) +
  theme_minimal(base_size = 13)
print(p_seasonal)

# -----------------------------
# Корреляции и лаги
# -----------------------------
cor_pearson <- cor(df[, numeric_cols], method = "pearson", use = "complete.obs")
cor_spearman <- cor(df[, numeric_cols], method = "spearman", use = "complete.obs")

print(round(cor_pearson, 3))
print(round(cor_spearman, 3))

for (fac in c("temperature", "precipitation", "volga", "evaporation")) {
  ct <- cor.test(df$sea_level, df[[fac]], method = "pearson")
  cat(sprintf("%-15s r = %6.3f | p = %.4f\n", fac, ct$estimate, ct$p.value))
}

lags <- c(1, 2, 3, 6, 12)
for (fac in c("volga", "evaporation", "precipitation")) {
  cat("\n", fac, "\n", sep = "")
  for (lag in lags) {
    r <- cor(df$sea_level, dplyr::lag(df[[fac]], lag), use = "complete.obs")
    cat(sprintf("lag %2d: %6.3f\n", lag, r))
  }
}

lag_results <- tidyr::crossing(
  var = c("volga", "evaporation", "precipitation"),
  lag = 0:24
) %>%
  mutate(
    correlation = purrr::map2_dbl(
      var, lag,
      ~ cor(df$sea_level, dplyr::lag(df[[.x]], .y), use = "complete.obs")
    ),
    var = dplyr::recode(
      var,
      volga = "Волга",
      evaporation = "Испарение",
      precipitation = "Осадки"
    )
  )

p_lags <- ggplot(lag_results, aes(lag, correlation, color = var)) +
  geom_line(linewidth = 0.9) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  facet_wrap(~ var, ncol = 1, scales = "free_y") +
  labs(
    title = "Лаговые корреляции с уровнем Каспия",
    x = "Лаг, месяцев",
    y = "Корреляция Пирсона"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")
print(p_lags)

par(mfrow = c(1, 2))
ccf(df$sea_level, df$volga,
    lag.max = 24,
    main = "Кросс-корреляция: уровень и Волга",
    ylab = "Корреляция", xlab = "Лаг")
ccf(df$sea_level, df$evaporation,
    lag.max = 24,
    main = "Кросс-корреляция: уровень и испарение",
    ylab = "Корреляция", xlab = "Лаг")
par(mfrow = c(1, 1))

adf_res <- adf.test(ts_level, alternative = "stationary")
kpss_res <- kpss.test(ts_level, null = "Level")

ts_diff1 <- diff(ts_level, differences = 1)
adf_d1 <- adf.test(ts_diff1, alternative = "stationary")
kpss_d1 <- kpss.test(ts_diff1, null = "Level")

print(adf_res)
print(kpss_res)
print(adf_d1)
print(kpss_d1)

d_value <- ifelse(adf_d1$p.value < 0.05 & kpss_d1$p.value > 0.05, 1, 2)
cat("d =", d_value, "\n")

par(mfrow = c(2, 2))
acf(ts_level, lag.max = 36, main = "ACF: исходный ряд")
pacf(ts_level, lag.max = 36, main = "PACF: исходный ряд")
acf(ts_diff1, lag.max = 36, main = "ACF: первые разности")
pacf(ts_diff1, lag.max = 36, main = "PACF: первые разности")
par(mfrow = c(1, 1))

df_model <- df %>%
  arrange(date) %>%
  mutate(
    volga_lag1 = dplyr::lag(volga, 1),
    evap_lag3 = dplyr::lag(evaporation, 3)
  ) %>%
  na.omit()

m1 <- lm(sea_level ~ temperature + precipitation + volga + evaporation, data = df_model)
m2 <- lm(sea_level ~ temperature + precipitation + volga_lag1 + evaporation, data = df_model)
m3 <- lm(sea_level ~ temperature + precipitation + volga_lag1 + evap_lag3, data = df_model)

model_compare <- data.frame(
  Model = c("m1", "m2", "m3"),
  R2 = c(summary(m1)$r.squared, summary(m2)$r.squared, summary(m3)$r.squared),
  Adj_R2 = c(summary(m1)$adj.r.squared, summary(m2)$adj.r.squared, summary(m3)$adj.r.squared),
  AIC = c(AIC(m1), AIC(m2), AIC(m3)),
  BIC = c(BIC(m1), BIC(m2), BIC(m3))
) %>%
  mutate(across(where(is.numeric), ~ round(., 4)))

print(summary(m2))
print(model_compare)

print(vif(m2))
print(bptest(m2))
print(dwtest(m2))
print(shapiro.test(residuals(m2)))

diag_df <- data.frame(
  fitted = fitted(m2),
  residuals = residuals(m2)
)

p_resid <- ggplot(diag_df, aes(fitted, residuals)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Остатки и прогнозные значения",
    x = "Прогнозные значения",
    y = "Остатки"
  ) +
  theme_minimal(base_size = 12)
print(p_resid)

qqnorm(residuals(m2), main = "Q-Q график остатков модели 2")
qqline(residuals(m2), col = "red", lwd = 2)


best_model <- auto.arima(
  ts_level,
  seasonal = TRUE,
  stepwise = FALSE,
  approximation = FALSE,
  trace = FALSE
)

print(best_model)
checkresiduals(best_model)

lb <- Box.test(
  residuals(best_model),
  lag = 24,
  type = "Ljung-Box",
  fitdf = length(coef(best_model))
)
print(lb)

fc_best <- forecast(best_model, h = 24)

forecast_df <- data.frame(
  date = seq(as.Date("2025-01-01"), by = "month", length.out = 24),
  forecast = as.numeric(fc_best$mean),
  lo80 = as.numeric(fc_best$lower[, 1]),
  hi80 = as.numeric(fc_best$upper[, 1]),
  lo95 = as.numeric(fc_best$lower[, 2]),
  hi95 = as.numeric(fc_best$upper[, 2])
)

print(forecast_df)

last_actual <- df %>%
  select(date, sea_level) %>%
  tail(36)

p_forecast <- ggplot() +
  geom_line(
    data = last_actual,
    aes(x = date, y = sea_level),
    color = "black",
    linewidth = 0.8
  ) +
  geom_ribbon(
    data = forecast_df,
    aes(x = date, ymin = lo95, ymax = hi95),
    fill = "lightblue",
    alpha = 0.25
  ) +
  geom_ribbon(
    data = forecast_df,
    aes(x = date, ymin = lo80, ymax = hi80),
    fill = "deepskyblue",
    alpha = 0.35
  ) +
  geom_line(
    data = forecast_df,
    aes(x = date, y = forecast),
    color = "red",
    linewidth = 1
  ) +
  geom_point(
    data = forecast_df,
    aes(x = date, y = forecast),
    color = "red",
    size = 1.8
  ) +
  geom_text(
    data = forecast_df %>% slice(seq(1, n(), by = 3)),
    aes(x = date, y = forecast, label = round(forecast, 2)),
    vjust = -0.8,
    size = 3
  ) +
  labs(
    title = "Прогноз уровня Каспийского моря",
    x = "Дата",
    y = "Уровень моря, м"
  ) +
  theme_minimal(base_size = 12)

print(p_forecast)

