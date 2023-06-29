install.packages("sparklyr")
install.packages("tidyverse")

library(sparklyr)
library(tidyr)
library(dplyr)
library(ggplot2)
library(magrittr)
library(knitr)

# Install
sparklyr::spark_install_tar(tarfile ="/install_tar/spark-3.4.0-bin-hadoop3.tgz")

# Connect
sc <- sparklyr::spark_connect(master = "local")

pathToDataset <- "~/workspace/1500000.csv"

df.unfiltered <- spark_read_csv(sc, path = pathToDataset,
                                header = FALSE,
                                infer_schema = FALSE,
                                columns = list(
                                  transaction_id = "character",
                                  price = "double",
                                  date = "timestamp",
                                  postcode = "character",
                                  property_type = "character",
                                  new_property = "character",
                                  duration = "character",
                                  PAON = "character",
                                  SAON = "character",
                                  street = "character",
                                  locality = "character",
                                  town = "character",
                                  district = "character",
                                  county = "character",
                                  ppd = "character",
                                  record_status = "character"
                                ))


df <- df.unfiltered %>% 
  filter(!is.na(transaction_id) & 
           !is.na(price) &
           !is.na(date) &
           !is.na(postcode) & 
           !is.na(property_type) & 
           !is.na(new_property) & 
           !is.na(duration) & 
           !is.na(street) & 
           !is.na(locality) &
           !is.na(town) & 
           !is.na(district) & 
           !is.na(county))
df
sparklyr::sdf_nrow(df)

price <- df %>% 
  select(price) %>%
  collect()

price.quantiles <- quantile(price$price)
price.low <- price.quantiles[[2]]
price.lowerMid <- price.quantiles[[3]]
price.upperMid <- price.quantiles[[4]]
price.high <- price.quantiles[[5]]

df <- df %>% 
  mutate(price_category = case_when(
    price < price.low ~ 0,
    price < price.lowerMid ~ 1,
    price < price.upperMid ~ 2,
    price < price.high ~ 3,
    TRUE ~ 3),
  )
df

df.split <- sdf_random_split(df, training = 0.8, testing = 0.2)

num.iterations <- c(3, 6, 10, 50, 100)
lr.performance.metrics.accuracy <- c(length(num.iterations))
lr.performance.metrics.weighted_precision <- c(length(num.iterations))
lr.performance.metrics.weighted_recall <- c(length(num.iterations))
lr.performance.metrics.weighted_f_measure <- c(length(num.iterations))
iterator <- 0

for (num in num.iterations) {
  iterator <- iterator + 1
  lr <- ml_logistic_regression(x = df.split$training,
                               formula = property_type ~ price + new_property + duration + district + county,
                               max_iter = num, family = "multinomial",
                               handle_invalid = "keep")
  results <- ml_evaluate(lr, df.split$testing)
  lr.performance.metrics.accuracy[iterator] <- results$accuracy()
  print(lr.performance.metrics.accuracy[iterator])
  lr.performance.metrics.weighted_precision[iterator] <- results$weighted_precision()
  lr.performance.metrics.weighted_recall[iterator] <- results$weighted_recall()
  lr.performance.metrics.weighted_f_measure[iterator] <- results$weighted_f_measure()
}

df <- data.frame(max_iterations = num.iterations, accuracy = lr.performance.metrics.accuracy, 
                 weighted_precision = lr.performance.metrics.weighted_precision,
                 weighted_recall = lr.performance.metrics.weighted_recall,
                 weighted_f_measure = lr.performance.metrics.weighted_f_measure)
df