install.packages("sparklyr")
install.packages("tidyverse")
install.packages("cowplot")

library(sparklyr)
library(tidyverse)
library(tidyr)
library(dplyr)
library(ggplot2)
library(cowplot)
library(magrittr)
library(knitr)

# Install
sparklyr::spark_install_tar(tarfile ="/install_tar/spark-3.4.0-bin-hadoop3.tgz")

# Connect
sc <- sparklyr::spark_connect(master = "local")

pathToDataset <- "~/workspace/500000.csv"

# Load dataset
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


# Data cleansing and transformation
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

# Label encoding using StringIndexer
df <- df %>%
  ft_string_indexer(input_col = "county", output_col = "county_encoded")
df <- df %>%
  ft_one_hot_encoder("county_encoded", "county_encoded2")

df <- df %>%
  ft_string_indexer(input_col = "district", output_col = "district_encoded")
df <- df %>%
  ft_one_hot_encoder("district_encoded", "district_encoded2")
df


# 1.1. Classification with different parameters

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
                               formula = property_type ~ price + new_property + duration + district_encoded2 + county_encoded2,
                               max_iter = num, family = "multinomial",
                               handle_invalid = "keep")
  results <- ml_evaluate(lr, df.split$testing)
  lr.performance.metrics.accuracy[iterator] <- results$accuracy()
  print(lr.performance.metrics.accuracy[iterator])
  lr.performance.metrics.weighted_precision[iterator] <- results$weighted_precision()
  lr.performance.metrics.weighted_recall[iterator] <- results$weighted_recall()
  lr.performance.metrics.weighted_f_measure[iterator] <- results$weighted_f_measure()
}

lr.results.table <- data.frame(max_iterations = num.iterations, accuracy = lr.performance.metrics.accuracy, 
                 weighted_precision = lr.performance.metrics.weighted_precision,
                 weighted_recall = lr.performance.metrics.weighted_recall,
                 weighted_f_measure = lr.performance.metrics.weighted_f_measure)
lr.results.table


# 1.2. Visualization

lr.acc.plot <- lr.results.table %>% 
  ggplot(aes(max_iterations, accuracy)) +
  geom_line() +
  geom_point()

lr.prec.plot <- lr.results.table %>% 
  ggplot(aes(max_iterations, weighted_precision)) +
  geom_line() +
  geom_point()

lr.rec.plot <- lr.results.table %>% 
  ggplot(aes(max_iterations, weighted_recall)) +
  geom_line() +
  geom_point()

lr.f.plot <- lr.results.table %>% 
  ggplot(aes(max_iterations, weighted_f_measure)) +
  geom_line() +
  geom_point()

plot_grid(lr.acc.plot, lr.prec.plot, lr.rec.plot, lr.f.plot, nrow=2, ncol=2)


# 2.1. Classification with different model types

# number of folds
k = 5

# create the weights
weights <- rep(1 / k, times = k)

names(weights) <- paste0("fold", 1:k)

# create the partitioned data
df.models.partitioned <- sdf_partition(
  df, 
  weights = weights, 
  seed = 2023
)

models.formula <- property_type ~ price + new_property + duration + district_encoded2 + county_encoded2

lr.acc <- c()
random_forest.acc <- c()
decision_tree.acc <- c()

for (i in 1:k) {
  training.set <- sdf_bind_rows(df.models.partitioned[-i])
  test.set <- df.models.partitioned[[i]]
  
  lr <- ml_logistic_regression(x = training.set,
                               formula = models.formula,
                               max_iter = 50, family = "multinomial",
                               handle_invalid = "keep")
  lr.results <- ml_evaluate(lr, test.set)
  lr.acc[i] <- lr.results$accuracy()
  print(lr.acc[i])
  
  random_forest <- ml_random_forest_classifier(x = training.set,
                               formula = models.formula,
                               max_depth = 5,
                               handle_invalid = "keep")
  random_forest.results <- ml_evaluate(random_forest, test.set)
  random_forest.acc[i] <- random_forest.results$Accuracy
  print(random_forest.acc[i])
  
  decision_tree <- ml_decision_tree_classifier(x = training.set,
                                               formula = models.formula,
                                               max_depth = 5,
                                               handle_invalid = "keep")
  decision_tree.results <- ml_evaluate(decision_tree, test.set)
  decision_tree.acc[i] <- decision_tree.results$Accuracy
  print(decision_tree.acc[i])
}

print(mean(lr.acc))
print(mean(random_forest.acc))
print(mean(decision_tree.acc))