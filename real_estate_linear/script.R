# Predicting real-estate prices with penalized linear regression
# Andrew J. Graves
# 09/17/20

# Import packages
library(tidyverse)
library(glmnet)
library(caret)
library(doParallel)

# Convenient MSE function
mse <- function(pred, obs) mean((pred - obs)^2)

# Read in training data and test data
train_read <- read_csv("data/realestate-train.csv")
test_read <- read_csv("data/realestate-test.csv")

# Feature engineer training and test set
all_dat <- train_read %>%
  select(-price) %>%
  bind_rows(test_read, .id = "set") %>%
  # Make factors (treatment coding)
  mutate_if(is_character, factor) %>%
  
  mutate(# Transformations guided by Box-Cox
    Age = sqrt(Age),
    LotSize = log(LotSize),
    Baths = sqrt(Baths),
    # Decrease # of categories sorted by average group price
    HouseStyle = case_when(
      HouseStyle %in% c("1.5Unf", "2.5Unf", "1.5Fin", "SFoyer") ~ "Low",
      HouseStyle == "1Story" ~ "Medium",
      TRUE ~ "High") %>%
      factor()
  ) %>%
  # Remove low variance feature
  select(-PoolArea)

# Separate and expand polynomials for training set
train_re <- all_dat %>%
  filter(set == "1") %>%
  bind_cols(poly(train_read$SqFeet, degree = 4)[, -c(1, 2)],
            poly(train_read$Age, degree = 2)[, -1]
  ) %>%
  bind_cols("price" = train_read$price) %>%
  select(-set)

# Separate and expand polynomials for test set
test_re <- all_dat %>%
  filter(set == "2") %>%
  bind_cols(poly(test_read$SqFeet, degree = 4)[, -c(1, 2)],
            poly(test_read$Age, degree = 2)[, -1]
  ) %>%
  select(-set)

# Set repeated CV parameters
tr_ctrl <- trainControl(
  method = "repeatedcv", number = 10, repeats = 5
)

# Hyper parameter grid
tune_grid <- expand.grid(lambda = exp(seq(-3, 3, length = 1000)),
                         alpha = seq(.8, .9, length = 10)
)

# Set up parallel computing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Fit the glmnet model via caret with three-way interactions
set.seed(420)
best_mod <- train(price ~ .^3, train_re,
                  method = "glmnet",
                  trControl = tr_ctrl,
                  tuneGrid = tune_grid,
                  allowParallel = TRUE)
# Terminate the parallel cluster
stopCluster(cl)

# Anticipated RMSE
pred_rmse <- getTrainPerf(best_mod)$TrainRMSE
# Make predictions (scale negative price to 0)
final_preds <- data.frame("yhat" = predict(best_mod, test_re)) %>%
  mutate(yhat = if_else(yhat < 0, 0, yhat))
write_csv(final_preds, "graves_andrew.csv")