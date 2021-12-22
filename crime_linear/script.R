# Predicting crime-linkage probabilities
# Andrew J. Graves
# 09/24/20

# Import packages
library(tidyverse)
library(caret)
library(splines)
library(doParallel)

# Set column types and read in the data
cols <- "ddddfffd"
train <- read_csv("data/linkage_train.csv", col_types = str_glue(cols, "d"))
test <- read_csv("data/linkage_test.csv", col_types = cols)

# Set repeated CV parameters
tr_ctrl <- trainControl(
  method = "repeatedcv", number = 10, repeats = 5,
  summaryFunction = mnLogLoss,
  classProbs = TRUE,
  savePredictions = TRUE
)

# Hyper parameter grid
tune_grid <- expand.grid(alpha = 1,
                         lambda = exp(seq(-12.5, -10.5, length = 400))
)

# Store scale of training data
nums <- train %>% 
  select(-y) %>%
  select_if(is.numeric)

# Numeric means
means <- nums %>%
  apply(2, mean)

# Numeric standard deviations
sds <- nums %>%
  apply(2, sd)

# Feature engineer the training data
train_final <- train %>%
  mutate(y = case_when(y == 1 ~ "link1",
                       TRUE ~ "link0") %>%
           factor()) %>%
  bind_cols(bs(train$spatial, degree = 5, df = 9)[, -c(6,8)],
            bs(train$temporal, degree = 8, df = 9)[, -c(2,8)],
            bs(train$tod, deg = 3, df = 4)[, -c(1,2)],
            bs(train$TIMERANGE, degree = 2, df = 6)[, -c(2,4,5)],
  ) %>%
  select(-spatial, -temporal, -tod, 
         -TIMERANGE, -MOA)

# Apply training scale to test
scaled_test <- test %>%
  select_if(is.numeric) %>%
  data.frame() %>%
  sweep(2, means, "-") %>% 
  sweep(2, sds, "/") %>%
  bind_cols(select_if(test, is.factor)) %>%
  select(-TIMERANGE, everything()) %>%
  mutate(y = NA)

# Feature engineer the test data with scale from training
test_final <- scaled_test %>%
  bind_cols(bs(scaled_test$spatial, degree = 5, df = 9)[, -c(6,8)],
            bs(scaled_test$temporal, degree = 8, df = 9)[, -c(2,8)],
            bs(scaled_test$tod, deg = 3, df = 4)[, -c(1,2)],
            bs(scaled_test$TIMERANGE, degree = 2, df = 6)[, -c(2,4,5)],
  ) %>%
  select(-spatial, -temporal, -tod, 
         -TIMERANGE, -MOA, -y)

# Set up parallel computing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Fit the glmnet model via caret
set.seed(42)
final_mod <- train(y ~ ., train_final,
                   method = "glmnet",
                   metric = "logLoss",
                   trControl = tr_ctrl,
                   tuneGrid = tune_grid,
                   allowParallel = TRUE)
# Terminate the parallel cluster
stopCluster(cl)

# Get probability predictions on test data
logloss_preds <- predict(final_mod, test_final, type = "prob")[, 2]

# Write probability predictions to file
prob_preds <- tibble("p" = logloss_preds)
write_csv(prob_preds, "graves_andrew.csv")