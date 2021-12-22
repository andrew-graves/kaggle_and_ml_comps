# Predicting real-estate prices with non-linear methods
# Andrew J. Graves
# 11/03/20

# Import packages
library(tidyverse)
library(doParallel)
library(xgboost)
library(catboost)
library(caret)

# Convenient one-hot encoding function
one_hot_encode <- function(x){
  x %>%
    data.table::data.table(stringsAsFactors = TRUE) %>%
    mltools::one_hot() %>%
    as.matrix() %>%
    return()
}

# Read in the data
train_read <- read_csv("data/train.csv")

test_read <- read_csv("data/test.csv")

# Clean and feature engineer the data (categories)
new_vars <- train_read %>%
  bind_rows(test_read, .id = "set") %>%
  select(-Id, -Utilities, -MiscFeature, -PoolQC, -PoolArea, -LandSlope) %>%
  mutate(# Log the outcome
    SalePrice = log(SalePrice),
    # Categorical features
    SaleCondition = case_when(SaleCondition == "Partial" ~ "hi",
                              SaleCondition == "Normal" ~ "med",
                              TRUE ~ "lo"),
    SaleType = case_when(SaleType %in% c("CWD", "New", "Con") ~ "hi",
                         SaleType == "WD" ~ "med",
                         TRUE ~ "lo"),
    MiscVal = case_when(MiscVal > 0 ~ 1,
                        TRUE ~ 0),
    Fence = case_when(Fence %in% c("GdWo", "MnWw", "MnPrv") ~ "lo",
                      TRUE ~ "hi"),
    GarageCond = case_when(GarageCond %in% c("Ex", "Gd", "TA") ~ "hi",
                           TRUE ~ "lo"),
    GarageQual = case_when(GarageQual %in% c("Ex", "Gd", "TA") ~ "hi",
                           TRUE ~ "lo"),
    GarageType = case_when(GarageType == "BuiltIn" ~ "hi",
                           GarageType == "Attchd" ~ "medhi",
                           GarageType %in% c("Detchd", "2Types", "Basment") ~ "med",
                           TRUE ~ "lo"),
    FireplaceQu = replace_na(FireplaceQu, "Po"),
    Functional = case_when(Functional == "Typ" ~ "hi",
                           Functional %in% c("Sev", "Min2") ~ "med",
                           Functional == "Maj2" ~ "lo",
                           TRUE ~ "medhi"),
    Electrical = case_when(
      Electrical %in% c("Mix", "FuseP", "FuseF") ~ "lo",
      Electrical == "FuseA" ~ "med",
      TRUE ~ "hi"),
    HeatingQC = str_replace(HeatingQC, "Po", "Fa"),
    Heating = case_when(Heating %in% c("GasA", "GasW") ~ "hi",
                        TRUE ~ "lo"),
    BsmtFinType2 = case_when(
      BsmtFinType2 %in% c("ALQ", "Unf", "GLQ") ~ "hi",
      BsmtFinType2 %in% c("LwQ", "Rec", "BLQ") ~ "med",
      TRUE ~ "lo"),
    BsmtFinType1 = case_when(BsmtFinType1 == "GLQ" ~ "hi",
                             BsmtFinType1 %in% c("ALQ", "Unf") ~ "medhi",
                             BsmtFinType1 %in% c("LwQ", "Rec", "BLQ") ~ "med",
                             TRUE ~ "lo"),
    BsmtCond = str_replace(replace_na(BsmtCond, "Fa"), "Po", "Fa"),
    BsmtQual = replace_na(BsmtQual, "none"),
    Foundation = recode(Foundation, Wood = "PConc", Stone = "CBlock"),
    ExterCond = case_when(ExterCond %in% c("Po", "Fa") ~ "lo",
                          TRUE ~ "hi"),
    Exterior2nd = case_when(
      Exterior2nd %in% c("CBlock", "AsbShng", "Brk Cmn", 
                         "AsphShn", "WdSdng") ~ "lo",
      Exterior2nd %in% c("Stucco", "MetalSd", "WdShng", "Stone") ~ "med",
      Exterior2nd %in% c("HdBoard", "Plywood") ~ "medhi",
      TRUE ~ "hi"),
    Exterior1st = case_when(
      Exterior1st %in% c("CBlock", "AsbShng", 
                         "BrkComm", "AsphShn") ~ "lo",
      Exterior1st %in% c("Wd Sdng", "WdShing", 
                         "Stucco", "MetalSd") ~ "med",
      Exterior1st %in% c("HdBoard", "Plywood") ~ "hi",
      TRUE ~ "4"),
    RoofMatl = case_when(
      RoofMatl %in% c("WdShake", "Membran", "WdShngl") ~ "hi",
      TRUE ~ "lo"),
    RoofStyle = case_when(RoofStyle %in% c("Flat", "Hip", "Shed") ~ "hi",
                          TRUE ~ "lo"),
    HouseStyle = case_when(
      HouseStyle %in% c("1.5Unf", "1.5Fin", "SFoyer") ~ "lo",
      HouseStyle %in%  c("1Story", "2.5Unf", "SLvl") ~ "med",
      TRUE ~ "hi"),
    BldgType = recode(BldgType, `2fmCon` = "Duplex", Twnhs = "Duplex"),
    Condition2 = case_when(
      Condition2 %in% c("PosA", "PosN") ~ "hi",
      Condition2 %in% c("Norm", "RRAn", "RRAe") ~ "med",
      TRUE ~ "lo"),
    Condition1 = case_when(
      Condition1 %in% c("Artery", "Feedr", "RRAe") ~ "lo",
      Condition1 == "Norm" ~ "med",
      TRUE ~ "hi"),
    Neighborhood = case_when(
      Neighborhood %in% c("IDOTRR", "MeadowV", "BrDale") ~ "vlo",
      Neighborhood %in% c("BrkSide", "OldTown", "Edwards") ~ "lo",
      Neighborhood %in% c("Sawyer", "Blueste", "SWISU") ~ "medlo",
      Neighborhood %in% c("NPkVill", "NAmes", "Mitchel") ~ "med",
      Neighborhood %in% c("SawyerW", "NWAmes", "Gilbert") ~ "medhi",
      Neighborhood %in% c("CollgCr", "Blmngtn", 
                          "Crawfor", "ClearCr") ~ "hi",
      Neighborhood %in% c("Somerst", "Veenker", "Timber") ~ "vhi",
      Neighborhood %in% c("StoneBr", "NridgHt") ~ "vvhi",
      TRUE ~ "vvvhi"),
    LotConfig = case_when(LotConfig %in% c("CulDSac", "FR3") ~ "hi",
                          TRUE ~ "lo"),
    LotShape = recode(LotShape, IR3 = "IR1"),
    Alley = replace_na(Alley, "Pave")) %>%
  mutate_if(is.character, factor)

# Split the data into train and test
train_split <- new_vars %>%
  filter(set == "1") %>%
  select(-set)

test_split <- new_vars %>%
  filter(set == "2") %>%
  select(-set, -SalePrice)

# Extract outcome vector
y <- train_split$SalePrice

# Impute missing values separately on train and test
train_dat <- train_split %>%
  select(-SalePrice) %>%
  VIM::kNN(imp_var = FALSE, k = 1)

test_dat <- test_split %>%
  VIM::kNN(imp_var = FALSE, k = 1)

# Set xgb parameters
xgb_params <- list(
  eta = .001, max_depth = 5, subsample = .5,
  colsample_bytree = 1/3, 
  nthread = detectCores() - 1, 
  gamma = .01
)

# Set catboost parameters
cat_iters <- 45000
metric_cycle <- 1500

cat_params <- list(
  loss_function = "RMSE", learning_rate = .004,
  iterations = cat_iters, metric_period = metric_cycle, 
  depth = 2, rsm = .5, l2_leaf_reg = .001,
  bootstrap_type = "Bernoulli", subsample = .5,
  border_count = 254
)

# Set caret.catboost parameters
caret_grid <- expand.grid(depth = 2, learning_rate = .075,
                          iterations = c(3000, 4000), 
                          l2_leaf_reg = 5e-4,
                          rsm = .95, 
                          border_count = 254)

# Set up caret cross-validation scheme
tr_ctrl <- trainControl(method = "repeatedcv", number = 5,
                        repeats = 4)

# Initialize prediction output
preds <- list()

# Select seed
set.seed(999)

# Train the models

# XGBoost
# ----------
# Get train and test xgb matrices
xgb_train_mat <- train_dat %>%
  one_hot_encode() %>%
  xgb.DMatrix(label = y)

xgb_test_mat <- test_dat %>%
  one_hot_encode() %>%
  xgb.DMatrix()

# Cross-validate the number of trees for xgb
xgb_cv <- xgb.cv(data = xgb_train_mat, params = xgb_params, 
                 nrounds = 80000, nfold = 5, print_every_n = metric_cycle,
                 verbose = TRUE, early_stopping_rounds = metric_cycle)

# Get best number of trees
best_xgb <- xgb_cv$best_ntreelimit

# Fit the xgb model
xgb_mod <- xgboost(data = xgb_train_mat,
                   nrounds = best_xgb, verbose = FALSE,
                   params = xgb_params)

# Make predictions with xgb model
pred_xgb <- predict(xgb_mod, xgb_test_mat, best_xgb)
write_csv(data.frame(pred_xgb), "pred_xgb.csv")
# ----------

# Catboost 
# ----------
# Specify categorical features
cat_feats <- train_dat %>%
  map_lgl(is.factor) %>%
  which() %>%
  unname()

# Load train and test pool
train_pool <- catboost.load_pool(data = train_dat, label = y, 
                                 cat_features = cat_feats - 1)

test_pool <- catboost.load_pool(data = test_dat, 
                                cat_features = cat_feats - 1)

# Cross-validate the number of trees for catboost
cat_cv <- catboost.cv(train_pool, fold_count = 2, params = cat_params)

# Get optimal iteration length
best_iters <- which.min(cat_cv$test.RMSE.mean) - 1
cat_params$iterations <- best_iters * metric_cycle

# Fit the catboost model
cat_mod <- catboost.train(train_pool, params = cat_params)

# Make predictions with catboost model
pred_cat <- catboost.predict(cat_mod, test_pool)
write_csv(data.frame(pred_cat), "pred_cat.csv")
# ---------

# Catboost.caret
# ----------
# Set up parallel computing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Fit the caret.catboost model
caret_mod <- train(train_dat, y,
                   method = catboost.caret,
                   metric = "RMSE",
                   tuneGrid = caret_grid, 
                   trControl = tr_ctrl,
                   metric_period = metric_cycle)
# Print summary of results
print(caret_mod)

# Terminate the parallel cluster
stopCluster(cl)

# Make predictions with caret.catboost
pred_caret <- predict(caret_mod, test_dat)
write_csv(data.frame(pred_caret), "pred_caret.csv")
# ----------

preds <- tibble(pred_xgb, pred_cat, pred_caret)

# 50% XGB: 25% for each catboost version
wts <- c(2, 1, 1)

pred_test <- preds %>%
  map_dfc(bind_cols) %>%
  apply(1, weighted.mean, wts / sum(wts)) %>%
  exp()

pred_df <- data.frame("Id" = test_read$Id, "SalePrice" = pred_test)
write_csv(pred_df, "graves_preds.csv")