# Author: Andrew Graves
# Date: 4.27.19
# Goal: Minimize RMSE for IMLP Kaggle competition

# Load --------------------

# Packages

library(plyr)
library(stringr)
library(dplyr)
library(doParallel)
library(tm)
library(RWeka)
library(SentimentAnalysis)
library(textfeatures)
library(EGA)
library(NetworkToolbox)
library(caret)
library(xgboost)
library(mboost)

# Data

load("data/train.Rdata")
load("data/test.Rdata")

# Seed

set.seed(1)

# Functions

removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
removeNonWords <- function(x) gsub("[^a-zA-Z]+", " ", x)
n_gram <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 3))

# Combine train and test data for feature engineering

no_rating <- select(train.kaggle.final, -rating)
full_data <- rbind(no_rating, test.kaggle.final)

# Create vectors and lists

train_rows <- nrow(train.kaggle.final)
ID <- test.kaggle.final$ID
rating <- train.kaggle.final$rating
sparse_vec <- c(.994, .997)
predict_xgb <- list()
predict_glm_boost <- list()

# Feature Engineering --------------------

# Extract information from date column

year <- full_data$date %>%
          format("%Y") %>%
          as.numeric()

month <- full_data$date %>%
           format("%m") %>%
           as.numeric()

day <- full_data$date %>%
         format("%d") %>%
         as.numeric()

dow <- full_data$date %>%
         format("%A") %>%
         factor() %>%
         as.numeric()

full_data$date <- full_data$date %>%
          as.numeric()

# Create document term matrix from text column with maximum n-gram of 3

corpus <- full_data$text %>%
  VectorSource() %>%
  VCorpus() %>%
  tm_map(removeURL) %>%
  tm_map(removeNonWords) %>%
  tm_map(removeNumbers) %>%
  tm_map(tolower) %>%
  tm_map(PlainTextDocument) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(stemDocument)

dtm_one_gram <- DocumentTermMatrix(corpus)
dtm <- DocumentTermMatrix(corpus, control = list(tokenize = n_gram))

# Load sentiment and emotion analysis functionality

for(j in 1:length(corpus)){
  dtm$text.tm[j] <- strwrap(corpus[[j]], width = 10000)
}

liu.pos.words <- scan("data/positive-words.txt", what = "character", 
                      comment.char = ";", encoding = "UTF-8")
liu.neg.words <- scan("data/negative-words.txt", what = "character", 
                      comment.char = ";", encoding = "UTF-8")
liu.pos.scores <- rep(1, length(liu.pos.words))
liu.neg.scores <- rep(-1, length(liu.neg.words))
liu.lexicon <- c(liu.pos.words, liu.neg.words)
liu.scores <- c(liu.pos.scores, liu.neg.scores)

scoring.texts <- function(text, pos, neg) {
  scores <- ldply(text, function(text, pos, neg) {
    words0 <- str_split(text, '\\s+')
    words <- unlist(words0)
    positive <- sum(!is.na(match(words, pos)))
    negative <- sum(!is.na(match(words, neg)))
    score <- positive - negative
    all <- data.frame(score, positive, negative)
    return(all)
  }, pos, neg)
  scores.df = data.frame(scores, text=text)
  return(scores.df)
}

emotions <- read.csv("data/emotions.csv", header = TRUE, 
                     stringsAsFactors = FALSE)
load("data/emotions.texts.Rdata")

# Sentiment analysis

scores <- scoring.texts(text = dtm$text.tm, pos = liu.pos.words, 
                        neg = liu.neg.words) %>%
            select(-text)
sentiments <- analyzeSentiment(dtm_one_gram)

# Emotion analysis

emotion_scores <- emotions.texts(text = dtm$text.tm, emotions = emotions) %>%
                    select(-text)

# Recode missing values to 0

emotion_scores[is.na(emotion_scores)] <- 0

# Feature extraction using textfeatures

text_feat <- textfeatures(full_data$text, normalize = TRUE) %>%
              select(-n_hashtags, -n_uq_hashtags)

# Iterate predictive modeling by raising sparsity threshold of n-grams in 
# feature space ----------

for(i in 1:length(sparse_vec)){

n_grams <- removeSparseTerms(dtm, sparse_vec[i]) %>% 
  as.matrix() %>%
  as.data.frame()

paste0("This is iteration #", as.character(i), ": There are ", 
       as.character(ncol(n_grams)), " n-grams for this round.") %>%
  print()

# Cluster data to generate new features -----

# Construct data frame for EGA

train_data_ega <- data.frame(select(full_data, -ID, -text), year, month, day, 
                            dow, n_grams, scores, emotion_scores, sentiments, 
                            text_feat)

# Run EGA

cor_matrix <- cor(train_data_ega)
ega <- EGA(cor_matrix, model = "TMFG", n = nrow(train_data_ega))

# Network analysis on EGA

network_scores <- nams(train_data_ega, A = ega$network, 
                       comm = ega$wc, standardize = TRUE)

# Append network scores and scale all data

train_data <- data.frame(train_data_ega, network_scores$Standardized) %>%
                 scale()

# Apply PCA to combine information across features

pca <- prcomp(train_data, center = TRUE, scale. = TRUE)
cor_pca <- cor(pca$x)

# Apply EGA and network analysis to PCA output to extract community 
# information in a reduced feature space

ega_pca <- EGA(cor_pca, model = "TMFG", n = nrow(pca$x))
network_pca <- nams(pca$x, A = ega_pca$network, comm = ega_pca$wc, 
                    standardize = TRUE)
train_data <- data.frame(train_data, network_pca$Standardized)

# Split/save training/test data -----

test_data <- data.frame(train_data[-(1:train_rows),])
train_model <- data.frame(train_data[1:train_rows,], rating)
save(train_model, file = paste0("train_model_", as.character(i)))
save(test_data, file = paste0("test_data_", as.character(i)))

# Predictive modeling --------------------

# XGB -----

# Set parameters

params <- list(eta = 0.001, max_depth = 6, subsample = 0.5,
               colsample_bytree = 0.5, seed = 1, nthread = detectCores() - 1, 
               gamma = 0, silent = 0)

# Cross validate to select number of trees for XGB

xgb_cv <- xgb.cv(data = data.matrix(select(train_model, -rating)), 
                 label = rating, params = params, nrounds = 100000, nfold = 5, 
                 verbose = TRUE, early_stopping_rounds = 1000)
best_tree_limit <- xgb_cv$best_ntreelimit
save(xgb_cv, file = paste0("xgb_cv_model_", as.character(i)))

# Run XGB

xgb <- xgboost(data = data.matrix(select(train_model, -rating)), 
                  label = rating, nrounds = best_tree_limit, params = params)
save(xgb, file = paste0("xgb_model_", as.character(i)))

# Store XGB predictions in list

predict_xgb[[i]] <- predict(xgb, data.matrix(test_data), 
                            n.trees = best_tree_limit)
save(predict_xgb, file = paste0("xgb_predict_", as.character(i)))

# GLM boost -----

# Parallel processing

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Set cross-validation design

fit_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, 
                            allowParallel = TRUE, savePrediction = "final")

# Grid parameters for cross-validation

glm_boost_grid <- expand.grid(mstop = seq(2300, 2500, 25), prune = "no")

# Run GLM boost

set.seed(1)
glm_boost <- train(rating~., data = train_model, trControl = fit_control,
                      method = "glmboost", tuneGrid = glm_boost_grid)
stopCluster(cl)
save(glm_boost, file = paste0("glmboost_model_", as.character(i)))

# Store GLM boost predictions in list

predict_glm_boost[[i]] <- predict(glm_boost, test_data) %>%
                            as.vector()
save(predict_glm_boost, file = paste0("glmboost_predict_", as.character(i)))

} # End predictive modeling

# Create prediction submission -----

# Explore correlations across the predictions from all models

pred_data <- c(predict_xgb, predict_glm_boost)
cor_data <- do.call(cbind, pred_data)
cor(cor_data)

# Create prediction vector

Prediction1 <- apply(cor_data, MARGIN = 1, FUN = mean)

old_prediction <- read.csv("graves_predictions.csv")
old_vs_new <- cor(old_prediction$Prediction1, Prediction1)

# This if statement ensures new prediction is different from previous prediction

if(old_vs_new < 1){
 
  # Create prediction data frame
  
  prediction_data <- data.frame(ID, Prediction1) 
  
  # Write prediction to file
  
  write.csv(prediction_data, "graves_predictions.csv", row.names = FALSE) 
  print("New prediction is ready for submission.") 
  
} else { 
  
  print("Did not rename/write new prediction.")
  
}