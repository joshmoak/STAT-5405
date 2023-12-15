# This is where we add all the code

# ---------------------------------------------------------------------------------------
# ----- SECTION 1: Add 5 new columns to our data set, derived from current columns. -----
# ---------------------------------------------------------------------------------------

df <- read.csv("chess_games.csv")
df <- df[-which(df["turns"] <= 1), ]

# Function to find white castle move
find_castle_move_white <- function(x) {
  chess_game <- x[11]
  moves <- unlist(strsplit(chess_game, " "))
  move_number <- 1
  white_castling_move <- NULL
  
  for (move in moves) {
    if (grepl("O-O|O-O-O", move)) {
      if (move_number %% 2 == 1) {
        white_castling_move <- ceiling(move_number / 2)
      }
    }
    move_number <- move_number + 1
  }
  if (is.null(white_castling_move)){
    return(0)
  }
  return(white_castling_move)
}

# Function to find black castle move
find_castle_move_black <- function(x) {
  chess_game <- x[11]
  moves <- unlist(strsplit(chess_game, " "))
  move_number <- 1
  black_castling_move <- NULL
  
  for (move in moves) {
    if (grepl("O-O|O-O-O", move)) {
      if (move_number %% 2 == 0) {
        black_castling_move <- (move_number / 2)
      }
    }
    move_number <- move_number + 1
  }
  if (is.null(black_castling_move)){
    return(0)
  }
  return(black_castling_move)
}

# Function to find white pawn push move
find_pawn_moves_white <- function(x) {
  chess_game <- x[11]
  moves <- unlist(strsplit(chess_game, " "))
  pawn_moves <- 0
  pattern <- "^[a-h][1-8x]"
  move_number <- 1
  
  for (move in moves) {
    if (grepl(pattern, move)) {
      if (move_number %% 2 == 1) {
        pawn_moves <- pawn_moves + 1
      }
    }
    move_number <- move_number + 1
  }
  return(pawn_moves)
}


# Function to find black pawn push move
find_pawn_moves_black <- function(x) {
  chess_game <- x[11]
  moves <- unlist(strsplit(chess_game, " "))
  pawn_moves <- 0
  pattern <- "^[a-h][1-8x]"
  move_number <- 1
  
  for (move in moves) {
    if (grepl(pattern, move)) {
      if (move_number %% 2 == 0) {
        pawn_moves <- pawn_moves + 1
      }
    }
    move_number <- move_number + 1
  }
  return(pawn_moves)
}

# Function to find game type
find_game_type <- function(x) {
  type_code <- x[6]
  
  moves <- unlist(strsplit(type_code, "+", fixed = TRUE))
  move <- as.numeric(moves[1])
  if (move <= 5){
    return("bullet")
  }
  else if (move <= 10){
    return("blitz")
  }
  else {return("rapid")}
  
}

df$white_castle <- apply(df, 1, FUN = find_castle_move_white)
df$black_castle <- apply(df, 1, FUN = find_castle_move_black)
df$white_pawn_moves <- apply(df, 1, FUN = find_pawn_moves_white)
df$black_pawn_moves <- apply(df, 1, FUN = find_pawn_moves_black)
df$game_type <- apply(df, 1, FUN = find_game_type)


# Create training and testing sets
set.seed(69420)
train.prop <- 0.9
trnset <- sort(sample(1:nrow(df), ceiling(nrow(df) * train.prop)))
# create the training and test sets
train <- df[trnset,]
test  <- df[-trnset,]

# ---------------------------------------------------------------------------------------
# ----- SECTION 2: GLIM Stuff -----
# ---------------------------------------------------------------------------------------

library(dplyr) 
library(ordinal) 
library(fastDummies)
library(car)
library(sure) 
library(MASS) 
library(nnet)
library(pROC)

# Ordinal Outcome Assumption

# Preproccessing-- One hot encoding for game type
ex.cat <- model.matrix(~ -1 + game_type, 
                       data = df)
df2 <- cbind(df, ex.cat)

# Eliminate unwanted columns, formatting
df2 = subset(df2, select = -c(white_id, black_id, game_id, opening_fullname, opening_response, opening_variation, moves, time_increment, opening_code, opening_shortname, victory_status, game_type) )
df2$rated <- as.numeric(df2$rated)
char_columns <- sapply
df2$winner <- factor(df2$winner, levels = c("White", "Draw", "Black"), labels = c(1, 2, 3))
df2 <- na.omit(df2)
df2.train <- df2[trnset,]
df2.test <- df2[-trnset,]

# Ensure split is adequate
table(df2.train$winner)/nrow(df2.train)
table(df2.test$winner)/nrow(df2.test)

# Multicollinearity check
pred.df <- subset(df2.train, select=-c(winner)) #data frame of predictors only
cor.pred <- cor(pred.df)
off.diag <- function(x) x[col(x) > row(x)]
v <- off.diag(cor.pred)
table(v >=0.95)

#Full model construction
full.model <- polr(formula = as.factor(winner) ~ rated + turns + game_typeblitz + game_typebullet + game_typerapid + black_rating + white_rating + 
                     black_castle + white_castle + black_pawn_moves + white_pawn_moves + 
                     opening_moves, data = df2.train)
full.model

# Null model construction
null.model <- polr(as.factor(winner) ~ 1, data = df2.train)
summary(null.model)

# Null model coefficients
(coef.table2 <- coef(summary(null.model)))
p <- pnorm(abs(coef.table2[, "t value"]), lower.tail = FALSE) * 2
(coef.table2 <- cbind(coef.table2, "p value" = p))

# Stepwise Model
vs.s <- polr(as.factor(winner) ~ 1, data = df2.train)
mod.s <- stepAIC(vs.s, scope = ~ rated + turns + white_rating + black_rating + opening_moves + white_castle + black_castle + white_pawn_moves + black_pawn_moves + game_typeblitz + game_typebullet + game_type_rapid, trace = FALSE,
                 direction = "both")
summary(mod.s)

# Stepwise Coefficients/Odds Ratios
coef.table2 <- coef(summary(mod.s))
p <- pnorm(abs(coef.table2[, "t value"]), lower.tail = FALSE) * 2
(coef.table2 <- cbind(coef.table2, "p value" = p))
exp(coef(mod.s))

# Predictions, classification table for test
df2.test$pred3 <- predict(mod.s, newdata = df2.test, type = "class") 
(ctable.pred3.test <- table(df2.test$pred3, df2.test$winner))
round((sum(diag(ctable.pred3.test))/sum(ctable.pred3.test))*100, 2) # accuracy 

# Predictions, classification for train
df2.train$pred3 <- predict(mod.s, newdata = df2.train, type = "class") 
ctable.pred3.train <- table(df2.train$pred3, df2.train$winner) # classification table
round((sum(diag(ctable.pred3.train))/sum(ctable.pred3.train))*100, 2) # accuracy 

# Nominal Outcome Assumption

# Removing the predictions from the test and train sets
df2.train = subset(df2.train, select = -c(pred3) )
df2.test = subset(df2.test, select = -c(pred3) )

# Fitting multinomial logit model
fit.gl <- multinom(as.factor(winner) ~ ., data = df2.train)
summary(fit.gl)

# Test data predictions and accuracy
df2.test$pred <- predict(fit.gl, newdata = df2.test, type = "class")
table <- cbind(df2.test$winner, df2.test$pred)
ctable.pred.test <- table(df2.test$winner, df2.test$pred) 
ctable.pred.test
round((sum(diag(ctable.pred.test))/sum(ctable.pred.test))*100, 2)

# Train data predictions and accuracy
df2.train$pred <- predict(fit.gl, newdata = df2.train, type = "class")
table <- cbind(df2.train$winner, df2.train$pred)
ctable.pred.train <- table(df2.train$winner, df2.train$pred) 
ctable.pred.train
round((sum(diag(ctable.pred.train))/sum(ctable.pred.train))*100, 2) #train accuracy

# Binary Outcome Assumption

# Deleting of drawn games, conversion of winner column to factor
df3 <- subset(df, winner != "Draw")
table(df3$winner)/nrow(df3)
df3$winner[df3$winner == "White"] <- 0
df3$winner[df3$winner == "Black"] <- 1
df3$winner <- as.factor(df3$winner)

# Preprocessing for df3
ex.cat <- model.matrix(~ -1 + game_type, 
                       data = df3)
df3 <- cbind(df3, ex.cat)
df3 = subset(df3, select = -c(white_id, black_id, game_id, opening_fullname, opening_response, opening_variation, moves, time_increment, opening_code, opening_shortname, victory_status, game_type) )
df3$rated <- as.numeric(df3$rated)

# Train/Test (Need new versions, since some rows were deleted)
set.seed(123467)
train.prop <- 0.90
strats <- df3$winner
df3$winner <- as.factor(df3$winner)
rr <- split(1:length(strats), strats)
idx <- sort(as.numeric(unlist(sapply(rr, 
                                     function(x) sample(x, length(x)*train.prop)))))
df3.train <- df3[idx, ]
df3.test <- df3[-idx, ]

# Ensure split is roughly equal
summary(df3.train$winner)/nrow(df3.train)
summary(df3.test$winner)/nrow(df3.test)

# Fit full binary logit model
full.logit <- glm(winner ~ . ,data = df3.train, 
                  family = binomial(link = "logit"))
summary(full.logit)

# Residual spread for full model
full.logit.res <- resid(full.logit, type = "deviance") 
summary(full.logit.res)

# Null model construction
null.logit <- glm(winner ~ 1, data = df3.train, 
                  family = binomial(link = "logit"))
summary(null.logit)

# Anova comparison of null and full model
(an.nb <- anova(null.logit, full.logit, test = "Chisq"))

# Stepwise model construction
both.logit <- step(null.logit, list(lower = formula(null.logit),
                                    upper = formula(full.logit)),
                   direction = "both", trace = 0, data = df3.train)
formula(both.logit)
summary(both.logit)

# Anova comparison of stepwise and full model
(an.nb <- anova(both.logit, full.logit, test = "Chisq"))

# Residual deviance comparison of all three binary logit models
null.logit$deviance
both.logit$deviance
full.logit$deviance

# Full model predictions, accuracy
pred.full <- predict(full.logit, newdata = df3.test, type = "response")
(table.full <- table(pred.full > 0.5, df3.test$winner))
(accuracy.full <- round((sum(diag(table.full))/sum(table.full))*100, 3))

# Full model ROC analysis
roc.full <- roc(df3.test$winner ~ pred.full)
plot.roc(roc.full, legacy.axes = TRUE, print.auc = TRUE)

# Both model predictions, accuracy
pred.both <- predict(both.logit, newdata = df3.test, type = "response")
(table.both <- table(pred.both > 0.5, df3.test$winner))
(accuracy.both <- round((sum(diag(table.both))/sum(table.both))*100, 3))

# Both model ROC analysis
roc.both <- roc(df3.test$winner ~ pred.both)
plot.roc(roc.both, legacy.axes = TRUE, print.auc = TRUE)

# ---------------------------------------------------------------------------------------
# ----- SECTION 3: XGBoost model -----
# ---------------------------------------------------------------------------------------


library(xgboost)
library(Matrix)
library(caret)

# One hot encode cat variables in each data set
dmy_train <- dummyVars(" ~ game_type + victory_status + rated", data = train, fullRank = TRUE)
dummy_df_train <- data.frame(predict(dmy_train, newdata = train))

dmy_test <- dummyVars(" ~ game_type + victory_status + rated", data = test, fullRank = TRUE)
dummy_df_test <- data.frame(predict(dmy_train, newdata = test))

train <- do.call(cbind, list(train, dummy_df_train))
test <- do.call(cbind, list(test, dummy_df_test))

# These are the relevant predictors
predictors <- c("turns", "white_rating", "black_rating", "opening_moves", "white_castle", "black_castle", 
                "white_pawn_moves", "black_pawn_moves", "game_typebullet", "game_typerapid", "victory_statusMate", 
                "victory_statusOut.of.Time", "victory_statusResign", "ratedTRUE")


# Turn `winner` feature into categorical
winner_cats <- function(df, i) {
  winner <- df["winner"][i,]
  
  if (winner == "White"){
    return(2)
  }
  else if (winner == "Black"){
    return(1)
  }
  else {return(0)}
}
train.winner.gbm <- c()
for (i in seq(1:nrow(train))) {  
  train.winner.gbm <- c(train.winner.gbm, winner_cats(train,i))
}
test.winner.gbm <- c()
for (i in seq(1:nrow(test))) {  
  test.winner.gbm <- c(test.winner.gbm, winner_cats(test,i))
}


#Next, we setup our train and test data-sets in the desired format.
# Train dataset
pred.train.gbm <- data.matrix(train[,predictors]) # predictors only
dtrain <- xgb.DMatrix(data = pred.train.gbm, label = train.winner.gbm)

# Test dataset
pred.test.gbm <- data.matrix(test[,predictors]) # predictors only
dtest <- xgb.DMatrix(data = pred.test.gbm, label = test.winner.gbm)

# Set up our parameters for xgboost
watchlist <- list(train = dtrain, test = dtest)
param <- list(max_depth = 5, eta = 1, nthread = 2, num_class = 3,
              objective = "multi:softmax", eval_metric = "merror")


# Train the model
model.xgb <- xgb.train(param, dtrain, nrounds = 50, watchlist, verbose = 0)

# Here is the confusion matrix and accuracy for our training set
pred.y.train <- predict(model.xgb, pred.train.gbm)
(tab<-table(train.winner.gbm, pred.y.train))
sum(diag(tab))/sum(tab)

# Confusion matrix and accuracy for testing set.
pred.y.test <- predict(model.xgb, pred.test.gbm)
(tab1<-table(test.winner.gbm, pred.y.test))
sum(diag(tab1))/sum(tab1)


imp <- xgb.importance(colnames(dtrain), model = model.xgb)

# Check proportions of winner in each category
t0 <- table(df[df$turns < mean(df[,"turns"]), ]["winner"])
t0 / sum(t0)
t1 <- table(df[df$turns >= mean(df[,"turns"]), ]["winner"])
t1 / sum(t1)

# test for significance
res1 <- prop.test(x = c(t0["Draw"], t1["Draw"]), n = c(sum(t0), sum(t1)), correct = FALSE)
res1$p.value
res2 <- prop.test(x = c(t0["Black"], t1["Black"]), n = c(sum(t0), sum(t1)), correct = FALSE)
res2$p.value
res3 <- prop.test(x = c(t0["White"], t1["White"]), n = c(sum(t0), sum(t1)), correct = FALSE)
res3$p.value










