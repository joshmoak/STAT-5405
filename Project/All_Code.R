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


# 


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










