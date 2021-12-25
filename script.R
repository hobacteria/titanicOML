TRAIN_PATH = "./input/titanic/train.csv"
TEST_PATH = "./input/titanic/test.csv"
SAMPLE_SUBMISSION_PATH = "submission.csv"

SUBMISSION_PATH = "submission.csv"

ID = 'PassengerId'
TARGET = "Survived"
MAX_RUNTIME_SECS = 600

TEST_SIZE = 0.2
TRAIN_SIZE = 0.8
THRESHOLD = 0.5

library(h2o)
library(data.table)

dtrain <- fread(TRAIN_PATH)
dtest <- fread(TEST_PATH)

dtrain[, weight := ifelse(TARGET == 1, 2, 1)]

# Train/test split
sample_index <- sample(c(TRUE, FALSE), size = nrow(dtrain), replace = TRUE, prob = c(TRAIN_SIZE, TEST_SIZE))
train <- dtrain[sample_index, !c('PassengerId')]
val <- dtrain[!sample_index, !c('PassengerId')]

h2o.init()

train <- as.h2o(train)
val <- as.h2o(val)

# test_ids <- as.integer(dtest[,ID])
dtest <- as.h2o(dtest)

y <- TARGET
x <- setdiff(names(train), y)

train[,y] <- as.factor(train[,y])
val[,y] <- as.factor(val[,y])

aml <- h2o.automl(x = x, y = y,
                  training_frame = train,
                  leaderboard_frame = val,
                  weights_column = "weight",
                  max_runtime_secs = MAX_RUNTIME_SECS)

lb <- aml@leaderboard
lb

# The leader model is stored here
aml@leader

pred <- as.data.frame(h2o.predict(aml@leader, dtest))
pred[1:5,]

pred_test <- ifelse(pred$p1>=THRESHOLD,1,0)
pred_test[1:5]

prediction<-read.csv(SAMPLE_SUBMISSION_PATH,na.strings = c("","NA"))
prediction[,TARGET]= as.integer(pred_test)

write.csv(prediction,SUBMISSION_PATH, row.names = FALSE)

head(prediction)
