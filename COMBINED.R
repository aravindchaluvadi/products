# Kishore Puppala
# to predict product A or B from merged data
######### start ##########
# clear memory
rm(list=ls(all=TRUE))
# set working directory
setwd('D:/DS')
# setwd("C:/DataScience/R")

# Read input csv files 
prodA <- read.csv("ProductA.csv",header = T)
prodB <- read.csv("ProductB.csv",header = T)

# add new attribute 'Type' to identify product A and B 
prodA$Type <- 'A'
prodB$Type <- 'B'

# merge both the csv files
mydata <- rbind(prodA,prodB)
str(mydata)
mydata$Type <- as.factor(mydata$Type)
# look for NA values
sum(is.na(mydata))

###### Install required packages ########
# install.packages("caretEnsemble")
library(caretEnsemble)
library(caret)
library(DMwR)

# use CARET package for creating train and test data
x<- createDataPartition(mydata$Type,times = 1,p=0.7,list=F)
train=mydata[x,]
test=mydata[-x,]

table(train$Type)
table(test$Type)
summary(train)

# PreProcess the data to standadize the numeric attributes
preProc<-preProcess(train[,setdiff(names(train),"Type")],method = c("center", "scale"))
train<-predict(preProc,train)
test<-predict(preProc,test)
summary(train)
str(train)

# convert class variable using make.names
levels(train$Type) <- make.names(levels(factor(train$Type)))
str(train)
levels(test$Type) <- make.names(levels(factor(test$Type)))
str(test)


###### Stacking algorithms #######
# create submodels
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
# algorithmList <- c('rpart', 'knn', 'svmRadial')
algorithmList <- c('rpart', 'C5.0','knn','svmRadial')
set.seed(12345)
# Build models
models <- caretList(Type~., data=train, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)

######### results of stacking using caret package ########
# Call:
#   summary.resamples(object = results)
# 
# Models: rpart, C5.0, knn, svmRadial 
# Number of resamples: 30 
# 
# Accuracy 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# rpart     0.8923077 0.9456044 0.9582418 0.9572063 0.9670330 0.9868132    0
# C5.0      0.9868132 0.9939536 0.9956044 0.9956765 0.9978022 1.0000000    0
# knn       0.9802198 0.9912088 0.9934066 0.9928925 0.9956020 1.0000000    0
# svmRadial 0.9890110 0.9955971 0.9978022 0.9964825 0.9978022 1.0000000    0
# 
# Kappa 
#                Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# rpart     0.6882647 0.8584415 0.8888179 0.8848546 0.9100860 0.9646803    0
# C5.0      0.9642530 0.9837344 0.9881202 0.9882847 0.9940601 1.0000000    0
# knn       0.9465412 0.9761687 0.9822274 0.9808286 0.9881359 1.0000000    0
# svmRadial 0.9701204 0.9880780 0.9940601 0.9904869 0.9940957 1.0000000    0



######## create stack using glm ######
library(caret)
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(1234)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)

########## predictions and evaluation using stack.glm #######
trainPredGLM <- predict(stack.glm, newdata = train, type = 'raw')
confusionMatrix(trainPredGLM, train$Type,positive="A")
testPredGLM <- predict(stack.glm, newdata = test, type = 'raw')
confusionMatrix(testPredGLM, test$Type,positive="A")


#################### ******  VARIOUS OTHER EXPERIMENTS  **** ###########

rm(list=ls(all=TRUE))

#read the data
data1 <- read.csv('ProductA.csv',header = T)
data2 <- read.csv('ProductB.csv',header = T)

#adding column for classification
#note if you are giving 'A' and 'B' as product categories in 
#place of 1 and 2 you might have NULL values as a whole column in 
#Tableau.
#choose the values with cross checking in Tableau.
data1$type <- rep(1, nrow(data1))
data2$type <- rep(2, nrow(data2))

#row bind the two data frames
data <- rbind.data.frame(data1,data2)

#convert to factors
data$type <- as.factor(data$type)
data$Target <- as.factor(data$Target)
str(data)

#writing the file
write.csv(data,file = 'blend.csv',row.names = FALSE)

#with the help of range, see if any visualisation can be created
summary(data)

########### ***********  MODELS ON THE COMBINED DATA SET ************* ############

## reading the data set 

blend<-read.csv("blend.csv",header = T)
str(blend)
summary(blend)

## converting the necessary data types

blend_cat<-blend[,c("Target","type")]
blend_num<-subset(blend,select = -c(Target,type))

blend_cat <- data.frame(apply(X = blend_cat, MARGIN = 2, FUN = as.factor))
str(blend_cat)

table(blend$type)

data_final<-cbind(blend_num,blend_cat)

table(data_final$type)
head(data_final)



#######################################  ***** ADA BOOSTING ***#####################################################################



## converting into dummies for ada boosting technique

dummies=dummyVars(type~.,data=data_final)

x.final=predict(dummies, newdata = data_final)
y.final=data_final$type


library(caret)
set.seed(125)
rows=createDataPartition(data_final$type,p = 0.7,list = FALSE)
train=data_final[rows,]
val=data_final[-rows,]


preProc<-preProcess(train[,setdiff(names(train),"type")],method = c("center", "scale"))
train<-predict(preProc,train)
val<-predict(preProc,val)

x.train=train$type
y.val=val$type
target=subset(train, select = -type)


a = subset(val, select = -type) 
b=val$type

library(ada) 

model = ada(target, x.train, iter=20, loss="logistic") # 20 Iterations 
model


pred = predict(model, a);pred 
result <- table(pred, b);result # 0(-ve) and 1(+ve)
accuracy <- sum(diag(result))/sum(result)*100;accuracy

confusionMatrix(result)

#########################  ******* DECISION TREES ****     ########################################


dtCart=rpart(type~., data=train, method="class")


### PLOTTING THE DECISION TREES ###

plot(dtCart,main="Classification Tree for loan Class",
     margin=0.15,uniform=TRUE)
text(dtCart,use.n=T)
summary(dtCart)

### plotting the decison trees.

library(rpart.plot)
rpart.plot(dtCart,fallen.leaves = T)


a=table(train$type, predict(dtCart, newdata=train, type="class"))

b=table(val$type, predict(dtCart, newdata=val, type="class"))


confusionMatrix(a)

confusionMatrix(b)


#######   ****    USING CP  ***** ############

dtCart_cp=rpart(type ~.,data=train,method="class", cp=0.01 )
printcp(dtCart)

plot(dtCart_cp,main="Classification Tree for loan Class",margin=0.15,uniform=TRUE)
text(dtCart_cp,use.n=T)
summary(dtCart_cp)

a_cp=table(train$type, predict(dtCart_cp, newdata=train, type="class"))
a_cp


b_cp=table(val$type, predict(dtCart_cp, newdata=val, type="class"))
b_cp


confusionMatrix(a_cp)

confusionMatrix(b_cp)


















