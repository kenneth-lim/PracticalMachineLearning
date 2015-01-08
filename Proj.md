# Practical Machine Learning Course Project

The objective of this project is to devise a machine learning algorithm to determine the way dumbbell lifts are being performed by participants. There are five possible specifications of dumbbell lifts (some incorrect), and data comes from accelerometers on the belt, forearm, arm and dumbbell of 6 participants. 

Analysis starts from having the training and testing datasets in my folder. As the dataset is large (N = 19622), and the testing dataset is separately provided, I should divide the training data into 75% training and 25% validation (rule of thumb: 60% training, 20% testing, 20% cross validation). 



## Preliminaries

Load all the packages and read the data:


```r
df <- read.csv("pml-training.csv", na.strings=c("NA", "", "#DIV/0!")) # Read training data into R
library(AppliedPredictiveModeling)
library(caret)
library(randomForest)
library(rattle)
library(rpart)
library(gbm)
set.seed(123) # for reproducibility
```

Scanning the data suggests many potential predictors with many missing values. I want omit these to avoid problems associated with the treatment of observations with missing values by R.



```r
inTrain <- createDataPartition(df[,"classe"], p = .75)[[1]] # assign 75 pct of obs to training dataset
training <- df[inTrain,]
training <- training[,complete.cases(t(training))] # 60 var left
training <- training[,-1] # 59
nzv <- nearZeroVar(training, saveMetrics=TRUE) # new_window also has nzv
training <- training[,-5] # remove new_window
crossval <- df[-inTrain,]
crossval <- crossval[,complete.cases(t(crossval))] 
crossval <- crossval[,-1] 
crossval <- crossval[,-5]
```

## Prediction Model

I fit a generalised boosted regression model in an attempt to obtain stronger predictors for better accuracy. Using boosting was also motivated by other methods being very very slow. 


```r
modFit <- train(classe ~ ., data=training, method="gbm", distribution="multinomial", verbose=FALSE)
```

```
## Loading required package: plyr
```

```r
print(modFit)
```

```
## Stochastic Gradient Boosting 
## 
## 14718 samples
##    57 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ... 
## 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD
##   1                   50      0.8400323  0.7969232  0.007020557
##   1                  100      0.8996246  0.8728138  0.005501221
##   1                  150      0.9258514  0.9060312  0.003874767
##   2                   50      0.9556949  0.9438850  0.003179411
##   2                  100      0.9849203  0.9809202  0.002035509
##   2                  150      0.9912392  0.9889156  0.001447806
##   3                   50      0.9814872  0.9765750  0.002434953
##   3                  100      0.9928249  0.9909221  0.001315622
##   3                  150      0.9956562  0.9945046  0.001145579
##   Kappa SD   
##   0.008798731
##   0.006904300
##   0.004873880
##   0.004006949
##   0.002570501
##   0.001831816
##   0.003060968
##   0.001663500
##   0.001447861
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3 and shrinkage = 0.1.
```

## Check for model's accuracy on training set


```r
predicttrain <- predict(modFit,training)
confusionMatrix(training$classe, predicttrain)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4184    1    0    0    0
##          B    1 2843    2    2    0
##          C    0    1 2559    7    0
##          D    0    0    2 2406    4
##          E    0    0    0    1 2705
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9986          
##                  95% CI : (0.9978, 0.9991)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9982          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9998   0.9993   0.9984   0.9959   0.9985
## Specificity            0.9999   0.9996   0.9993   0.9995   0.9999
## Pos Pred Value         0.9998   0.9982   0.9969   0.9975   0.9996
## Neg Pred Value         0.9999   0.9998   0.9997   0.9992   0.9997
## Prevalence             0.2843   0.1933   0.1741   0.1642   0.1841
## Detection Rate         0.2843   0.1932   0.1739   0.1635   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      0.9998   0.9994   0.9989   0.9977   0.9992
```

Prediction appears to be very accurate. 99.9% of observations were correctly classified, suggesting a very low in sample error rate of about 0.1%.

## Estimate model's out of sample error rate on cross validation set

Now on the cross-validation set.


```r
predictcv <- predict(modFit, crossval)
confusionMatrix(crossval$classe, predictcv)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    0  948    1    0    0
##          C    0    1  853    1    0
##          D    0    0    2  801    1
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9988          
##                  95% CI : (0.9973, 0.9996)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9985          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9989   0.9965   0.9988   0.9989
## Specificity            1.0000   0.9997   0.9995   0.9993   1.0000
## Pos Pred Value         1.0000   0.9989   0.9977   0.9963   1.0000
## Neg Pred Value         1.0000   0.9997   0.9993   0.9998   0.9998
## Prevalence             0.2845   0.1935   0.1746   0.1635   0.1839
## Detection Rate         0.2845   0.1933   0.1739   0.1633   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      1.0000   0.9993   0.9980   0.9990   0.9994
```

Again, prediction appears to be accurate; 99.9% of observations were correctly classified. Based on the cross-validation sample, I expect the out of sample error rate to be 0.1%.

## Submit test set


```r
test <- read.csv("pml-testing.csv", na.strings=c("NA", "", "#DIV/0!"))
test <- test[,complete.cases(t(test))] 
test <- test[,-1]
test <- test[,-5]
predicttest <- predict(modFit, test)
predicttest <- as.character(predicttest)
answers <- predicttest
print(answers)
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

```r
#pml_write_files = function(x){
#  n = length(x)
#  for(i in 1:n){
#    filename = paste0("problem_id_",i,".txt")
#    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
#  }
#}
#pml_write_files(answers)
```




