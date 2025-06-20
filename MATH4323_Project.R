data <- read.csv("Student Depression Dataset.csv", header=T)

student_depression_cleaned <- subset(data, select = -`Work.Pressure`)
student_depression_cleaned <- subset(student_depression_cleaned, select = -`Job.Satisfaction`)
student_depression_cleaned <- student_depression_cleaned[student_depression_cleaned$Profession == "Student", ]
student_depression_cleaned <- subset(student_depression_cleaned, select = -`Profession`)
sdc <- subset(student_depression_cleaned, select = -`id`)
names(sdc)[10] <- "Suicidal.Thoughts" 

colSums(is.na(sdc)) # Check for missing values - returns three missing financial.stress values.
median(sdc$Financial.Stress, na.rm = TRUE) # grab median - returns 3
sdc$Financial.Stress[is.na(sdc$Financial.Stress)] <- 3
colSums(is.na(sdc)) # returns all 0s, so we're good there.

sapply(sdc, sd) # check to see if the data needs to be scaled. 
boxplot(sdc$Age) # definitely has outliers
# I concluded age should be scaled 
# concern for outliers being present while utilizing two outlier sensitive methods
sdc$Age <- scale(sdc$Age)
names(sdc)[2] <- "Age"
sapply(sdc, sd) # check to see if the data needs to be scaled. 
boxplot(sdc$Work.Study.Hours) # High standard dev but no outliers - acceptable. No more scaling.

# Check for duplicated rows
any(duplicated(sdc)) # since this returns false, we're safe on dupe rows.
sdc$Gender <- as.factor(sdc$Gender)
sdc$City <- as.factor(sdc$City)
sdc$Sleep.Duration <- as.factor(sdc$Sleep.Duration)
sdc$Dietary.Habits <- as.factor(sdc$Dietary.Habits)
sdc$Degree <- as.factor(sdc$Degree)
sdc$Suicidal.Thoughts <- as.factor(sdc$Suicidal.Thoughts)
sdc$Family.History.of.Mental.Illness <- as.factor(sdc$Family.History.of.Mental.Illness)
sdc$Depression <- as.factor(sdc$Depression)


#KNN:
# we are using numerical variables to predict
numeric_predictors = c("Age", "Academic.Pressure", "CGPA", "Study.Satisfaction", "Work.Study.Hours", "Financial.Stress")
X = sdc[, numeric_predictors]
y = sdc$Depression

#dividing into train/test set
set.seed(123)
n = nrow(X)
train = sample(1:n, 0.8*n)
X.train = X[train, ]
y.train = y[train]
X.test = X[-train, ]
y.test = y[-train]

library(ISLR)
# Scale training set
X.train.scaled = scale(X.train)
# Save the mean and standard deviation
train.center = attr(X.train.scaled, "scaled:center")
train.scale = attr(X.train.scaled, "scaled:scale")
#test set is scaled the same as train set
X.test.scaled = scale(X.test, center = train.center, scale = train.scale)

library(class)
#validation set approach testing
K.set <-seq(1,401,by = 5)
knn.test.err<-numeric(length(K.set))
set.seed(123)
for (j in 1:length(K.set)){knn.pred<-knn(train = X.train.scaled, test = X.test.scaled, cl= y.train, k = K.set[j]); knn.test.err[j] <-mean(knn.pred!=y.test)}
#test error for best K
min(knn.test.err)

#value of best K
K.set[which.min(knn.test.err)]




