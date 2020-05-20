#STA9890 Final Project
#Jingyi Zhu

#PACKAGE LOAD
library(dplyr)
library(randomForest)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(grid)

#DATA LOAD 
datapath <- 'https://raw.githubusercontent.com/juliazhu1122/sta9890project/master/dataset/World%20Bank%20WDI.csv'
wdi <- read.csv(datapath)

#SHOW TOP 5 ROWS
head(wdi, n=5)

#IMPUTE NA WITH MEAN
for(i in 1:ncol(wdi)){
  wdi[is.na(wdi[,i]), i] <- mean(wdi[,i], na.rm = TRUE)
}

#STANDARDIZE
wdi_std <- wdi %>%
  select(-SP.DYN.LE00.IN) %>%
  mutate_all(.funs = function(x) {x / sqrt(mean((x - mean(x))^2))})

colnames(wdi_std) <- c('GDP1',
                       'GDP2',
                       'Inflation1',
                       'Inflation2',
                       'Agricultural',
                       'CO2',
                       'Infant',
                       'Under5',
                       'Physicians',
                       'Beds',
                       'Food',
                       'Crop',
                       'Livestock',
                       'Fertility',
                       'Adolescent',
                       'Births',
                       'Popu014',
                       'Popu1564',
                       'Popu65',
                       'Popugrowth',
                       'Popudensity',
                       'Homicides',
                       'Water1',
                       'Water2',
                       'Sanitation1',
                       'Sanitation2',
                       'GINI',
                       'Outschool1',
                       'Outschool2',
                       'Education',
                       'Birth_rate',
                       'Health',
                       'Health1',
                       'DPT',
                       'HepB3',
                       'Measles',
                       'Smoking',
                       'Rural',
                       'Urban',
                       'Fuels',
                       'Electricity')

#SHOW TOP 5 ROWS
head(wdi_std, n=5)

#SHOW Y DISTRIBUTION
hist(wdi$SP.DYN.LE00.IN)

#PREPARE X
X <- data.matrix(wdi_std)

#FIX LEFT SKEWED
y <- sqrt(wdi$SP.DYN.LE00.IN)

#BASIC PARAMS
set.seed(42)

n <- nrow(wdi_std) # number of total observations
p <- ncol(wdi_std)    # number of predictors
n.train <- floor(0.8 * n)
n.test  <- n - n.train

M <- 100

#LASSO R-SQUARED
Rsq.test.la  <- rep(0, M)  
Rsq.train.la <- rep(0, M)

#ELASTIC NET R-SQUARED
Rsq.test.en  <- rep(0, M)
Rsq.train.en <- rep(0, M)

#RIDGE R-SQUARED
Rsq.test.ri  <- rep(0, M)
Rsq.train.ri <- rep(0, M)

#RANDOM FOREST R-SQUARED
Rsq.test.rf  <- rep(0, M)
Rsq.train.rf <- rep(0, M)

for (m in c(1:M)) {
  
  shuffled_indexes <-     sample(n)
  train            <-     shuffled_indexes[1:n.train]
  test             <-     shuffled_indexes[(1+n.train):n]
  X.train          <-     X[train, ]
  y.train          <-     y[train]
  X.test           <-     X[test, ]
  y.test           <-     y[test]
  
  # Fit lasso and calculate and record the train and test R squares, and estimated coefficients 
  cv.fit           <-     cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
  fit              <-     glmnet(X.train, y.train, alpha = 1, lambda = cv.fit$lambda.min)
  y.train.hat      <-     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       <-     predict(fit, newx = X.test, type = "response")  # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.la[m]   <-     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.la[m]  <-     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  
  # Fit elastic-net and calculate and record the train and test R squares, and estimated coefficients  
  cv.fit           <-     cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
  fit              <-     glmnet(X.train, y.train, alpha = 0.5, lambda = cv.fit$lambda.min)
  y.train.hat      <-     predict(fit, newx = X.train, type = "response")
  y.test.hat       <-     predict(fit, newx = X.test, type = "response") 
  Rsq.test.en[m]   <-     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.en[m]  <-     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  
  # Fit ridge and calculate and record the train and test R squares, and estimated coefficients 
  cv.fit           <-     cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
  fit              <-     glmnet(X.train, y.train, alpha = 0, lambda = cv.fit$lambda.min)
  y.train.hat      <-     predict(fit, newx = X.train, type = "response") 
  y.test.hat       <-     predict(fit, newx = X.test, type = "response") 
  Rsq.test.ri[m]   <-     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.ri[m]  <-     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  
  # Fit RF and calculate and record the train and test R squares, and estimated coefficients  
  rf               <-     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE, ntree = 100, nodesize = 100)
  y.test.hat       <-     predict(rf, X.test)
  y.train.hat      <-     predict(rf, X.train)
  Rsq.test.rf[m]   <-     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m]  <-     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  
  cat(sprintf("m=%3.f| Rsq.test.rf=%.2f,  Rsq.test.en=%.2f| Rsq.train.rf=%.2f,  Rsq.train.en=%.2f| \n", m,  Rsq.test.rf[m], Rsq.test.en[m],  Rsq.train.rf[m], Rsq.train.en[m]))
  
}

#SIDE-BY-SIDE BOXPLOTS
par(mfrow=c(1,2))
boxplot(Rsq.train.la, Rsq.train.en, Rsq.train.ri, Rsq.train.rf,
        main = "TRAIN SET R-SQUARED",
        names = c("LASSO", "EN", "RIDGE", "RF"),
        col = c("indianred2","lightgoldenrod2", "lightskyblue2", "tan2"))

boxplot(Rsq.test.la, Rsq.test.en, Rsq.test.ri, Rsq.test.rf,
        main = "TEST SET R-SQUARED",
        names = c("LASSO", "EN", "RIDGE", "RF"),
        col = c("indianred2","lightgoldenrod2", "lightskyblue2", "tan2"))

#10-FOLD CV CURVES
cv.la <- cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
cv.en <- cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
cv.ri <- cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)

par(mfrow=c(1,3))
plot(cv.la)
title('LASSO', line = 2.5)
plot(cv.en)
title('EN', line = 2.5)
plot(cv.ri)
title('RIDGE', line = 2.5)

#SIDE-BY-SIDE BOXPLOTS OF RESIDUALS
cv.fit <- cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
fit <- glmnet(X.train, y.train, alpha = 1, lambda = cv.fit$lambda.min)
y.train.hat <- predict(fit, newx = X.train, type = "response") 
y.test.hat <- predict(fit, newx = X.test, type = "response")  
Res.test.la <- y.test - y.test.hat
Res.test.la <- as.vector(Res.test.la)
Res.train.la <- y.train - y.train.hat
Res.train.la <- as.vector(Res.train.la)

cv.fit <- cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
fit <- glmnet(X.train, y.train, alpha = 0.5, lambda = cv.fit$lambda.min)
y.train.hat <- predict(fit, newx = X.train, type = "response") 
y.test.hat <- predict(fit, newx = X.test, type = "response")  
Res.test.en <- y.test - y.test.hat
Res.test.en <- as.vector(Res.test.en)
Res.train.en <- y.train - y.train.hat
Res.train.en <- as.vector(Res.train.en)

cv.fit <- cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
fit <- glmnet(X.train, y.train, alpha = 0, lambda = cv.fit$lambda.min)
y.train.hat <- predict(fit, newx = X.train, type = "response") 
y.test.hat <- predict(fit, newx = X.test, type = "response")  
Res.test.ri <- y.test - y.test.hat
Res.test.ri <- as.vector(Res.test.ri)
Res.train.ri <- y.train - y.train.hat
Res.train.ri <- as.vector(Res.train.ri)

rf <- randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE, ntree = 100, nodesize = 100)
y.test.hat <- predict(rf, X.test)
y.train.hat <- predict(rf, X.train)
Res.test.rf <- y.test - y.test.hat
Res.train.rf <- y.train - y.train.hat

par(mfrow=c(1,2))
boxplot(Res.train.la, Res.train.en, Res.train.ri, Res.train.rf,
        main = "RESIDUALS IN TRAIN SET",
        names = c("LASSO", "EN", "RIDGE", "RF"),
        col = c("indianred2","lightgoldenrod2", "lightskyblue2", "tan2"))

boxplot(Res.test.la, Res.test.en, Res.test.ri, Res.test.rf,
        main = "RESIDUALS IN TEST SET",
        names = c("LASSO", "EN", "RIDGE", "RF"),
        col = c("indianred2","lightgoldenrod2", "lightskyblue2", "tan2"))

#BOOTSTRAP
bootstrapSamples <- 100
beta.la.bs <- matrix(0, nrow = p, ncol = bootstrapSamples)  
beta.en.bs <- matrix(0, nrow = p, ncol = bootstrapSamples)  
beta.ri.bs <- matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.rf.bs <- matrix(0, nrow = p, ncol = bootstrapSamples)         

for (m in 1:bootstrapSamples){
  bs_indexes       <-     sample(n, replace=T)
  X.bs             <-     X[bs_indexes, ]
  y.bs             <-     y[bs_indexes]
  
  # fit bs rf
  rf               <-     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m]   <-     as.vector(rf$importance[,1])
  
  # fit bs la
  a                <-     1 # lasso
  cv.fit           <-     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit              <-     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.la.bs[,m]   <-     as.vector(fit$beta)
  
  # fit bs en
  a                <-     0.5 # elastic-net
  cv.fit           <-     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit              <-     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m]   <-     as.vector(fit$beta)
  
  # fit bs ri
  a                <-     0 # ridge
  cv.fit           <-     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit              <-     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)
  beta.ri.bs[,m]   <-     as.vector(fit$beta)
  
  cat(sprintf("Bootstrap Sample %3.f \n", m))
}

la.bs.sd <- apply(beta.la.bs, 1, "sd")
en.bs.sd <- apply(beta.en.bs, 1, "sd")
ri.bs.sd <- apply(beta.ri.bs, 1, "sd")
rf.bs.sd <- apply(beta.rf.bs, 1, "sd")

#BAR-PLOTS WITH BOOTSTRAPPED ERROR BARS
rf <- randomForest(X, y, mtry = sqrt(p), importance = TRUE)

cv.la <- cv.glmnet(X, y, alpha = 1, nfolds = 10)
la <- glmnet(X, y, alpha = 1, lambda = cv.la$lambda.min)

cv.en <- cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
en <- glmnet(X, y, alpha = 0.5, lambda = cv.en$lambda.min)

cv.ri <- cv.glmnet(X, y, alpha = 0, nfolds = 10)
ri <- glmnet(X, y, alpha = 0, lambda = cv.ri$lambda.min)

betaS.rf <- data.frame(names(X[1,]), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf) <- c( "feature", "value", "err")

betaS.la <- data.frame(names(X[1,]), as.vector(la$beta), 2*la.bs.sd)
colnames(betaS.la) <- c( "feature", "value", "err")

betaS.en <- data.frame(names(X[1,]), as.vector(en$beta), 2*en.bs.sd)
colnames(betaS.en) <- c( "feature", "value", "err")

betaS.ri <- data.frame(names(X[1,]), as.vector(ri$beta), 2*ri.bs.sd)
colnames(betaS.ri) <- c( "feature", "value", "err")

betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.la$feature     =  factor(betaS.la$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.ri$feature     =  factor(betaS.ri$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])

plot_rf <- ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle('RANDOM FOREST') + theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
                                   axis.text.x = element_text(angle = 90, hjust = 1), 
                                   axis.title.x = element_blank())

plot_la <- ggplot(betaS.la, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle('LASSO') + theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
                                 axis.text.x = element_text(angle = 90, hjust = 1), 
                                 axis.title.x = element_blank())

plot_en <- ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle('ELASTIC NET') + theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
                                 axis.text.x = element_text(angle = 90, hjust = 1), 
                                 axis.title.x = element_blank())

plot_ri <- ggplot(betaS.ri, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle('RIDGE') + theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
                                 axis.text.x = element_text(angle = 90, hjust = 1), 
                                 axis.title.x = element_blank())

grid.arrange(plot_rf, plot_la, plot_en, plot_ri, nrow = 2)

#PERFORMANCE AND TIME SUMMARY
LA_start <- Sys.time()
cv.la <- cv.glmnet(X, y, alpha = 1, nfolds = 10)
la <- glmnet(X, y, alpha = 1, lambda = cv.la$lambda.min)
LA_end <- Sys.time()
LA_time <- LA_end - LA_start

EN_start <- Sys.time()
cv.en <- cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
en <- glmnet(X, y, alpha = 0.5, lambda = cv.en$lambda.min)
EN_end <- Sys.time()
EN_time <- EN_end - EN_start

RI_start <- Sys.time()
cv.ri <- cv.glmnet(X, y, alpha = 0, nfolds = 10)
ri <- glmnet(X, y, alpha = 0, lambda = cv.ri$lambda.min)
RI_end <- Sys.time()
RI_time <- RI_end - RI_start

RF_start <- Sys.time()
rf <- randomForest(X, y, mtry = sqrt(p), importance = TRUE, ntree = 100, nodesize = 100)
RF_end <- Sys.time()
RF_time <- RF_end - RF_start

model <- c('LASSO', 'ELASTIC NET', 'RIDGE', 'RANDOM FOREST')
performance <- round(c(mean(Rsq.test.la), mean(Rsq.test.en), mean(Rsq.test.ri), mean(Rsq.test.rf)),4)
time <- round(c(LA_time, EN_time, RI_time, RF_time),4)

comparison <- data.frame(model, performance, time)
colnames(comparison) <- c('MODEL', 'PERFORMANCE', 'TIME')

grid.table(comparison)