library(psych)
library(xlsx)
library(corrplot)
# library(GGally)
setwd("Desktop/Enginyeria Computacional i Matemàtica/7. Análisis de datos multivariantes/")
census = read.csv(file="CASCrefmicrodata.csv")
str(census)
stats = describe(census)
stats$variation = stats$sd / stats$mean
stats$m_m = stats$mad / stats$median
stats$centrality_dis = stats$mean / stats$median
stats$variability_dis = stats$sd / stats$mad
latexCentralityStats = data.frame(stats$vars, stats$mean, stats$sd, stats$skew, stats$kurtosis, stats$variation)
latexCentralityStats[,1] = colnames(census)
colnames(latexCentralityStats) = c("Var.", "Mean", "Standard Dev.", "Skewness", "Kurtosis", "Variation")
<<results=tex>>
  xtable(latexCentralityStats)
write.xlsx(stats,file = "stats.xlsx")
latexRobustStats = data.frame(stats$vars, stats$median, stats$centrality_dis, stats$mad, stats$variability_dis, stats$m_m)
latexRobustStats[,1] = colnames(census)
colnames(latexRobustStats) = c("Var.", "Median", "Mean/Med.", "MAD", "SD/Mad", "MAD/Med.")
<<results=tex>>
  xtable(latexRobustStats)
write.xlsx(stats,file = "stats.xlsx")


attach(census)
par(mfrow=c(1,2)) 
# 1 
hist(census$AFNLWGT)
boxplot(census$AFNLWGT)
# 2
hist(census$AGI)
boxplot(census$AGI)
# 3 
hist(census$EMCONTRB)
boxplot(census$EMCONTRB)
# 4 
hist(census$FEDTAX)
boxplot(census$FEDTAX)
# 5 
hist(census$PTOTVAL)
boxplot(census$PTOTVAL)
# 6
hist(census$STATETAX)
boxplot(census$STATETAX)
# 7
hist(census$TAXINC)
boxplot(census$TAXINC)
# 8 
hist(census$POTHVAL)
boxplot(census$POTHVAL)
# 9
hist(census$INTVAL)
boxplot(census$INTVAL)
# 10 
hist(census$PEARNVAL)  
boxplot(census$PEARNVAL)  
# 11
hist(census$FICA)
boxplot(census$FICA)
# 12 
hist(census$WSALVAL)
boxplot(census$WSALVAL)
# 13
hist(census$ERNVAL)
boxplot(census$ERNVAL)
# Análisis de la covarianza
covariance = cov(census)
write.xlsx(covariance,file = "covariance.xlsx")
logCensus = log(census)
logCovariance = cov(logCensus)
write.xlsx(logCovariance,file = "logCovariance.xlsx")
eigen = eigen(covariance)
write.xlsx(eigen$values,file = "eigenvalues.xlsx")
besley = max(eigen$values) / eigen$values
write.xlsx(besley,file = "besley.xlsx")
weights = eigen$vectors[,13] / max(eigen$vectors[,13])
write.xlsx(weights,file = "weights.xlsx")
correlations = cor(census)
par(mfrow=c(1,1)) 
corrplot(correlations, method = "number")
corrplot(correlations, order = "hclust", addrect = 2, method="number")
totalVariance = sum(diag(covariance))
avgVariance = totalVariance / length(diag(covariance))
genVariance = det(covariance)
effVariance = genVariance**(1/length(diag(covariance)))
effStdDev = sqrt(effVariance)
pairs.panels(census)
# ggpairs(census)
install.packages("mvoutlier")







