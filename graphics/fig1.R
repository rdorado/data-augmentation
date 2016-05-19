cairo_pdf("~/Dropbox/RUBEN-PHD/DGA1005/writings/figures/fig1.pdf", height=8, width=8)

data <- read.csv("result005.table.dat", header = FALSE)
data <- rbind(data, read.csv("result01.table.dat",header = FALSE))
data <- rbind(data, read.csv("result015.table.dat",header = FALSE))
data <- rbind(data, read.csv("result02.table.dat",header = FALSE))
data <- rbind(data, read.csv("result025.table.dat",header = FALSE))
data <- rbind(data, read.csv("result03.table.dat",header = FALSE))
data <- rbind(data, read.csv("result035.table.dat",header = FALSE))
data <- rbind(data, read.csv("result04.table.dat",header = FALSE))
data <- rbind(data, read.csv("result045.table.dat",header = FALSE))
data <- rbind(data, read.csv("result05.table.dat",header = FALSE))

par(mfrow=c(2,2), mar=c(5,4,1,1))

colors <- c("blue", "green", "azure3", "azure3")
labels <- c("NB", "SVM", "NB+AUG", "SVM+AUG")

plot(data[,3],data[,5],type="n", ylim=c(0,1),pch=4, xlab="Percentage of data",ylab="Micro-aver Accuracy")
lines(lowess(data[,3],data[,5]), lty=4)

lines(lowess(data[,3],data[,9]), lty=3)

lines(lowess(data[,3],data[,19]), lty=2)

lines(lowess(data[,3],data[,23]), lty=1)

legend("bottomright", inset=.03, labels, lty=c(4, 3, 2, 1), bg = "white", cex=0.6)



plot(data[,3],data[,6],type="n",ylim=c(0,1),pch=4, xlab="Percentage of data",ylab="Macro-aver Precision")
lines(lowess(data[,3],data[,6]), lty=4)

lines(lowess(data[,3],data[,10]), lty=3)

lines(lowess(data[,3],data[,20]), lty=2)

lines(lowess(data[,3],data[,24]), lty=1)

legend("bottomright", inset=.03, labels, lty=c(4, 3, 2, 1), bg = "white", cex=0.6)



plot(data[,3],data[,7],type="n",ylim=c(0,1),pch=4, xlab="Percentage of data",ylab="Macro-aver Recall")
lines(lowess(data[,3],data[,7]), lty=4)

lines(lowess(data[,3],data[,11]), lty=3)

lines(lowess(data[,3],data[,21]), lty=2)

lines(lowess(data[,3],data[,25]), lty=1)

legend("bottomright", inset=.03, labels, lty=c(4, 3, 2, 1), bg = "white", cex=0.6)



plot(data[,3],data[,8],type="n",ylim=c(0,1),pch=4, xlab="Percentage of data",ylab="Macro-aver F1 score")
lines(lowess(data[,3],data[,8]), lty=4)

lines(lowess(data[,3],data[,12]), lty=3)

lines(lowess(data[,3],data[,22]), lty=2)

lines(lowess(data[,3],data[,26]), lty=1)

legend("bottomright", inset=.03, labels, lty=c(4, 3, 2, 1), bg = "white", cex=0.6)


dev.off()
