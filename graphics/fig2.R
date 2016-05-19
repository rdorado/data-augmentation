pdf("~/Dropbox/RUBEN-PHD/DGA1005/writings/figures/fig2.pdf", height=4, width=8)

data <- read.csv("result_t5_k50_p03.table.dat", header = FALSE)
data <- rbind(data, read.csv("result_t10_k50_p03.table.dat",header = FALSE))
data <- rbind(data, read.csv("result_t15_k50_p03.table.dat",header = FALSE))
data <- rbind(data, read.csv("result_t20_k50_p03.table.dat",header = FALSE))
data <- rbind(data, read.csv("result_t25_k50_p03.table.dat",header = FALSE))


par(mfrow=c(1,2), mar=c(5,4,1,1))

colors <- c("blue", "green", "azure3", "azure3")
labels <- c("NB", "SVM")

plot(data[,1],data[,19],type="n", ylim=c(0,1),pch=4, xlab="Number of topics",ylab="Micro-aver Accuracy")

lines(lowess(data[,1],data[,19]), lty=2)

lines(lowess(data[,1],data[,23]), lty=1)

legend("bottomright", inset=.03, labels, lty=c(2, 1), bg = "white", cex=0.6)



plot(data[,1],data[,22],type="n",ylim=c(0,1),pch=4, xlab="Number of topics",ylab="F1 Score")

lines(lowess(data[,1],data[,22]), lty=2)

lines(lowess(data[,1],data[,26]), lty=1)

legend("bottomright", inset=.03, labels, lty=c(2, 1), bg = "white", cex=0.6)



dev.off()

