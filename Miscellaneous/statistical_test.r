############################################################
## N-GRAM VS. T5 ON PERFECT PREDICTIONS
############################################################

library(exact2x2)
library(effsize)
library(xtable)


datasets=c("javadoc","inside",'overall')

res=list(Dataset=c(),McNemar.p=c(),McNemar.OR=c())
for(d in datasets){
    t<-read.csv(paste("/Users/antonio/Desktop/result_comparison_",d,".csv",sep=""))
    m=mcnemar.exact(t$is_perfect_baseline,t$is_perfect_T5)
    res$Dataset=c(res$Dataset,as.character(d))
    res$McNemar.p=c(res$McNemar.p, m$p.value)
    res$McNemar.OR=c(res$McNemar.OR,m$estimate)
}

res=data.frame(res)
#p-value adjustment
res$McNemar.p=p.adjust(res$McNemar.p,method="BH")
print(res)
# print(xtable(res),include.rownames=FALSE)
