library(lme4)
library(lmerTest)
library(tidyverse)
library(effectsize)
library(afex)
library(ggsignif)
library(gtools)

processed_data_dir <- "~/Documents/Attn_MdPRL/Py-attention-project-analysis/data/Processed"
figure_data_dir <- "~/Documents/Attn_MdPRL/Py-attention-project-analysis/figures/All Processed"

# fit results using ground truth reward probabilities
gt_value_data <- read.csv(file.path(processed_data_dir, "fit_gt_value_df.csv"))
gt_value_data$subj <- as.factor(gt_value_data$subj)
gt_value_data$bout <- scale(log(as.numeric(gt_value_data$bout)+1))

model <- lmer(formula=prob~bout*(pFinf+pFnoninf+pO)+(pFinf+pFnoninf+pO||subj), 
              data=gt_value_data,
              control=lmerControl(optimizer="bobyqa"),
              REML=F)
model_summary <- coef(summary(model))
print(summary(model))
print(contest(model, c(0,0,1,-1,0,0,0,0), joint=FALSE))

gt_val_results <- data.frame(Dimension=c("F0", "F1", "O"), 
                             Coefficients=model_summary[,'Estimate'][3:5], 
                             SEs=model_summary[,'Std. Error'][3:5], 
                             pvals=model_summary[,'Pr(>|t|)'][3:5])

ggplot(data=gt_val_results, mapping=aes(x=Dimension, fill=Dimension)) + 
  geom_bar(mapping=aes(y=Coefficients), stat="identity") +
  geom_errorbar(mapping=aes(ymin=Coefficients-SEs, ymax=Coefficients+SEs), width=0.1) +
  geom_signif(mapping=aes(y=Coefficients), comparisons = list(c("F0", "F1")), 
              map_signif_level = TRUE, 
              annotations = stars.pval(contest(model, c(0,0,1,-1,0,0,0,0), joint=FALSE)[1,'Pr(>|t|)']), 
              color = "black", size=0.5, textsize=8, margin_top=0.1) +
  theme(text = element_text(size = 22), axis.text = element_text(size = 22),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  scale_fill_manual(breaks = c("F0", "F1", "O"), values=c('deepskyblue', 'darkorange', 'green3')) + 
  scale_y_continuous(limits=c(-0.0, 0.65)) +
  theme(legend.position="none")
ggsave(filename="prob_est_betas_lme.pdf", path=figure_data_dir, device='pdf',
       width=4.5, height=4.5)


anova_data <- read.csv(file.path(processed_data_dir, "fit_anova_df.csv"))
anova_data$subj <- as.factor(anova_data$subj)
anova_data$Finf <- as.factor(anova_data$Finf)
anova_data$Fnoninf <- as.factor(anova_data$Fnoninf)
anova_data$bout <- scale(log(as.numeric(anova_data$bout)+1))
# contrasts(anova_data$bout) <- cbind(c(-1, 1, 0, 0),
#                                     c(0, -1, 1, 0),
#                                     c(0, 0, -1, 1))
  
model <- lmer_alt(formula=prob~bout*(Finf*Fnoninf)+(bout+Finf+Fnoninf||subj), 
              data=anova_data,
              contrasts = list(Finf = "contr.sum", Fnoninf = "contr.sum"),
              control=lmerControl(optimizer="bobyqa"),
              REML = F)
print(anova(model))
print(eta_squared(model))

anova_results <- data.frame(Dimension=c("F0", "F1", "O"), 
                            EtaSquared=eta_squared(anova(model), partial = TRUE)$Eta2[2:4])

ggplot(data=anova_results, mapping=aes(x=Dimension, fill=Dimension)) + 
  geom_bar(mapping=aes(y=EtaSquared), stat='identity') +
  theme(text = element_text(size = 22), axis.text = element_text(size = 22),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  scale_fill_manual(breaks = c("F0", "F1", "O"), values=c('deepskyblue', 'darkorange', 'green3')) +
  scale_y_continuous(name=expression(eta[p]^2)) +
  theme(legend.position="none")
ggsave(filename="prob_est_eta2_lme.pdf", path=figure_data_dir, device='pdf', 
       width=4.5, height=4.5)

