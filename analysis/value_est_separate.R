library(lme4)
library(lmerTest)
library(tidyverse)
library(effectsize)
library(afex)
library(car)
library(gtools)
library(flextable)
library(scales)

processed_data_dir <- "~/Documents/Attn_MdPRL/Py-attention-project-analysis/data/Processed"
figure_data_dir <- "~/Documents/Attn_MdPRL/Py-attention-project-analysis/figures/All Processed"
table_data_dir <- "~/Documents/Attn_MdPRL/Py-attention-project-analysis/tables/All Processed"


gt_val_results <- data.frame(Bout=integer(),
                             Dimension=character(), 
                             Coefficients=numeric(), 
                             SEs=numeric(), 
                             pvals=numeric(),
                             pval_diff=character())
gt_val_results_for_table <- data.frame(Bout=integer(),
                                       Predictor=character(), 
                                       b=numeric(), 
                                       SE=numeric(), 
                                       t=numeric(),
                                       df=numeric(),
                                       p=numeric())
all_gt_value_data <- read.csv(file.path(processed_data_dir, "fit_gt_value_df.csv"))

for (idx_trial in 0:3){
  gt_value_data <- all_gt_value_data %>% filter(bout==idx_trial)
  gt_value_data$subj <- as.factor(gt_value_data$subj)
  
  model <- lmer(formula=prob~pFinf+pFnoninf+pO+(pFinf+pFnoninf+pO-1||subj), 
                data=gt_value_data,
                control=lmerControl(optimizer="bobyqa"),
                REML=F)
  model_summary <- coef(summary(model))
  print(summary(model))
  # print(linearHypothesis(model, "pFinf=pFnoninf", test=c("Chisq", "F")))
  contest_res <- contest(model, L=c(0,1,-1,0), joint=FALSE)
  print(contest_res)
  gt_val_results <- add_row(gt_val_results, 
                            Bout=rep(idx_trial+1, each=3),
                            Dimension=c("F[m]", "F[n]", "O"),
                            Coefficients=model_summary[,"Estimate"][2:4],
                            SEs=model_summary[,"Std. Error"][2:4],
                            pvals=model_summary[,"Pr(>|t|)"][2:4],
                            pval_diff=stars.pval(contest_res[,'Pr(>|t|)']))
  gt_val_results_for_table <- add_row(gt_val_results_for_table, 
                                      Bout=rep(idx_trial+1, each=4),
                                      Predictor=c("Intercept", "F[m]", "F[n]", "O"),
                                      b=model_summary[,"Estimate"],
                                      SE=model_summary[,"Std. Error"],
                                      t=model_summary[,"t value"],
                                      df=model_summary[,"df"],
                                      p=model_summary[,"Pr(>|t|)"])
  gt_val_results_for_table <- add_row(gt_val_results_for_table,
                                      Bout=idx_trial+1,
                                      Predictor=c("F_m-F_n"),
                                      b=contest_res[,"Estimate"],
                                      SE=contest_res[,"Std. Error"],
                                      t=contest_res[,"t value"],
                                      df=contest_res[,"df"],
                                      p=contest_res[,"Pr(>|t|)"])
}



ggplot(data=gt_val_results, mapping=aes(x=Bout, color=Dimension)) + 
  geom_line(mapping=aes(y=Coefficients), linewidth=1.5) +
  geom_errorbar(mapping=aes(ymin=Coefficients-SEs, ymax=Coefficients+SEs), width=0.15, linewidth=1.5) +
  geom_point(mapping=aes(y=Coefficients), shape=21, size=4, stroke=1.5, fill = "white") +
  geom_text(mapping=aes(y=max(Coefficients)+max(SEs)+0.05, label = pval_diff), size=12, color='black') + 
  theme(text = element_text(size = 25), axis.text = element_text(size = 25),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  scale_color_manual(breaks = c("F[m]", "F[n]", "O"), 
                     values=c('deepskyblue', 'darkorange', 'green3'),
                     name='', labels = parse_format())+
  theme(legend.position = c(0.7, 0.07), legend.direction = "horizontal", 
        legend.text.align = 0, legend.text = element_text(size=22), 
        axis.line=element_line(size=1), axis.ticks=element_line(size=1), axis.ticks.length=unit(0.1,"inch")) +
  scale_x_discrete(name ="Value estimation bout", limits=factor(1:4))

ggsave(filename="prob_est_betas_lme_sep.pdf", path=figure_data_dir, device='pdf', 
       width=5.5, height=5.5)

gt_val_results_for_table <- gt_val_results_for_table %>%
  mutate(p=if_else(p<0.001, '<0.001', as.character(round(p, 3))))
flextable(gt_val_results_for_table) %>% 
  colformat_double(j=c('b', 'SE', 't', 'df'), digits=2) %>% 
  colformat_double(j=c('p'), digits=3) %>%
  width(width=1.3) %>%
  italic(part='header') %>%
  align(align='center', part='all') %>%
  line_spacing(space=1.2) %>%
  padding(padding = 0, part = "body") %>% 
  merge_v(j='Bout') %>%
  hline(c(5,10,15,20)) %>%
  fix_border_issues() %>%
  save_as_html(path=file.path(table_data_dir, "prob_est_betas_lme_sep.html"))

anova_results <- data.frame(Bout=integer(),
                            Dimension=character(), 
                            EtaSquared=numeric(), 
                            pvals=numeric())
anova_results_for_table <- data.frame(Bout=integer(),
                                      Predictor=character(), 
                                      F=numeric(),
                                      NumDF=numeric(),
                                      DenDF=numeric(),
                                      pvals=numeric(),
                                      EtaSquared=numeric())
all_anova_data <- read.csv(file.path(processed_data_dir, "fit_anova_df.csv"))

for (idx_trial in 0:3){
  anova_data <- all_anova_data %>% filter(bout==idx_trial)
  anova_data$subj <- as.factor(anova_data$subj)
  anova_data$Finf <- as.factor(anova_data$Finf)
  anova_data$Fnoninf <- as.factor(anova_data$Fnoninf)
  
  model <- lmer_alt(formula=prob~Finf*Fnoninf+(Finf+Fnoninf-1||subj), 
                    data=anova_data,
                    contrasts = list(Finf = "contr.sum", Fnoninf = "contr.sum"),
                    control=lmerControl(optimizer="bobyqa"))
  # print(summary(model))
  print(anova(model, type="III"))
  print(eta_squared(model, alternative='two.sided'))
  
  anova_results <- add_row(anova_results, 
                           Bout=rep(idx_trial+1, each=3),
                           Dimension=c("F[m]", "F[n]", "O"),
                           EtaSquared=eta_squared(anova(model, type="III"))$Eta2,
                           pvals=anova(model, type="III")[,"Pr(>F)"][1:3])
  
  anova_results_for_table <- add_row(anova_results_for_table, 
                                     Bout=rep(idx_trial+1, each=3),
                                     Predictor=c("F[m]", "F[n]", "O"),
                                     F=anova(model, type="III")[,"F value"],
                                     NumDF=anova(model, type="III")[,"NumDF"],
                                     DenDF=anova(model, type="III")[,"DenDF"],
                                     EtaSquared=eta_squared(anova(model, type="III"))$Eta2,
                                     pvals=anova(model, type="III")[,"Pr(>F)"][1:3])
}

ggplot(data=anova_results, mapping=aes(x=Bout, color=Dimension)) + 
  geom_line(mapping=aes(y=EtaSquared), linewidth=1.5) +
  geom_point(mapping=aes(y=EtaSquared), shape=21, size=4, stroke=1.5, fill = "white") +
  theme(text = element_text(size = 25), axis.text = element_text(size = 25),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  scale_color_manual(breaks = c("F[m]", "F[n]", "O"), 
                     values=c('deepskyblue', 'darkorange', 'green3'),
                     name='', labels = parse_format())+
  theme(legend.position = c(0.68, 0.07), legend.direction = "horizontal", 
        legend.text = element_text(size=22), legend.text.align = 0,
        axis.line=element_line(size=1), axis.ticks=element_line(size=1), 
        axis.ticks.length=unit(0.1,"inch")) +
  ylab(expression(eta[p]^2)) +
  scale_x_discrete(name ="Value estimation bout", limits=factor(1:4)) + 
  scale_y_continuous(limits=c(-0.01,0.51))

ggsave(filename="prob_est_eta2_lme_sep.pdf", path=figure_data_dir, device='pdf',
       width=5.5, height=5.5)


names(anova_results_for_table) <- c('Bout', 'Predictor', 'F', 'Num. df', 'Den. df', 'p', 'eta[p]^2')
anova_results_for_table <- anova_results_for_table %>%
  mutate(p=if_else(p<0.001, '<0.001', as.character(round(p, 3))))
flextable(anova_results_for_table) %>% 
  colformat_double(j=c('F', 'Den. df', 'eta[p]^2'), digits=2) %>% 
  colformat_double(j=c('p'), digits=3) %>%
  width(width=1.3) %>%
  italic(part='header') %>%
  align(align='center', part='all') %>%
  line_spacing(space=1.2) %>%
  padding(padding = 0, part = "body") %>% 
  merge_v(j='Bout') %>%
  hline(seq(3,12,3)) %>%
  fix_border_issues() %>%
  save_as_html(path=file.path(table_data_dir, "prob_est_eta2_lme_sep.html"))

