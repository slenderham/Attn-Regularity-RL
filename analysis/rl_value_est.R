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
all_gt_value_data <- read.csv(file.path(processed_data_dir, "fit_subj_value_df.csv"))

for (idx_trial in 0:3){
  gt_value_data <- all_gt_value_data %>% filter(bout==idx_trial)
  gt_value_data$subj <- as.factor(gt_value_data$subj)
  
  model <- lmer(formula=prob~pF+pO+(pF+pO||subj), 
                data=gt_value_data,
                control=lmerControl(optimizer="bobyqa"),
                REML=F)
  model_summary <- coef(summary(model))
  print(summary(model))
  # print(anova(model, type='III'))
  # print(eta_squared(model))
  # print(linearHypothesis(model, "pF=pO", test=c("Chisq", "F")))
  gt_val_results <- add_row(gt_val_results, 
                            Bout=rep(idx_trial+1, each=2),
                            Dimension=c("F[AL]", "O"),
                            Coefficients=model_summary[,"Estimate"][2:3],
                            SEs=model_summary[,"Std. Error"][2:3],
                            pvals=model_summary[,"Pr(>|t|)"][2:3])
  gt_val_results_for_table <- add_row(gt_val_results_for_table, 
                                      Bout=rep(idx_trial+1, each=3),
                                      Predictor=c("Intercept", "F[AL]", "O"),
                                      b=model_summary[,"Estimate"],
                                      SE=model_summary[,"Std. Error"],
                                      t=model_summary[,"t value"],
                                      df=model_summary[,"df"],
                                      p=model_summary[,"Pr(>|t|)"])
}

ggplot(data=gt_val_results, mapping=aes(x=Bout, color=Dimension)) + 
  geom_line(mapping=aes(y=Coefficients), linewidth=1.5) +
  geom_errorbar(mapping=aes(ymin=Coefficients-SEs, ymax=Coefficients+SEs), width=0.1, linewidth=1) +
  geom_point(mapping=aes(y=Coefficients), shape=21, size=3, stroke=2, fill = "white") +
  theme(text = element_text(size = 25), axis.text = element_text(size = 25),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  scale_color_manual(breaks = c("F[AL]",  "O"), values=c('brown', 'green3'), 
                     name='', labels = parse_format())+
  theme(legend.position = c(0.7, 0.8), legend.direction = "horizontal", 
        legend.text.align = 0, legend.text = element_text(size=25),
        axis.line=element_line(size=1), axis.ticks=element_line(size=1), axis.ticks.length=unit(0.1,"inch"))+
  scale_x_discrete(name ="Value estimation bout", limits=factor(1:4))
ggsave(filename="rl_prob_est_betas.pdf", path=figure_data_dir, device='pdf', 
       width=6.5, height=4.5)

gt_val_results_for_table <- gt_val_results_for_table %>%
  mutate(p=if_else(p<0.001, '<0.001', as.character(round(p, 3))))
flextable(gt_val_results_for_table) %>% 
  colformat_double(j=c('b', 'SE', 't', 'df'), digits=2) %>% 
  colformat_double(j=c('p'), digits=3) %>%
  width(width=1) %>%
  italic(part='header') %>%
  align(align='center', part='all') %>%
  line_spacing(space=1.5) %>%
  padding(padding = 0, part = "body") %>% 
  merge_v(j='Bout') %>%
  hline(seq(3,12,3)) %>%
  fix_border_issues() %>%
  save_as_html(path=file.path(table_data_dir, "rl_prob_est_betas.html"))
