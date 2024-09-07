library(lme4)
library(lmerTest)
library(tidyverse)
library(effectsize)
library(car)
library(gtools)
library(patchwork)
library(flextable)
library(scales)

stars.pval <- function(x){
  stars <- c("***", "**", "*", "")
  var <- c(0, 0.001, 0.01, 0.05, 1)
  i <- findInterval(x, var, left.open = T, rightmost.closed = T)
  stars[i]
}


processed_data_dir <- "~/Documents/Attn_MdPRL/Py-attention-project-analysis/data/Processed"
figure_data_dir <- "~/Documents/Attn_MdPRL/Py-attention-project-analysis/figures/All Processed"
table_data_dir <- "~/Documents/Attn_MdPRL/Py-attention-project-analysis/tables/All Processed"

all_credit_assignment_data <- read.csv(file.path(processed_data_dir, "credit_assignment_df.csv"))

# fit first half

credit_assignment_data1 <- all_credit_assignment_data %>% filter(block==0)
model1 <- glmer(choice ~ F0_Chosen_R + F0_Chosen_C + F0_Foregone_R + F0_Foregone_C + 
                 F1_Chosen_R + F1_Chosen_C + F1_Foregone_R + F1_Foregone_C + 
                 O_Chosen_R + O_Chosen_C + O_Foregone_R + O_Foregone_C + 
                 (F0_Chosen_C + F0_Foregone_R + F0_Foregone_C + 
                    F1_Chosen_C + F1_Foregone_C + 
                    O_Foregone_R -1||subj), 
               data=credit_assignment_data1, family = binomial(link = "logit"),
               control=glmerControl(optimizer = "bobyqa"))
print(summary(model1))
model1_summary <- coef(summary(model1))

# make results summary with first half

results_df_for_table <- data.frame(model1_summary)
names(results_df_for_table) <- c('b', 'SE', 'z', 'p')

results_df_for_table <- results_df_for_table %>%
  add_column(Predictor=c('Intercept', 'F0 chosen WSLS', 'F0 chosen CA', 'F0 foregone WSLS', 'F0 foregone CA', 
                         'F1 chosen WSLS', 'F1 chosen CA', 'F1 foregone WSLS', 'F1 foregone CA', 
                         'O chosen WSLS', 'O chosen CA', 'O foregone WSLS', 'O foregone CA'),
             .before=0) %>%
  mutate(p=if_else(p<0.001, '<0.001', as.character(round(p, 3))))
flextable(results_df_for_table) %>% 
  colformat_double(j=c('b', 'SE', 'z'), digits=2) %>% 
  colformat_double(j=c('p'), digits=3) %>%
  width(width=1.5) %>%
  italic(part='header') %>%
  align(align='center', part='all') %>%
  line_spacing(space=1.2) %>%
  padding(padding = 0, part = "body") %>% 
  save_as_html(path=file.path(table_data_dir, "credit_assignment_lme_1.html"))


#  fit second half
credit_assignment_data2 <- all_credit_assignment_data %>% filter(block==1)
model2 <- glmer(choice ~ F0_Chosen_R + F0_Chosen_C + F0_Foregone_R + F0_Foregone_C +
                         F1_Chosen_R + F1_Chosen_C + F1_Foregone_R + F1_Foregone_C +
                         O_Chosen_R + O_Chosen_C + O_Foregone_R + O_Foregone_C +
                        ( F0_Chosen_C + F0_Foregone_C +
                          F1_Chosen_C + F1_Foregone_C +
                           -1 ||subj),
                data=credit_assignment_data2, family = binomial(link = "logit"),
                control=glmerControl(optimizer = "bobyqa"))
print(summary(model2))
model2_summary <- coef(summary(model2))

# make results summary for second half
results_df_for_table <- data.frame(model2_summary)
names(results_df_for_table) <- c('b', 'SE', 'z', 'p')

results_df_for_table <- results_df_for_table %>%
  add_column(Predictor=c('Intercept', 'F0 chosen WSLS', 'F0 chosen CA', 'F0 foregone WSLS', 'F0 foregone CA', 
                         'F1 chosen WSLS', 'F1 chosen CA', 'F1 foregone WSLS', 'F1 foregone CA', 
                         'O chosen WSLS', 'O chosen CA', 'O foregone WSLS', 'O foregone CA'),
             .before=0) %>%
  mutate(p=if_else(p<0.001, '<0.001', as.character(round(p, 3))))
flextable(results_df_for_table) %>% 
  colformat_double(j=c('b', 'SE', 'z'), digits=2) %>% 
  colformat_double(j=c('p'), digits=3) %>%
  width(width=1.5) %>%
  italic(part='header') %>%
  align(align='center', part='all') %>%
  line_spacing(space=1.2) %>%
  padding(padding = 0, part = "body") %>% 
  save_as_html(path=file.path(table_data_dir, "credit_assignment_lme_2.html"))


# plot coefficients, chosen, wsls

varnames <- rep(c("1st half", "2nd half"), each=3)
dimnames <- rep(c("Ft[man]", "Ft[non]", "Obj"), time=2)

credit_assignment_results1 <- data.frame(PlotOrder=1:6,
                                         Dimension=dimnames, 
                                         VarNames=varnames,
                                         Coefficients=c(model1_summary[,"Estimate"][c(2,6,10)], 
                                                        model2_summary[,"Estimate"][c(2,6,10)]), 
                                         SEs=c(model1_summary[,"Std. Error"][c(2,6,10)], 
                                               model2_summary[,"Std. Error"][c(2,6,10)]),
                                         pvals=stars.pval(c(model1_summary[,"Pr(>|z|)"][c(2,6,10)], 
                                                            model2_summary[,"Pr(>|z|)"][c(2,6,10)])))

p1 <- ggplot(data=credit_assignment_results1, mapping = aes(x=VarNames, group=Dimension)) +
  geom_bar(mapping=aes(y=Coefficients, fill=Dimension), stat='identity', position=position_dodge(.9), 
           show.legend = FALSE, size=1.5, color='black') + 
  geom_errorbar(mapping=aes(ymin=Coefficients-SEs, ymax=Coefficients+SEs), color='black', 
                position=position_dodge(.9), width=0.25, linewidth=1.5, show.legend = FALSE) +
  geom_text(mapping=aes(y=Coefficients+sign(Coefficients)*(SEs+0.05)-0.04, label=pvals),
            position=position_dodge(.9), vjust='center', size=10) +
  theme(text = element_text(size = 25), axis.text = element_text(size = 25), axis.title.x = element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  scale_color_manual(breaks = c("Ft[man]", "Ft[non]", "Obj"), values=c('#4dbbd5', '#e64b35', 'grey50')) +
  scale_fill_manual(breaks = c("Ft[man]", "Ft[non]", "Obj"), values=c('#4dbbd5', '#e64b35', 'grey50'), 
                    labels = parse_format())+
  theme(legend.text = element_text(size=22), legend.text.align = 0, 
        legend.title=element_blank(), legend.position = c(0.4, 0.9),
        legend.direction="horizontal", legend.background = element_rect(fill='transparent'),
        axis.line=element_line(size=1), axis.ticks=element_line(size=1), 
        axis.ticks.length=unit(0.1,"inch"), axis.text.x=element_blank(), 
        plot.title = element_text(hjust=0.5, size=25)) +
  ylim(c(-0.32, 0.8))+
  ylab('WSLS')+
  ggtitle('Chosen')

# plot coefficients, unchosen, wsls

credit_assignment_results2 <- data.frame(PlotOrder=1:6,
                                         Dimension=dimnames, 
                                         VarNames=varnames,
                                         Coefficients=c(model1_summary[,"Estimate"][c(4,8,12)], 
                                                        model2_summary[,"Estimate"][c(4,8,12)]), 
                                         SEs=c(model1_summary[,"Std. Error"][c(4,8,12)], 
                                               model2_summary[,"Std. Error"][c(4,8,12)]),
                                         pvals=stars.pval(c(model1_summary[,"Pr(>|z|)"][c(4,8,12)], 
                                                            model2_summary[,"Pr(>|z|)"][c(4,8,12)])))

p2 <- ggplot(data=credit_assignment_results2, mapping = aes(x=VarNames, group=Dimension)) +
  geom_bar(mapping=aes(y=Coefficients, fill=Dimension), stat='identity', position=position_dodge(.9),
           size=1.5, color='black', show.legend = FALSE) + 
  geom_errorbar(mapping=aes(ymin=Coefficients-SEs, ymax=Coefficients+SEs), color='black', 
                position=position_dodge(.9), width=0.25, linewidth=1.5, show.legend = FALSE) +
  geom_text(mapping=aes(y=Coefficients+sign(Coefficients)*(SEs+0.05)-0.04, label=pvals),
            position=position_dodge(.9), vjust='center', size=10) +
  theme(text = element_text(size = 25), axis.text = element_text(size = 25), axis.title.x = element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  scale_color_manual(breaks = c("Ft[man]", "Ft[non]", "Obj"), values=c('#4dbbd5', '#e64b35', 'grey50')) +
  scale_fill_manual(breaks = c("Ft[man]", "Ft[non]", "Obj"), values=c('#4dbbd5', '#e64b35', 'grey50'), 
                    labels = parse_format())+
  theme(legend.text = element_text(size=25), legend.text.align = 0, 
        legend.title=element_blank(), legend.position = c(1.2, 0),
        axis.line=element_line(size=1), axis.ticks=element_line(size=1), 
        axis.ticks.length=unit(0.1,"inch"), axis.text.x=element_blank(),
        axis.title.y=element_blank(), plot.title = element_text(hjust=0.5,size=25)) +
  ylim(c(-0.32, 0.8))+
  ggtitle('Unchosen')

# plot coefficients, chosen, ca
credit_assignment_results3 <- data.frame(PlotOrder=1:6,
                                         Dimension=dimnames, 
                                         VarNames=varnames,
                                         Coefficients=c(model1_summary[,"Estimate"][c(3,7,11)], 
                                                        model2_summary[,"Estimate"][c(3,7,11)]), 
                                         SEs=c(model1_summary[,"Std. Error"][c(3,7,11)], 
                                               model2_summary[,"Std. Error"][c(3,7,11)]),
                                         pvals=stars.pval(c(model1_summary[,"Pr(>|z|)"][c(3,7,11)], 
                                                            model2_summary[,"Pr(>|z|)"][c(3,7,11)])))

p3 <- ggplot(data=credit_assignment_results3, mapping = aes(x=VarNames, group=Dimension)) +
  geom_bar(mapping=aes(y=Coefficients, fill=Dimension), stat='identity', position=position_dodge(.9), 
           show.legend = FALSE, size=1.5, color='black') + 
  geom_errorbar(mapping=aes(ymin=Coefficients-SEs, ymax=Coefficients+SEs), color='black', 
                position=position_dodge(.9), width=0.25, linewidth=1.5, show.legend = FALSE) +
  geom_text(mapping=aes(y=Coefficients+sign(Coefficients)*(SEs+0.05)-0.04, label=pvals),
            position=position_dodge(.9), vjust='center', size=10) +
  theme(text = element_text(size = 25), axis.text = element_text(size = 25), axis.title.x = element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  scale_color_manual(breaks = c("Ft[man]", "Ft[non]", "Obj"), values=c('#4dbbd5', '#e64b35', 'grey50')) +
  scale_fill_manual(breaks = c("Ft[man]", "Ft[non]", "Obj"), values=c('#4dbbd5', '#e64b35', 'grey50'), 
                    labels = parse_format())+
  theme(legend.text = element_text(size=25), legend.text.align = 0, 
        legend.title=element_blank(), legend.position = c(0.9, 0.9),
        axis.line=element_line(size=1), axis.ticks=element_line(size=1), 
        axis.ticks.length=unit(0.1,"inch"))+
  ylim(c(-0.8, 0.8))+
  ylab('CA')


# plot coefficients, unchosen, ca
credit_assignment_results4 <- data.frame(PlotOrder=1:6,
                                         Dimension=dimnames, 
                                         VarNames=varnames,
                                         Coefficients=c(model1_summary[,"Estimate"][c(5,9,13)], 
                                                        model2_summary[,"Estimate"][c(5,9,13)]), 
                                         SEs=c(model1_summary[,"Std. Error"][c(5,9,13)], 
                                               model2_summary[,"Std. Error"][c(5,9,13)]),
                                         pvals=stars.pval(c(model1_summary[,"Pr(>|z|)"][c(5,9,13)], 
                                                            model2_summary[,"Pr(>|z|)"][c(5,9,13)])))

p4 <- ggplot(data=credit_assignment_results4, mapping = aes(x=VarNames, group=Dimension)) +
  geom_bar(mapping=aes(y=Coefficients, fill=Dimension), stat='identity', position=position_dodge(.9), 
           show.legend = TRUE, size=1.5, color='black') + 
  geom_errorbar(mapping=aes(ymin=Coefficients-SEs, ymax=Coefficients+SEs), color='black', 
                position=position_dodge(.9), width=0.25, linewidth=1.5, show.legend = FALSE) +
  geom_text(mapping=aes(y=Coefficients+sign(Coefficients)*(SEs+0.05)-0.1, label=pvals),
            position=position_dodge(.9), vjust='center', size=10) +
  theme(text = element_text(size = 25), axis.text = element_text(size = 25), axis.title.x = element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  scale_color_manual(breaks = c("Ft[man]", "Ft[non]", "Obj"), values=c('#4dbbd5', '#e64b35', 'grey50')) +
  scale_fill_manual(breaks = c("Ft[man]", "Ft[non]", "Obj"), values=c('#4dbbd5', '#e64b35', 'grey50'), 
                    labels = parse_format())+
  theme(legend.text = element_text(size=25), legend.text.align = 0, 
        legend.direction = 'horizontal',
        legend.title=element_blank(), legend.position = c(0.5, 0.9),
        axis.line=element_line(size=1), axis.ticks=element_line(size=1), 
        axis.ticks.length=unit(0.1,"inch"), axis.title.y=element_blank(), )+
  ylim(c(-0.8, 0.8))



p1+plot_spacer()+p2+
  plot_spacer()+plot_spacer()+plot_spacer()+
  p3+plot_spacer()+p4+
  plot_layout(widths = c(4, 0.5 ,4), heights=c(4, 1.5, 4))


ggsave(filename="credit_assignment_lme.pdf", path=figure_data_dir, device='pdf',
       width=12, height=7)

