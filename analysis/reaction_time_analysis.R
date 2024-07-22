library(lme4)
library(lmerTest)
library(tidyverse)
library(patchwork)
library(effectsize)
library(car)
library(gtools)

processed_data_dir <- "~/Documents/Attn_MdPRL/Py-attention-project-analysis/data/Processed"
figure_data_dir <- "~/Documents/Attn_MdPRL/Py-attention-project-analysis/figures/All Processed"

all_rt_data <- read.csv(file.path(processed_data_dir, "reaction_time_analysis_df.csv"))
set_contrasts <- function(x) contrasts(x)<-contr.sum(length(levels(x)))

ggplot(data=all_rt_data, mapping=aes(x=trial, y=rts)) +
  stat_summary(fun.data="mean_se") + 
  stat_summary(fun.data="mean_se", geom='line')

all_rt_data <- all_rt_data %>% 
  filter(trial>1 & trial%%8!=0) %>%
  mutate_at(c('past_rew_ch', 'past_rew_unch', 'past_Finf_ch', 'past_Finf_unch',
              'past_Fnoninf_ch', 'past_Fnoninf_unch', 'past_O_ch', 'past_O_unch'),
            as.factor) 
  

rt_model <- lmer(log(rts)~scale(trial)*(abs(pFinf)+abs(pFnoninf)+abs(pO)+
                               past_rew_ch*(past_Finf_ch+past_Fnoninf_ch+past_O_ch)+
                               past_rew_unch*(past_Finf_unch+past_Fnoninf_unch+past_O_unch))+
                         (scale(trial)|subj),
                 data=all_rt_data, REML=F,
                 contrasts = list(past_rew_ch=contr.sum, past_rew_unch=contr.sum,
                                  past_Finf_ch=contr.sum, past_Finf_unch=contr.sum,
                                  past_Fnoninf_ch=contr.sum, past_Fnoninf_unch=contr.sum,
                                  past_O_ch=contr.sum, past_O_unch=contr.sum))

# rt_model <- glmer(rts/1000~scale(trial)*(pFinf+pFnoninf+pO+
#                                past_rew_ch*(past_Finf_ch+past_Fnoninf_ch+past_O_ch)+
#                                past_rew_unch*(past_Finf_unch+past_Fnoninf_unch+past_O_unch))+(1|subj),
#                  data=all_rt_data, family=Gamma('inverse'), 
#                  contrasts = list(past_rew_ch=contr.sum, past_rew_unch=contr.sum,
#                                   past_Finf_ch=contr.sum, past_Finf_unch=contr.sum,
#                                   past_Fnoninf_ch=contr.sum, past_Fnoninf_unch=contr.sum,
#                                   past_O_ch=contr.sum, past_O_unch=contr.sum))
summary(rt_model)
car::Anova(rt_model, type='3')



ggplot(data=all_rt_data, mapping=aes(y=rts)) +
  stat_summary(mapping=aes(x=pFinf), fun.data="mean_se", color='deepskyblue') + 
  stat_summary(geom='line', fun='mean', mapping=aes(x=pFinf), color='deepskyblue') + 
  stat_summary(mapping=aes(x=pFnoninf), fun.data="mean_se", color='darkorange') + 
  stat_summary(geom='line', fun='mean', mapping=aes(x=pFnoninf), color='darkorange') +
  stat_summary(mapping=aes(x=pO), fun.data="mean_se", color='green3') + 
  stat_summary(geom='line', fun='mean', mapping=aes(x=pO), color='green3') +
  theme(text = element_text(size = 22), axis.text = element_text(size = 22),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) + 
  scale_x_continuous(name='Log odds of reward')

p1 <- ggplot(data=all_rt_data, mapping=aes(x=past_Finf_ch, y=rts, group=past_rew_ch, linetype=past_rew_ch, shape=past_rew_ch)) + 
  stat_summary(fun.data="mean_se", color='deepskyblue') +
  stat_summary(geom='line', fun='mean', color='deepskyblue') + 
  theme(legend.position = "none") 
p2 <- ggplot(data=all_rt_data, mapping=aes(x=past_Fnoninf_ch, y=rts, group=past_rew_ch, linetype=past_rew_ch, shape=past_rew_ch)) + 
  stat_summary(fun.data="mean_se", color='darkorange') +
  stat_summary(geom='line', fun='mean', color='darkorange') + 
  theme(legend.position = "none") 
p3 <- ggplot(data=all_rt_data, mapping=aes(x=past_O_ch, y=rts, group=past_rew_ch, linetype=past_rew_ch, shape=past_rew_ch)) + 
  stat_summary(fun.data="mean_se", color='green3') +
  stat_summary(geom='line', fun='mean', color='green3') + 
  theme(legend.position = "none") 
p4 <- ggplot(data=all_rt_data, mapping=aes(x=past_Finf_unch, y=rts, group=past_rew_unch, linetype=past_rew_unch, shape=past_rew_unch)) + 
  stat_summary(fun.data="mean_se", color='deepskyblue') +
  stat_summary(geom='line', fun='mean', color='deepskyblue') + 
  theme(legend.position = "none") 
p5 <- ggplot(data=all_rt_data, mapping=aes(x=past_Fnoninf_unch, y=rts, group=past_rew_unch, linetype=past_rew_unch, shape=past_rew_unch)) + 
  stat_summary(fun.data="mean_se", color='darkorange') +
  stat_summary(geom='line', fun='mean', color='darkorange') + 
  theme(legend.position = "none") 
p6 <- ggplot(data=all_rt_data, mapping=aes(x=past_O_unch, y=rts, group=past_rew_unch, linetype=past_rew_unch, shape=past_rew_unch)) + 
  stat_summary(fun.data="mean_se", color='green3') +
  stat_summary(geom='line', fun='mean', color='green3') + 
  theme(legend.position = "none") 
p1+p2+p3+p4+p5+p6



