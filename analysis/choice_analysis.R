library(lme4)
library(lmerTest)
library(tidyverse)
library(effectsize)
library(car)
library(scales)
library(flextable)

processed_data_dir <- "~/Documents/Attn_MdPRL/Py-attention-project-analysis/data/Processed"
figure_data_dir <- "~/Documents/Attn_MdPRL/Py-attention-project-analysis/figures/All Processed"
table_data_dir <- "~/Documents/Attn_MdPRL/Py-attention-project-analysis/tables/All Processed"

choice_curve_data <- read.csv(file.path(processed_data_dir, "choice_curve_df.csv"))
# choice_curve_data <- choice_curve_data %>% filter(block>0)
choice_curve_data$block <- scale(choice_curve_data$block)
# choice_curve_data <- choice_curve_data %>% mutate(pFavg = (pFinf+pFnoninf))

model <- glmer(prob ~ block*(pFinf+pFnoninf+pO) + (block*(pFinf+pFnoninf+pO)||subj), 
              data=choice_curve_data, family = binomial(link = "logit"),
              control=glmerControl(optimizer = "bobyqa"))
# model_sim <- glmer(prob ~ (pFinf+pFnoninf+pO) + (pFinf+pFnoninf+pO-1||subj), 
#                data=choice_curve_data, family = binomial(link = "logit"), 
#                control=glmerControl(optimizer = "bobyqa"))
# anova(model, model_sim)
summary(model)
linearHypothesis(model, "pFinf-pFnoninf=0", verbose=TRUE)
# linearHypothesis(model, "block:pFinf-block:pFnoninf=0", verbose=TRUE)

sigmoid <- function(x) {1/(1+exp(-x))}

ggplot(data=choice_curve_data, mapping=aes(y=prob)) +
  geom_function(fun = \(x) sigmoid(coef(summary(model))['pFinf','Estimate']*x), color='#4dbbd5', linewidth=2) + 
  geom_function(fun = \(x) sigmoid(coef(summary(model))['pFnoninf','Estimate']*x), color='#e64b35', linewidth=2) +
  stat_summary(aes(x=pFinf, color="Ft[m]"), fun.data = mean_se,  geom = "errorbar",  width=0.05, linewidth=2) +
  stat_summary(aes(x=pFinf, color="Ft[m]"), fun = "mean", geom='point', shape=21, fill='white', size=3, stroke=2) +
  stat_summary(aes(x=pFnoninf, color="Ft[n]"), fun.data = mean_se,  geom = "errorbar", width=0.05, linewidth=2) +
  stat_summary(aes(x=pFnoninf, color="Ft[n]"), fun = "mean", geom='point',  shape=21, fill='white', size=3, stroke=2) +
  theme(text = element_text(size=25), axis.text = element_text(size=25),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))+
  labs(color='Feature')+
  scale_x_continuous(limits=c(-1.1, 1.1), name='Log ratio of feature values') + 
  scale_y_continuous(name='Choice probability') +
  scale_color_manual(breaks = c("Ft[m]", "Ft[n]"), 
                     values=c('#4dbbd5', '#e64b35'),
                     name='', labels = parse_format()) +
  theme(legend.position = "none", legend.direction = 'horizontal', legend.text = element_text(size=25),
        axis.line=element_line(size=1), axis.ticks=element_line(size=1), axis.ticks.length=unit(0.1,"inch"))
  
ggsave(filename="choice_curves_slope_lme.pdf", path=figure_data_dir, device='pdf', 
       width=5.5, height=5.5)

results_df <- data.frame(summary(model)$coefficients)
names(results_df) <- c('b', 'SE', 'z', 'p')
results_df <- results_df %>%
  add_column(Predictor=c('Intercept', 'Block', 'F0 value', 'F1 value', 'O value', 
                         'Block * F0 value', 'Block * F1 value', 'Block * O vale'),
             .before=0) %>%
  mutate(p=if_else(p<0.001, '<0.001', as.character(round(p, 3))))
flextable(results_df) %>% 
  colformat_double(j=c('b', 'SE', 'z'), digits=2) %>% 
  colformat_double(j=c('p'), digits=3) %>%
  width(width=1.3) %>%
  italic(part='header') %>%
  align(align='center', part='all') %>%
  line_spacing(space=1.2) %>%
  padding(padding = 0, part = "body") %>% 
  save_as_html(path=file.path(table_data_dir, "choice_curves_slope_lme.html"))

