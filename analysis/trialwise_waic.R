library(lme4)
library(lmerTest)
library(tidyverse)
library(afex)
library(car)
library(flextable)

processed_data_dir <- "~/Documents/Attn_MdPRL/Py-attention-project-analysis/data/Processed"
figure_data_dir <- "~/Documents/Attn_MdPRL/Py-attention-project-analysis/figures/All Processed"

all_trialwise_waic_data <- read.csv(file.path(processed_data_dir, "trialwise_waic.csv"))

model <- lmer(dWAIC~trial+(trial-1||subj), data=all_trialwise_waic_data,
              control=lmerControl(optimizer="bobyqa"), REML=F)

print(summary(model))
