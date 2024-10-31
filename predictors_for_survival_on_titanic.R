library(tidyverse)
library(corrplot)
library(dplyr)


# Step: Data import #################################################################################
titanic_data <- read_csv("titanic.csv")

# Step: Data cleaning and pre-processing ############################################################
titanic_data <- titanic_data |>
  select(-c(PassengerId, Name, Ticket, Cabin))

titanic_data <- titanic_data |> 
  mutate(
    Survived = as.factor(Survived),
    Pclass = as.factor(Pclass),
    Sex = as.factor(Sex),
    Embarked = as.factor(Embarked)
  )

# Step: Data exploration - Correlation between numerical values ######################################
titanic_data_num <- titanic_data |> select_if(is.numeric)
correlations <- cor(titanic_data_num, use = "pairwise.complete.obs")

correlations |>
  corrplot(
    method = "color",
    addCoef.col = "black",    # Color of the correlation coefficients
    number.cex = 0.7,         # Size of the numbers
    tl.col = "black",         # Color of labels
    tl.srt = 45,             # Rotate labels 45 degrees
    diag = TRUE              # Show diagonal
  )

# Step: Data exploration - Plots to get overview of data #############################################

# Histogram of age of passengers
ggplot(
  data = titanic_data,
  mapping = aes(x = Age)
) +
  geom_histogram()

# Plotting distribution of Pclass, Sex and Survived in Titanic Dataset
plot_data <- titanic_data %>%
  summarise(across(c(Survived, Sex, Pclass), ~ list(prop.table(table(.))))) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Percentages") %>%
  unnest_longer(Percentages) %>%
  mutate(Percentage = Percentages * 100)

plot_data <- plot_data %>%
  mutate(Category = case_when(
    Variable == "Survived" & Percentages_id == "0" ~ "No",
    Variable == "Survived" & Percentages_id == "1" ~ "Yes",
    Variable == "Sex" & Percentages_id == "male" ~ "Male",
    Variable == "Sex" & Percentages_id == "female" ~ "Female",
    TRUE ~ as.character(Percentages_id)
  ))

ggplot(plot_data, aes(x = Variable, y = Percentage, fill = Category)) +
  geom_bar(stat = "identity", position = "stack") +
  geom_text(aes(label = sprintf("%s: %.1f%%", Category, Percentage)), 
            position = position_stack(vjust = 0.5),
            size = 3, color = "white", fontface = "bold") +
  scale_y_continuous(labels = scales::percent_format(scale = 1)) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Distribution of Pclass, Sex and Survived in Titanic Dataset",
       y = "Percentage") +
  theme_minimal() +
  theme(axis.title.x = element_blank(),
        legend.position = "none")  # Remove the legend

# Step: Creating model ###############################################################################
logistic_model <- glm(Survived ~ Pclass + Sex + Age + SibSp, data = titanic_data, family = "binomial")
summary(logistic_model)
summary(logistic_model)$aic

# Step: Evaluating the created modeln ################################################################
predicted_probabilities_for_survival <- predict(logistic_model, newdata = titanic_data, type = "response")
predictied_survivers <- ifelse(predicted_probabilities_for_survival > 0.5, 1, 0)

correct_predictions <- sum(predictied_survivers == titanic_data$Survived, na.rm = TRUE)
total_predictions <- sum(!is.na(predictied_survivers))
accuracy <- correct_predictions / total_predictions
