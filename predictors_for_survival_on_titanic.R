library(tidyverse)  # includes dplyr, ggplot2, etc.
library(corrplot)


# Step: Data import #################################################################################
titanic_data <- read_csv("titanic.csv")

# Step: Data cleaning and pre-processing ############################################################
titanic_data <- titanic_data |>
  select(-c(PassengerId, Name, Ticket, Cabin)) |>
  mutate(
    Survived = as.factor(Survived),
    Pclass = as.factor(Pclass),
    Sex = as.factor(Sex),
    Embarked = as.factor(Embarked)
  )

# Step: Data exploration - Correlation between numerical values ######################################
titanic_data_num <- titanic_data |> select(where(is.numeric))
correlations <- cor(titanic_data_num, use = "pairwise.complete.obs")

correlations |>
  corrplot(
    method = "color",
    addCoef.col = "black",    # Color of the correlation coefficients
    number.cex = 0.7,         # Size of the numbers
    tl.col = "black",         # Color of labels
    tl.srt = 45,              # Rotate labels 45 degrees
    diag = TRUE               # Show diagonal
  )

# Step: Data exploration - Plots to get overview of data #############################################

# Histogram of age of passengers
ggplot(
  data = titanic_data,
  mapping = aes(x = Age)
) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  labs(title = "Age Distribution of Passengers",
       x = "Age",
       y = "Count") +
  theme_minimal()

# Plotting distribution of Pclass, Sex and Survived in Titanic Dataset
plot_data <- titanic_data |>
  summarise(across(c(Survived, Sex, Pclass), ~ list(prop.table(table(.))))) |>
  pivot_longer(everything(), names_to = "Variable", values_to = "Percentages") |>
  unnest_longer(Percentages) |>
  mutate(
    Percentage = Percentages * 100,
    Category = case_when(
      Variable == "Survived" & Percentages_id == "0" ~ "No",
      Variable == "Survived" & Percentages_id == "1" ~ "Yes",
      Variable == "Sex" & Percentages_id == "male" ~ "Male",
      Variable == "Sex" & Percentages_id == "female" ~ "Female",
      TRUE ~ as.character(Percentages_id)
    )
  )

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

# Step: Train/Test Split ############################################################################
# Remove rows with missing values in the variables we'll use
titanic_clean <- titanic_data |>
  filter(!is.na(Age), !is.na(Survived), !is.na(Sex), !is.na(Pclass), !is.na(SibSp))

# Set seed for reproducibility
set.seed(123)

# Split data: 80% training, 20% testing
sample_size <- floor(0.8 * nrow(titanic_clean))
train_indices <- sample(seq_len(nrow(titanic_clean)), size = sample_size)

train_data <- titanic_clean[train_indices, ]
test_data <- titanic_clean[-train_indices, ]

cat("\nTraining set size:", nrow(train_data))
cat("\nTest set size:", nrow(test_data), "\n")

# Step: Creating model ###############################################################################
logistic_model <- glm(Survived ~ Pclass + Sex + Age + SibSp, 
                      data = train_data, 
                      family = "binomial")
model_summary <- summary(logistic_model)
print(model_summary)
cat("\nAIC:", model_summary$aic, "\n")

# Step: Evaluating the created model #################################################################
# Predict on TEST data (not training data)
predicted_probabilities_test <- predict(logistic_model, 
                                        newdata = test_data, 
                                        type = "response")
predicted_survivors_test <- ifelse(predicted_probabilities_test > 0.5, 1, 0)

# Calculate test accuracy
correct_predictions_test <- sum(predicted_survivors_test == 
                                 as.numeric(as.character(test_data$Survived)))
test_accuracy <- correct_predictions_test / nrow(test_data)

# Also calculate training accuracy for comparison
predicted_probabilities_train <- predict(logistic_model, 
                                         newdata = train_data, 
                                         type = "response")
predicted_survivors_train <- ifelse(predicted_probabilities_train > 0.5, 1, 0)

correct_predictions_train <- sum(predicted_survivors_train == 
                                  as.numeric(as.character(train_data$Survived)))
train_accuracy <- correct_predictions_train / nrow(train_data)

cat("\n=== Model Performance ===")
cat("\nTraining Accuracy:", round(train_accuracy * 100, 2), "%")
cat("\nTest Accuracy:", round(test_accuracy * 100, 2), "%")
cat("\nDifference:", round((train_accuracy - test_accuracy) * 100, 2), "percentage points")
cat("\n\nTest set: Correct predictions:", correct_predictions_test, "out of", nrow(test_data), "\n")
