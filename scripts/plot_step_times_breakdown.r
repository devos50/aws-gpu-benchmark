library(ggplot2)
library(dplyr)
library(tidyr)

# Read the CSV file
data <- read.csv("../data/local_steps_time.csv")

# Reshape the data to long format for stacked bar plot
long_data <- data %>%
  select(gpu, model, dataset, batch_size, time_data, time_forward, time_backward) %>%
  pivot_longer(
    cols = starts_with("time_"),
    names_to = "component",
    values_to = "time"
  )

long_data$component <- factor(long_data$component, levels = c("time_backward", "time_forward", "time_data"))

# Calculate average time for each component, model, and GPU
summary_long_data <- long_data %>%
  group_by(gpu, model, dataset, batch_size, component) %>%
  summarise(
    avg_time = mean(time),
    .groups = "drop"
  )

# Optional: Order the models in a specific way
summary_long_data$model <- factor(summary_long_data$model, 
                                  levels = c("resnet18", "resnet34", "resnet50", 
                                             "resnet101", "resnet152", 
                                             "vit-base-patch16-224", "vit-large-patch16-224"))

# Create the stacked bar plot
p <- ggplot(summary_long_data, aes(x = model, y = avg_time, fill = component)) +
  geom_bar(stat = "identity") +
  labs(x = "Model", y = "Average Time [s]", fill = "Stage") +
  theme_bw() +
  scale_fill_discrete(name = "Component", labels = c("Backward Pass", "Forward Pass", "Data Loading")) +
  theme(legend.position = c(0.2, 0.6), axis.text.x = element_text(angle = 45, hjust = 1), legend.background = element_rect(color = "black", fill = "white"))

# Save the plot to a PDF file
ggsave("../data/local_step_time_breakdown.pdf", p, width = 8, height = 4)
