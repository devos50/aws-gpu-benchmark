# Load necessary libraries
library(ggplot2)
library(dplyr)

# Read the CSV file
data <- read.csv("../data/local_steps_time.csv")

# Calculate the average time and standard deviation for each model and GPU type
summary_data <- data %>%
    group_by(gpu, model, dataset, batch_size) %>%
    summarise(
        avg_time = mean(step_time),
        sd_time = sd(step_time)
    )

summary_data$model <- factor(summary_data$model, levels = c("resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "vit-base-patch16-224", "vit-large-patch16-224"))

# Create the bar plot
p <- ggplot(summary_data, aes(x = model, y = avg_time, fill = gpu)) +
     geom_bar(stat = "identity", position = position_dodge()) +
     geom_errorbar(aes(ymin = avg_time - sd_time, ymax = avg_time + sd_time), width = 0.2, position = position_dodge(0.9)) +
     labs(y = "Local Step Time [s]") +
     theme_bw() +
     theme(
         legend.position = c(0.2, 0.6),
         axis.text.x = element_text(angle = 45, hjust = 1),
         legend.background = element_rect(color = "black", fill = "white")
     )

# Save the plot to a PDF file
ggsave("../data/local_step_time.pdf", p, width=6, height=3)