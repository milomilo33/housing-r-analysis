setwd("F:/Fakultet/5.godina-master/RVPII/Projekat/housing-r-analysis")
getwd()

existing_csv_path <- "F:/Fakultet/5.godina-master/RVPII/Projekat/housing-r-analysis/workspace/uk-housing-official-1995-to-2023.csv"

new_csv_path <- "F:/Fakultet/5.godina-master/RVPII/Projekat/housing-r-analysis/workspace/subset.csv"

# Define the number of rows to extract
N <- 1500000

existing_data <- read.csv(existing_csv_path)

# Select the first N rows
new_data <- existing_data[1:N, ]

# Write the selected rows to a new CSV file
write.csv(new_data, file = new_csv_path, row.names = FALSE)