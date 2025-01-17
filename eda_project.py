import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Global_Superstore(CSV).csv')

# Display the first few rows of the dataset
print("Dataset Overview:")
print(df.head())

# Step 1: Clean the Data
# Handle missing values
print("\nHandling Missing Values:")
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Fill missing values with median for numeric columns
df = df.fillna(df.median(numeric_only=True))

# Drop rows with missing values for categorical columns
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()
print("\nShape of dataset after cleaning:", df.shape)

# Step 2: Statistical Analysis
# Select numeric columns
numeric_columns = df.select_dtypes(include=['number'])

# Basic statistics
print("\nBasic Statistics for Numeric Columns:")
print(numeric_columns.describe())

# Compute correlations
print("\nCorrelation Matrix:")
correlation_matrix = numeric_columns.corr()
print(correlation_matrix)

# Step 3: Detect and Handle Outliers
# Compute Q1, Q3, and IQR for numeric columns
Q1 = numeric_columns.quantile(0.25)
Q3 = numeric_columns.quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
filtered_df = numeric_columns[~((numeric_columns < lower_bound) | (numeric_columns > upper_bound)).any(axis=1)]
print("\nShape of dataset after removing outliers:", filtered_df.shape)

# Step 4: Data Visualization
# Histogram for numeric columns
print("\nGenerating Histograms...")
for col in numeric_columns.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(numeric_columns[col], kde=True, bins=30, color='blue')
    plt.title(f"Histogram for {col}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Boxplots to visualize outliers
print("\nGenerating Boxplots...")
for col in numeric_columns.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=numeric_columns[col], color='orange')
    plt.title(f"Boxplot for {col}")
    plt.ylabel(col)
    plt.show()

# Heatmap for correlations
print("\nGenerating Heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap of Correlations")
plt.show()
