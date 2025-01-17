import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
file_path = 'sales_data.csv'  # Corrected to use local path instead of '/mnt/data/sales_data.csv'
data = pd.read_csv(file_path)

# Step 1: Rename 'Revenue' to 'Sales'
data.rename(columns={'Revenue': 'Sales'}, inplace=True)

# Step 2: Ensure 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Step 3: Plot Sales Trend
def plot_sales_trend():
    plt.figure(figsize=(10, 6))
    data.sort_values('Date').groupby('Date')['Sales'].sum().plot()
    plt.title('Sales Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.grid(True)
    plt.show()

# Step 4: Plot Profit vs Discount (assuming Discount needs to be derived)
data['Discount'] = (data['Unit_Price'] - data['Unit_Cost']) / data['Unit_Price']

sns.scatterplot(x='Discount', y='Profit', data=data)
plt.title('Profit vs Discount')
plt.xlabel('Discount')
plt.ylabel('Profit')
plt.grid(True)
plt.show()

# Step 5: Model Training
dependent_vars = ['Sales', 'Profit', 'Discount']
if all(col in data.columns for col in dependent_vars):
    X = data[['Profit', 'Discount']]
    y = data['Sales']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Model Mean Squared Error: {mse}')
else:
    print("Cannot train model due to missing columns: 'Sales', 'Profit', or 'Discount'.")

# Execute the functions
plot_sales_trend()
