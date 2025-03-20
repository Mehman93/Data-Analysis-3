import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Function to plot time series
def plot_time_series(data):
    """Plots time series for v1 and v2."""
    plt.figure(figsize=(6, 2))
    
    plt.subplot(2, 1, 1)
    plt.plot(data['v1'], label='v1', color='blue')
    plt.title('v1 Time Series')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(data['v2'], label='v2', color='red')
    plt.title('v2 Time Series')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Function to analyze correlation
def plot_correlation(data):
    """Plots scatter plot and calculates correlation between v1 and v2."""
    plt.figure(figsize=(10, 2))
    plt.scatter(data['v1'], data['v2'], alpha=0.5)
    plt.title('Correlation between v1 and v2')
    plt.xlabel('v1')
    plt.ylabel('v2')
    plt.grid(True)
    
    correlation = data['v1'].corr(data['v2'])
    plt.annotate(f'Correlation: {correlation:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
    
    # plt.savefig('correlation_plot.png')
    plt.show()
    
    print(f"\n✅ Correlation between v1 and v2: {correlation:.4f}")
    # print("✅ Correlation plot saved as 'correlation_plot.png'.")

# Function to analyze seasonality by hour
def plot_hourly_seasonality(data):
    """Plots average v1 and v2 by hour."""
    data['hour'] = data.index.hour
    hourly_v1 = data.groupby('hour')['v1'].mean()
    hourly_v2 = data.groupby('hour')['v2'].mean()
    
    plt.figure(figsize=(6, 2))
    
    plt.subplot(1, 2, 1)
    hourly_v1.plot(kind='bar', color='blue')
    plt.title('Average v1 by Hour')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    hourly_v2.plot(kind='bar', color='red')
    plt.title('Average v2 by Hour')
    plt.grid(True)
    
    plt.tight_layout()
    
    # plt.savefig('correlation_plot.png')
    plt.show()
    # print("✅ Hourly seasonality plot saved as 'hourly_patterns.png'.")

# Function to analyze seasonality by day of the week
def plot_weekly_seasonality(data):
    """Plots average v1 and v2 by day of the week."""
    data['dayofweek'] = data.index.dayofweek
    weekly_v1 = data.groupby('dayofweek')['v1'].mean()
    weekly_v2 = data.groupby('dayofweek')['v2'].mean()
    
    plt.figure(figsize=(6, 2))
    
    plt.subplot(1, 2, 1)
    weekly_v1.plot(kind='bar', color='blue')
    plt.title('Average v1 by Day of Week')
    plt.grid(True)
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    plt.subplot(1, 2, 2)
    weekly_v2.plot(kind='bar', color='red')
    plt.title('Average v2 by Day of Week')
    plt.grid(True)
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    plt.tight_layout()
    # plt.savefig('weekly_patterns.png')
    plt.show()
    
    # print("✅ Weekly seasonality plot saved as 'weekly_patterns.png'.")

# Function to analyze seasonality by month
def plot_monthly_seasonality(data):
    """Plots average v1 and v2 by month."""
    data['month'] = data.index.month
    monthly_v1 = data.groupby('month')['v1'].mean()
    monthly_v2 = data.groupby('month')['v2'].mean()
    
    plt.figure(figsize=(6, 2))
    
    plt.subplot(1, 2, 1)
    monthly_v1.plot(kind='bar', color='blue')
    plt.title('Average v1 by Month')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    monthly_v2.plot(kind='bar', color='red')
    plt.title('Average v2 by Month')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    # print("✅ Monthly seasonality plot saved as 'monthly_patterns.png'.")

# Function to check autocorrelation and partial autocorrelation
def plot_autocorrelation(data):
    """Plots ACF and PACF for v1 and v2."""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(2, 2, 1)
    plot_acf(data['v1'], lags=48, ax=plt.gca())
    plt.title('ACF for v1')

    plt.subplot(2, 2, 2)
    plot_pacf(data['v1'], lags=48, ax=plt.gca())
    plt.title('PACF for v1')

    plt.subplot(2, 2, 3)
    plot_acf(data['v2'], lags=48, ax=plt.gca())
    plt.title('ACF for v2')

    plt.subplot(2, 2, 4)
    plot_pacf(data['v2'], lags=48, ax=plt.gca())
    plt.title('PACF for v2')

    plt.tight_layout()
    # plt.savefig('autocorrelation.png')
    plt.show()
    
    # print("✅ Autocorrelation plots saved as 'autocorrelation.png'.")

# Boxplot to Detect Outliers & Display Summary as a DataFrame
def plot_outlier(data):
    """Plots a boxplot for v1 and v2, detects outliers, counts them, and returns a DataFrame summary."""
    
    # Plot Boxplot
    plt.figure(figsize=(6, 2))
    sns.boxplot(data=data[['v1', 'v2']], palette=["blue", "red"])
    plt.title("Boxplot of v1 and v2")
    plt.ylabel("Values")
    plt.show()

    # Initialize Dictionary to Store Outlier Info
    outlier_summary = []

    # Detect Outliers for Each Column
    for col in ['v1', 'v2']:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify Outliers
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
        total_outliers = len(outliers)

        # Append Summary to List
        outlier_summary.append({
            "Column": col,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Total Outliers": total_outliers
        })

    # Convert to DataFrame
    outlier_df = pd.DataFrame(outlier_summary)

    # Display Summary
    print("Outlier Summary: \n")
    print(outlier_df)

    return outlier_df  # Return DataFrame with summary info


# ✅ Function to Plot Monthly Boxplots for Each Column
def plot_monthly_boxplots(data):
    """Plots boxplots for each month separately for v1 and v2."""
    
    # ✅ Extract Month Name
    data['Month'] = data.index.strftime('%B')  # Convert month number to name
    month_order = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]

    # ✅ Plot Boxplots for v1
    plt.figure(figsize=(6, 2))
    sns.boxplot(x=data["Month"], y=data["v1"], order=month_order, palette="Blues")
    plt.title("Monthly Boxplot of v1")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # ✅ Plot Boxplots for v2
    plt.figure(figsize=(6, 2))
    sns.boxplot(x=data["Month"], y=data["v2"], order=month_order, palette="Reds")
    plt.title("Monthly Boxplot of v2")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


# Main execution
if __name__ == "__main__":
    # Load dataset (assuming data is already available)
    data = pd.read_csv("DA3_Ass3/data.csv")

    # Convert Date column to datetime format and set as index
    data["Date"] = pd.to_datetime(data["Date"], format="%d-%m-%y %H:%M")
    data = data.set_index("Date")
    # Run all functions
    plot_outlier(data)
    plot_monthly_boxplots(data)
    plot_time_series(data)
    plot_correlation(data)
    plot_hourly_seasonality(data)
    plot_weekly_seasonality(data)
    plot_monthly_seasonality(data)

    
