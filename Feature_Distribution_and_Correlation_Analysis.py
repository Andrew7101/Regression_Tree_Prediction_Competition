import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training and test datasets
train1 = pd.read_csv('/path/to/train_data1.csv')
train2 = pd.read_csv('/path/to/train_data2.csv')
train3 = pd.read_csv('/path/to/train_data_small.csv')
test = pd.read_csv('/path/to/test_data.csv')

# Define the features to compare
features = ['city_fuel_economy', 'engine_displacement', 'highway_fuel_economy', 'horsepower', 'mileage', 'year']

# Dictionary of different training datasets for easy iteration
training_datasets = {
    'Training 1': train1,
    'Training 2': train2,
    'Training Small': train3
}

# Function to plot feature distributions in training vs. test datasets
def plot_distributions(training_name, train_df, test_df, features):
    num_features = len(features)
    cols = 3
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    fig.suptitle(f'Distribution Comparison: {training_name} vs Test', fontsize=16)
    
    for i, feature in enumerate(features):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # Plot training data distribution for each feature
        sns.kdeplot(
            train_df[feature],
            label=training_name,
            shade=True,
            color='blue',
            ax=ax
        )
        
        # Plot test data distribution for the same feature
        sns.kdeplot(
            test_df[feature],
            label='Test',
            shade=True,
            color='orange',
            ax=ax
        )
        
        ax.set_title(feature.replace('_', ' ').title())
        ax.legend()
    
    # Remove any empty subplots if the number of features is less than the grid slots
    total_subplots = rows * cols
    if num_features < total_subplots:
        for j in range(num_features, total_subplots):
            fig.delaxes(axes.flatten()[j])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Generate plots for each training dataset against the test dataset
for name, df in training_datasets.items():
    plot_distributions(name, df, test, features)

# Log-transform the price for consistency with Q1
train3['log_price'] = np.log(train3['price'])

# Calculate the correlation matrix between features and the target variable
correlation_matrix = train3[features + ['log_price']].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('The Correlation between each feature and the response variable')
plt.show()
