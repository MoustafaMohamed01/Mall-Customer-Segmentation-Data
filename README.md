# Customer Segmentation using K-Means Clustering

## Overview
This project performs customer segmentation using the K-Means clustering algorithm. It applies unsupervised learning to categorize customers based on their annual income and spending score.

## Dataset
The project uses the **Mall_Customers.csv** dataset, which contains the following features:
- **CustomerID**: Unique ID assigned to each customer.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Annual Income (K$)**: Annual income of the customer in thousands.
- **Spending Score (1-100)**: Score assigned by the mall based on customer behavior and spending patterns.

Dataset Source: [Kaggle - Mall Customers Dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python).

## Project Structure
- **K-Means Clustering**: 
  - Determines the optimal number of clusters using the Elbow Method.
  - Applies K-Means clustering to segment customers.
  - Visualizes the clusters with a scatter plot.

## Installation & Setup
### Prerequisites
Ensure you have Python installed along with the required dependencies:
```sh
pip install numpy pandas seaborn matplotlib scikit-learn
```

### Running the Script
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/customer-segmentation.git
   cd customer-segmentation
   ```
2. Place the **Mall_Customers.csv** file in the same directory as the script.
3. Run the Python script:
   ```sh
   python customer_segmentation.py
   ```

## Methodology
1. **Load Data**: Reads the dataset into a Pandas DataFrame.
2. **Data Preprocessing**: Extracts relevant features (Annual Income and Spending Score).
3. **Finding Optimal Clusters**: Uses the **Elbow Method** to determine the best value of K.
4. **Clustering with K-Means**: Applies K-Means with the optimal K value.
5. **Visualization**: Plots the Elbow Graph and Cluster Scatter Plot.
