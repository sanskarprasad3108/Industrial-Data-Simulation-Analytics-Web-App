"""
Synthetic Data Generation and Analysis Project

This project generates a large synthetic dataset simulating e-commerce transactions
and performs comprehensive data analysis and visualization.

Author: Data Science Student
Date: July 22, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class SyntheticDataGenerator:
    """
    Class to generate synthetic e-commerce transaction data
    """
    
    def __init__(self, n_records=50000):
        self.n_records = n_records
        
    def generate_dataset(self):
        """
        Generate a comprehensive synthetic dataset with multiple data types
        """
        print(f"Generating {self.n_records} synthetic records...")
        
        # Generate customer IDs
        customer_ids = np.random.randint(1000, 9999, self.n_records)
        
        # Generate product categories
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 
                     'Beauty', 'Automotive', 'Food', 'Toys', 'Health']
        product_category = np.random.choice(categories, self.n_records, 
                                          p=[0.2, 0.15, 0.12, 0.08, 0.1, 0.08, 0.05, 0.07, 0.08, 0.07])
        
        # Generate prices based on category
        price_ranges = {
            'Electronics': (50, 2000),
            'Clothing': (20, 300),
            'Home & Garden': (15, 500),
            'Books': (5, 50),
            'Sports': (25, 800),
            'Beauty': (10, 150),
            'Automotive': (20, 1500),
            'Food': (5, 100),
            'Toys': (10, 200),
            'Health': (15, 300)
        }
        
        prices = []
        for category in product_category:
            min_price, max_price = price_ranges[category]
            # Use log-normal distribution for more realistic price distribution
            price = np.random.lognormal(np.log(min_price + 50), 0.5)
            price = np.clip(price, min_price, max_price)
            prices.append(round(price, 2))
        
        # Generate quantities (mostly 1-3, occasionally higher)
        quantities = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                    self.n_records, 
                                    p=[0.4, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.005])
        
        # Generate transaction dates (last 2 years with seasonality)
        start_date = datetime.now() - timedelta(days=730)
        date_range = []
        
        for i in range(self.n_records):
            # Add seasonality - more purchases in Nov-Dec, less in Jan-Feb
            month_weights = [0.7, 0.7, 0.9, 1.0, 1.0, 1.1, 1.0, 1.0, 1.0, 1.1, 1.4, 1.6]
            days_offset = np.random.randint(0, 730)
            base_date = start_date + timedelta(days=days_offset)
            month = base_date.month
            
            # Adjust probability based on month
            if np.random.random() < month_weights[month-1]/1.6:
                date_range.append(base_date)
            else:
                # Regenerate for different month
                days_offset = np.random.randint(0, 730)
                date_range.append(start_date + timedelta(days=days_offset))
        
        # Generate customer age groups
        age_groups = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        customer_age_group = np.random.choice(age_groups, self.n_records,
                                            p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])
        
        # Generate customer locations
        regions = ['North', 'South', 'East', 'West', 'Central']
        customer_region = np.random.choice(regions, self.n_records,
                                         p=[0.22, 0.20, 0.18, 0.25, 0.15])
        
        # Generate payment methods
        payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer', 'Cash on Delivery']
        payment_method = np.random.choice(payment_methods, self.n_records,
                                        p=[0.35, 0.25, 0.20, 0.15, 0.05])
        
        # Generate discount amounts (0-30%)
        discount_percent = np.random.exponential(scale=5, size=self.n_records)
        discount_percent = np.clip(discount_percent, 0, 30)
        
        # Generate shipping costs
        shipping_costs = []
        for price, region in zip(prices, customer_region):
            base_shipping = 5.99
            if region in ['East', 'West']:
                base_shipping += 2.0
            if price > 100:
                shipping_cost = 0  # Free shipping for orders over $100
            else:
                shipping_cost = base_shipping + np.random.normal(0, 1)
                shipping_cost = max(0, shipping_cost)
            shipping_costs.append(round(shipping_cost, 2))
        
        # Generate customer ratings (1-5 stars)
        ratings = np.random.choice([1, 2, 3, 4, 5], self.n_records,
                                 p=[0.05, 0.08, 0.15, 0.35, 0.37])
        
        # Generate boolean flags
        is_repeat_customer = np.random.choice([True, False], self.n_records, p=[0.65, 0.35])
        is_mobile_purchase = np.random.choice([True, False], self.n_records, p=[0.60, 0.40])
        
        # Calculate total amounts
        subtotal = np.array(prices) * np.array(quantities)
        discount_amount = subtotal * (np.array(discount_percent) / 100)
        total_amount = subtotal - discount_amount + np.array(shipping_costs)
        
        # Create DataFrame
        df = pd.DataFrame({
            'transaction_id': range(100000, 100000 + self.n_records),
            'customer_id': customer_ids,
            'transaction_date': date_range,
            'product_category': product_category,
            'unit_price': prices,
            'quantity': quantities,
            'subtotal': np.round(subtotal, 2),
            'discount_percent': np.round(discount_percent, 2),
            'discount_amount': np.round(discount_amount, 2),
            'shipping_cost': shipping_costs,
            'total_amount': np.round(total_amount, 2),
            'customer_age_group': customer_age_group,
            'customer_region': customer_region,
            'payment_method': payment_method,
            'rating': ratings,
            'is_repeat_customer': is_repeat_customer,
            'is_mobile_purchase': is_mobile_purchase
        })
        
        # Add derived features
        df['month'] = df['transaction_date'].dt.month
        df['day_of_week'] = df['transaction_date'].dt.day_name()
        df['quarter'] = df['transaction_date'].dt.quarter
        df['year'] = df['transaction_date'].dt.year
        
        print(f"Dataset generated successfully with {len(df)} records and {len(df.columns)} features!")
        return df

class DataAnalyzer:
    """
    Class to perform comprehensive data analysis and visualization
    """
    
    def __init__(self, df):
        self.df = df
        
    def basic_statistics(self):
        """
        Generate basic statistical summary of the dataset
        """
        print("=== BASIC DATASET STATISTICS ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print("\nData types:")
        print(self.df.dtypes.value_counts())
        print("\nMissing values:")
        print(self.df.isnull().sum().sum())
        print("\nNumerical features summary:")
        print(self.df.describe())
        
    def correlation_analysis(self):
        """
        Analyze correlations between numerical features
        """
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numerical_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append((correlation_matrix.columns[i], 
                                      correlation_matrix.columns[j], corr_val))
        
        print("Strong correlations (|r| > 0.5):")
        for feat1, feat2, corr in strong_corr:
            print(f"{feat1} - {feat2}: {corr:.3f}")
    
    def temporal_analysis(self):
        """
        Analyze trends and seasonality over time
        """
        print("\n=== TEMPORAL ANALYSIS ===")
        
        # Monthly trends
        monthly_sales = self.df.groupby(['year', 'month']).agg({
            'total_amount': ['sum', 'mean', 'count'],
            'rating': 'mean'
        }).round(2)
        
        # Create date column for plotting
        self.df['year_month'] = self.df['transaction_date'].dt.to_period('M')
        monthly_trends = self.df.groupby('year_month').agg({
            'total_amount': 'sum',
            'transaction_id': 'count'
        })
        
        # Plot temporal trends
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Monthly sales trend
        monthly_trends['total_amount'].plot(ax=axes[0,0], kind='line', marker='o')
        axes[0,0].set_title('Total Sales Amount Over Time')
        axes[0,0].set_ylabel('Total Sales ($)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Monthly transaction count
        monthly_trends['transaction_id'].plot(ax=axes[0,1], kind='line', marker='s', color='green')
        axes[0,1].set_title('Number of Transactions Over Time')
        axes[0,1].set_ylabel('Transaction Count')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Day of week analysis
        dow_sales = self.df.groupby('day_of_week')['total_amount'].sum().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        dow_sales.plot(ax=axes[1,0], kind='bar', color='orange')
        axes[1,0].set_title('Sales by Day of Week')
        axes[1,0].set_ylabel('Total Sales ($)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Quarterly analysis
        quarterly_sales = self.df.groupby('quarter')['total_amount'].sum()
        quarterly_sales.plot(ax=axes[1,1], kind='bar', color='purple')
        axes[1,1].set_title('Sales by Quarter')
        axes[1,1].set_ylabel('Total Sales ($)')
        
        plt.tight_layout()
        plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Seasonal insights:")
        print(f"Best performing month: {monthly_trends['total_amount'].idxmax()}")
        print(f"Best performing day: {dow_sales.idxmax()}")
        print(f"Best performing quarter: Q{quarterly_sales.idxmax()}")
    
    def category_analysis(self):
        """
        Analyze product categories and customer segments
        """
        print("\n=== CATEGORY & CUSTOMER ANALYSIS ===")
        
        # Category performance
        category_stats = self.df.groupby('product_category').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'rating': 'mean',
            'discount_percent': 'mean'
        }).round(2)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sales by category
        category_sales = self.df.groupby('product_category')['total_amount'].sum().sort_values(ascending=False)
        category_sales.plot(ax=axes[0,0], kind='bar')
        axes[0,0].set_title('Total Sales by Product Category')
        axes[0,0].set_ylabel('Total Sales ($)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Average rating by category
        category_ratings = self.df.groupby('product_category')['rating'].mean().sort_values(ascending=False)
        category_ratings.plot(ax=axes[0,1], kind='bar', color='green')
        axes[0,1].set_title('Average Rating by Product Category')
        axes[0,1].set_ylabel('Average Rating')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Regional analysis
        region_sales = self.df.groupby('customer_region')['total_amount'].sum()
        region_sales.plot(ax=axes[1,0], kind='pie', autopct='%1.1f%%')
        axes[1,0].set_title('Sales Distribution by Region')
        
        # Age group analysis
        age_sales = self.df.groupby('customer_age_group')['total_amount'].mean().sort_values(ascending=False)
        age_sales.plot(ax=axes[1,1], kind='bar', color='orange')
        axes[1,1].set_title('Average Purchase Amount by Age Group')
        axes[1,1].set_ylabel('Average Purchase ($)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('category_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Category insights:")
        print(f"Top category by sales: {category_sales.index[0]}")
        print(f"Highest rated category: {category_ratings.index[0]}")
        print(f"Most valuable age group: {age_sales.index[0]}")
    
    def distribution_analysis(self):
        """
        Analyze distributions of key features
        """
        print("\n=== DISTRIBUTION ANALYSIS ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Total amount distribution
        self.df['total_amount'].hist(bins=50, ax=axes[0,0], alpha=0.7)
        axes[0,0].set_title('Distribution of Total Amount')
        axes[0,0].set_xlabel('Total Amount ($)')
        axes[0,0].set_ylabel('Frequency')
        
        # Unit price distribution
        self.df['unit_price'].hist(bins=50, ax=axes[0,1], alpha=0.7, color='green')
        axes[0,1].set_title('Distribution of Unit Price')
        axes[0,1].set_xlabel('Unit Price ($)')
        axes[0,1].set_ylabel('Frequency')
        
        # Quantity distribution
        self.df['quantity'].hist(bins=20, ax=axes[0,2], alpha=0.7, color='orange')
        axes[0,2].set_title('Distribution of Quantity')
        axes[0,2].set_xlabel('Quantity')
        axes[0,2].set_ylabel('Frequency')
        
        # Rating distribution
        rating_counts = self.df['rating'].value_counts().sort_index()
        rating_counts.plot(ax=axes[1,0], kind='bar', color='purple')
        axes[1,0].set_title('Distribution of Ratings')
        axes[1,0].set_xlabel('Rating')
        axes[1,0].set_ylabel('Count')
        
        # Discount distribution
        self.df['discount_percent'].hist(bins=30, ax=axes[1,1], alpha=0.7, color='red')
        axes[1,1].set_title('Distribution of Discount Percentage')
        axes[1,1].set_xlabel('Discount (%)')
        axes[1,1].set_ylabel('Frequency')
        
        # Box plot for total amount by category (top 5)
        top_categories = self.df['product_category'].value_counts().head(5).index
        df_top = self.df[self.df['product_category'].isin(top_categories)]
        sns.boxplot(data=df_top, x='product_category', y='total_amount', ax=axes[1,2])
        axes[1,2].set_title('Total Amount Distribution by Top Categories')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def clustering_analysis(self):
        """
        Perform customer clustering analysis
        """
        print("\n=== CLUSTERING ANALYSIS ===")
        
        # Prepare features for clustering
        customer_features = self.df.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'rating': 'mean',
            'discount_percent': 'mean',
            'is_repeat_customer': 'first',
            'is_mobile_purchase': 'mean'
        }).round(2)
        
        customer_features.columns = ['total_spent', 'avg_order_value', 'order_frequency', 
                                   'avg_rating', 'avg_discount', 'is_repeat', 'mobile_usage']
        
        # Prepare data for clustering
        features_for_clustering = customer_features[['total_spent', 'avg_order_value', 
                                                   'order_frequency', 'avg_rating']].copy()
        
        # Handle any missing values
        features_for_clustering = features_for_clustering.fillna(features_for_clustering.mean())
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_for_clustering)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        K_range = range(2, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features_scaled)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(K_range, inertias, 'bo-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        
        # Perform clustering with k=4
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to customer features
        customer_features['cluster'] = clusters
        
        # Visualize clusters using PCA
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis')
        plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Customer Clusters (PCA Visualization)')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze cluster characteristics
        print("Cluster characteristics:")
        cluster_summary = customer_features.groupby('cluster').agg({
            'total_spent': 'mean',
            'avg_order_value': 'mean',
            'order_frequency': 'mean',
            'avg_rating': 'mean'
        }).round(2)
        
        print(cluster_summary)
        
        # Label clusters based on characteristics
        cluster_labels = {
            0: "Budget Conscious",
            1: "High Value",
            2: "Frequent Shoppers",
            3: "Premium Customers"
        }
        
        print("\nCluster interpretations:")
        for cluster_id, label in cluster_labels.items():
            count = (clusters == cluster_id).sum()
            print(f"Cluster {cluster_id} ({label}): {count} customers ({count/len(clusters)*100:.1f}%)")
    
    def anomaly_detection(self):
        """
        Detect anomalies/outliers in the dataset
        """
        print("\n=== ANOMALY DETECTION ===")
        
        # Prepare features for anomaly detection
        anomaly_features = self.df[['total_amount', 'unit_price', 'quantity', 
                                  'discount_percent', 'rating']].copy()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(anomaly_features)
        
        # Use Isolation Forest for anomaly detection
        isolation_forest = IsolationForest(contamination=0.05, random_state=42)
        anomaly_labels = isolation_forest.fit_predict(features_scaled)
        
        # Add anomaly labels to dataframe
        self.df['is_anomaly'] = anomaly_labels == -1
        
        print(f"Number of anomalies detected: {self.df['is_anomaly'].sum()}")
        print(f"Percentage of anomalies: {self.df['is_anomaly'].mean()*100:.2f}%")
        
        # Visualize anomalies
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Total amount vs unit price
        normal_data = self.df[~self.df['is_anomaly']]
        anomaly_data = self.df[self.df['is_anomaly']]
        
        axes[0].scatter(normal_data['unit_price'], normal_data['total_amount'], 
                       alpha=0.6, label='Normal', s=20)
        axes[0].scatter(anomaly_data['unit_price'], anomaly_data['total_amount'], 
                       color='red', label='Anomaly', s=20)
        axes[0].set_xlabel('Unit Price ($)')
        axes[0].set_ylabel('Total Amount ($)')
        axes[0].set_title('Anomalies: Unit Price vs Total Amount')
        axes[0].legend()
        
        # Quantity vs total amount
        axes[1].scatter(normal_data['quantity'], normal_data['total_amount'], 
                       alpha=0.6, label='Normal', s=20)
        axes[1].scatter(anomaly_data['quantity'], anomaly_data['total_amount'], 
                       color='red', label='Anomaly', s=20)
        axes[1].set_xlabel('Quantity')
        axes[1].set_ylabel('Total Amount ($)')
        axes[1].set_title('Anomalies: Quantity vs Total Amount')
        axes[1].legend()
        
        # Discount vs total amount
        axes[2].scatter(normal_data['discount_percent'], normal_data['total_amount'], 
                       alpha=0.6, label='Normal', s=20)
        axes[2].scatter(anomaly_data['discount_percent'], anomaly_data['total_amount'], 
                       color='red', label='Anomaly', s=20)
        axes[2].set_xlabel('Discount Percentage (%)')
        axes[2].set_ylabel('Total Amount ($)')
        axes[2].set_title('Anomalies: Discount vs Total Amount')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig('anomaly_detection.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze anomaly characteristics
        print("\nAnomalous transactions characteristics:")
        print(anomaly_data[['total_amount', 'unit_price', 'quantity', 
                          'discount_percent', 'product_category']].describe())

def main():
    """
    Main function to execute the complete analysis pipeline
    """
    print("🚀 Starting Synthetic Data Generation and Analysis Project")
    print("=" * 60)
    
    # Generate synthetic dataset
    generator = SyntheticDataGenerator(n_records=50000)
    df = generator.generate_dataset()
    
    # Save to Excel file
    excel_filename = 'synthetic_ecommerce_data.xlsx'
    print(f"\n📊 Saving dataset to {excel_filename}...")
    
    try:
        df.to_excel(excel_filename, index=False)
        print(f"✅ Dataset saved successfully!")
    except PermissionError:
        # File is likely open in Excel, try with a different name
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f'synthetic_ecommerce_data_{timestamp}.xlsx'
        print(f"⚠️ Original file is in use. Saving as {backup_filename}...")
        df.to_excel(backup_filename, index=False)
        print(f"✅ Dataset saved as {backup_filename}!")
        excel_filename = backup_filename
    
    # Initialize analyzer
    analyzer = DataAnalyzer(df)
    
    # Perform comprehensive analysis
    print("\n📈 Starting Data Analysis...")
    
    # Basic statistics
    analyzer.basic_statistics()
    
    # Correlation analysis
    analyzer.correlation_analysis()
    
    # Temporal analysis
    analyzer.temporal_analysis()
    
    # Category analysis
    analyzer.category_analysis()
    
    # Distribution analysis
    analyzer.distribution_analysis()
    
    # Clustering analysis
    analyzer.clustering_analysis()
    
    # Anomaly detection
    analyzer.anomaly_detection()
    
    print("\n🎉 Analysis complete! Check the generated visualizations and Excel file.")
    print("Files generated:")
    print("- synthetic_ecommerce_data.xlsx")
    print("- correlation_matrix.png")
    print("- temporal_analysis.png")
    print("- category_analysis.png")
    print("- distribution_analysis.png")
    print("- clustering_analysis.png")
    print("- anomaly_detection.png")

if __name__ == "__main__":
    main()
