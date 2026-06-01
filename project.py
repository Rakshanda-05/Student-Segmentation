## **Phase 1: Import Libraries**

# Data Handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Evaluation
from sklearn.metrics import silhouette_score

# Dimensionality Reduction
from sklearn.decomposition import PCA

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

"""## **Phase 2: Load Dataset**"""

# Load dataset
file_path = r'C:\Users\hp\OneDrive\Desktop\College\3rd Year\Sem 6\ML Project\Project\student_performance_prediction.csv'
df = pd.read_csv(file_path)

# Display first 5 rows
print(df.head())

"""## **Phase 3: Basic Dataset Information**"""

# Shape of dataset
print("Dataset Shape:", df.shape)

# Column names
print("\nColumns:\n", df.columns)

# Dataset information
print("\nDataset Info:")
print(df.info())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

"""## **Phase 4: Missing Value Analysis**"""

# Check missing values
missing_values = df.isnull().sum()
print(missing_values)

# Visualize missing values
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.show()

"""## **Phase 5: Duplicate Value Analysis**

"""

# Check duplicates
print("Duplicate Rows:", df.duplicated().sum())

# Remove duplicates if present
df.drop_duplicates(inplace=True)

"""## **Phase 6: Data Cleaning**"""

# Fill numeric missing values with median
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

"""## **Phase 7: Outlier Detection and Removal**"""

plt.figure(figsize=(15,8))
df.boxplot(rot=90)
plt.title('Boxplot for Outlier Detection')
plt.show()
# IQR Method
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("Dataset Shape After Outlier Removal:", df.shape)

"""## **Phase 8: Exploratory Data Analysis (EDA)**"""

#8.1 Distribution Analysis
# Histograms
plt.figure(figsize=(15,10))
df.hist(bins=30, figsize=(15,10))
plt.suptitle('Feature Distributions')
plt.show()

#8.2 Correlation Analysis
plt.figure(figsize=(12,8))
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#8.3 Attendance vs Performance
plt.figure(figsize=(10,6))
sns.scatterplot(x=df['Attendance Rate'], y=df['Previous Grades'])
plt.title('Attendance Rate vs Previous Grades')
plt.xlabel('Attendance Rate')
plt.ylabel('Previous Grades')
plt.show()

#8.4 Pairplot Analysis
sns.pairplot(df[numeric_cols[:5]])
plt.show()

#8.5 Category Countplots
# Ensure categorical_cols is defined (it was defined in a previous cell but might be lost if the kernel was reset)
categorical_cols = df.select_dtypes(include='object').columns

# Example categorical column - choose a more meaningful column than 'Student ID' for a countplot
# 'Parent Education Level' is a good candidate for a countplot
if 'Parent Education Level' in categorical_cols:
    categorical_example = 'Parent Education Level'
elif 'Participation in Extracurricular Activities' in categorical_cols:
    categorical_example = 'Participation in Extracurricular Activities'
elif 'Passed' in categorical_cols:
    categorical_example = 'Passed'
elif len(categorical_cols) > 0: # Fallback if specific ones not found, use the first available
    categorical_example = categorical_cols[0]
else:
    print("No categorical columns found for plotting.")
    # No categorical columns to plot, exit gracefully
    exit()

plt.figure(figsize=(10,6))
sns.countplot(x=df[categorical_example])
plt.title(f'{categorical_example} Distribution')
plt.xticks(rotation=45)
plt.show()

"""## **Phase 9: Feature Engineering**"""

# Example feature engineering

# Consistency Score
if 'Assignment_Score' in df.columns and 'Exam_Score' in df.columns:
    df['Consistency_Score'] = 100 - abs(df['Assignment_Score'] - df['Exam_Score'])

# Improvement Velocity
if 'Previous_Score' in df.columns and 'Exam_Score' in df.columns:
    df['Improvement_Velocity'] = df['Exam_Score'] - df['Previous_Score']

# Engagement Score
if 'Attendance' in df.columns and 'Participation_Score' in df.columns:
    df['Engagement_Score'] = (df['Attendance'] * 0.6) + (df['Participation_Score'] * 0.4)

print(df.head())

"""## **Phase 10: Encoding Categorical Features**"""

label_encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

"""## **Phase 11: Feature Scaling**"""

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

print(scaled_df.head())

"""## **Phase 12: Elbow Method for Optimal Clusters**"""

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,6))
plt.plot(range(1,11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

"""## **Phase 13: K-Means Clustering**"""

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)

clusters = kmeans.fit_predict(scaled_df)

# Add cluster column
df['Cluster'] = clusters

print(df['Cluster'].value_counts())

"""## **Phase 14: Silhouette Score**"""

score = silhouette_score(scaled_df, clusters)
print('Silhouette Score:', score)
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

# Calculate silhouette values
sample_silhouette_values = silhouette_samples(scaled_df, clusters)

fig, ax = plt.subplots(figsize=(10, 6))

y_lower = 10

for i in range(3):  # Number of clusters
    ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / 3)

    ax.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )

    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    y_lower = y_upper + 10

# Average silhouette score line
silhouette_avg = silhouette_score(scaled_df, clusters)
ax.axvline(x=silhouette_avg, color="red", linestyle="--")

ax.set_title("Silhouette Analysis for Student Clustering")
ax.set_xlabel("Silhouette Coefficient Values")
ax.set_ylabel("Cluster Label")

ax.set_yticks([])
ax.set_xlim([-0.1, 1])

plt.show()

"""## **Phase 15: PCA Visualization**"""

# Reduce dimensions
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_df)

pca_df = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = clusters

# Visualize clusters
plt.figure(figsize=(10,6))
sns.scatterplot(
    x='PCA1',
    y='PCA2',
    hue='Cluster',
    data=pca_df,
    palette='Set1'
)

plt.title('Student Clusters Visualization')
plt.show()

"""## **Phase 16: Cluster Analysis**"""

# Mean values of clusters
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)

print(cluster_summary)

"""## **Phase 17: Naming Student Segments**"""

# Rename clusters
cluster_names = {
    0: 'Average Students',
    1: 'High Performers',
    2: 'Students At Risk'
}

# Create segment column
df['Student_Segment'] = df['Cluster'].map(cluster_names)

print(df[['Cluster', 'Student_Segment']].head())

