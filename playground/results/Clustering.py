import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the provided CSV file
file_path = "./cifar100_results.csv"
df = pd.read_csv(file_path)

# Compute the difference between "original" and other transformation results
difference_df = df.copy()
for col in df.columns[3:]:  # Exclude "Metric" and "original"
    difference_df[col] = df["original"] - df[col]

# Drop the original column as we now have differences
difference_df = difference_df.drop(columns=["original"])

# Normalize the data
scaler = StandardScaler()
scaled_diff_data = scaler.fit_transform(difference_df.iloc[:, 2:])  # Exclude "Metric" column

# Determine the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans.fit(scaled_diff_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# Apply K-Means clustering on the difference data with the chosen k
optimal_k = 4  # 예시로 4를 사용, 엘보우 그래프를 보고 결정
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, max_iter=300, random_state=42)
difference_df["Cluster"] = kmeans.fit_predict(scaled_diff_data)

# Save the clustered data to a CSV file
output_file_path = "./cifar100_difference_clustered_results.csv"
difference_df.to_csv(output_file_path, index=False)

# Provide the download link
output_file_path