from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(df, x, y, cluster_column):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=cluster_column, palette='Set1')
    plt.title(f'{x} vs. {y} by Cluster')
    plt.show()

def apply_kmeans(df, n_clusters=3):
    # Standardize the data
    scaler = StandardScaler()
    # data = df[['TextLength', 'LikeCount', 'CommentCount', 'Hour']]
    data = df[['TextLength', 'LikeCount', 'Hour']]
    data_scaled = scaler.fit_transform(data)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=6, random_state=0)
    df['Cluster'] = kmeans.fit_predict(data_scaled)

    # Plot the clusters for text length vs. number of likes
    plot_clusters(df, 'TextLength', 'LikeCount', 'Cluster')

    # Plot the clusters for text length vs. number of comments
    # plot_clusters(df, 'TextLength', 'CommentCount', 'Cluster')

    # Plot the clusters for number of likes vs. number of comments
    # plot_clusters(df, 'LikeCount', 'CommentCount', 'Cluster')

    # Plot the clusters for hour vs. number of likes
    plot_clusters(df, 'Hour', 'LikeCount', 'Cluster')

    return df

# A bit more useful because we don't know the number of clusters and data is quite packed anyway
def apply_db_scan(df, eps=0.5, min_samples=5):
    # Step 1: Standardize the data
    scaler = StandardScaler()
    # data = df[['TextLength', 'LikeCount', 'CommentCount', 'Hour']]
    data = df[['TextLength', 'LikeCount', 'Hour']]
    data_scaled = scaler.fit_transform(data)

    # Step 2: Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Step 3: Add the cluster labels to your DataFrame
    df['Cluster'] = dbscan.fit_predict(data_scaled)

    # Step 4: Visualize the clusters
    plot_clusters(df, 'TextLength', 'LikeCount', 'Cluster')
    # plot_clusters(df, 'TextLength', 'CommentCount', 'Cluster')
    plot_clusters(df, 'Hour', 'LikeCount', 'Cluster')

    return df