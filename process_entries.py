import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import data_loader
from clusterings import apply_kmeans, apply_db_scan


### -------------------- Functions

def load_merged_entries(nrows=None, sampled=False):
    entries_1 = data_loader.load_entries("entries1.csv", nrows=nrows)
    entries_2 = data_loader.load_entries("entries2.csv", nrows=nrows)
    entries_3 = data_loader.load_entries("entries3.csv", nrows=nrows)
    merged= pd.concat([entries_1, entries_2, entries_3])
    # merged = entries_1

    # Sample a subset of  data - half of the data
    if sampled:
        merged = merged.sample(frac=1, random_state=1)

    return merged

def add_likes_comments(entries_df):
    likes = data_loader.load_likes("likes.csv")
    comments = data_loader.load_comments("commentAugSept.csv")

    # Step 1: Count the number of likes for each PostID
    like_count = likes.groupby('PostID').size().reset_index(name='LikeCount')
    # Step 2: Merge the likes count with the entries DataFrame
    entries_with_likes = pd.merge(entries_df, like_count, on='PostID', how='left')
    # Step 3: Replace NaN in NumLikes with 0
    entries_with_likes['LikeCount'] = entries_with_likes['LikeCount'].fillna(0)
    # Step 4: Count the number of comments for each PostID
    comment_count = comments.groupby('EntryID').size().reset_index(name='CommentCount')
    # Step 5: Merge the comments count with the entries_with_likes DataFrame
    entries_with_comments_and_likes = pd.merge(entries_with_likes, comment_count, left_on='PostID', right_on='EntryID',
                                               how='left')
    # Step 6: Replace NaN in CommentCount with 0
    entries_with_comments_and_likes['CommentCount'] = entries_with_comments_and_likes['CommentCount'].fillna(0)
    # Drop the 'EntryID' column, as it's now redundant
    entries_with_comments_and_likes.drop(columns=['EntryID'], inplace=True)

    return entries_with_comments_and_likes

def add_hours_by_timestamp(entries_df):
    entries_df['Timestamp'] = pd.to_datetime(entries_df['Timestamp'])
    entries_df['Hour'] = entries_df['Timestamp'].dt.hour

    return entries_df

# Function to perform basic analysis of text lengths
def basic_analysis(entries_df):
    # Total number of entries
    total_entries = df.shape[0]
    print(f"Total number of entries: {total_entries}")

    # Average text length
    average_length = entries_df['TextLength'].mean()
    print(f"Average Text Length: {average_length:.2f} characters")

    # Number of entries with text length greater than 500
    num_long_entries = entries_df[entries_df['TextLength'] > 600].shape[0]
    print(f"Number of entries with text length greater than 600: {num_long_entries}")

    # Number of entries with text length greater than 1000
    num_very_long_entries = entries_df[entries_df['TextLength'] > 1000].shape[0]
    print(f"Number of entries with text length greater than 1000: {num_very_long_entries}")

# Function to remove outliers using Z-scores
def remove_outliers_zscores(entries_df, column):
    # Calculate Z-scores for the specified column
    entries_df['Zscore'] = stats.zscore(entries_df[column])

    # Filter out rows with Z-score greater than 3 or less than -3 (so keep about 99.7% of the data in a normal distribution)
    entries_df = entries_df[(entries_df['Zscore'] < 3) & (entries_df['Zscore'] > -3)]

    return entries_df

# Function to remove outliers using a threshold value
def remove_outliers_threshold(entries_df, threshold, column):
    # Filter out rows with values greater than the threshold
    entries_df = entries_df[entries_df[column] < threshold]

    return entries_df

# Function to plot a histogram of text lengths
def plot_text_length_histogram(entries_df):
    plt.figure(figsize=(10, 6))
    sns.histplot(entries_df['TextLength'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    plt.show()

# Function to plot a count plot of text length groups
def plot_text_length_group_count(entries_df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='LengthGroup', data=entries_df, palette='Set2')
    plt.title('Count of Entries by Text Length Group')
    plt.xlabel('Text Length Group')
    plt.ylabel('Number of Entries')
    plt.xticks(rotation=45)
    plt.show()

# Function to plot a boxplot of text lengths by group
def plot_text_length_boxplot(entries_df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='LengthGroup', y='TextLength', data=entries_df, palette='Set3')
    plt.title('Boxplot of Text Lengths by Group')
    plt.xlabel('Text Length Group')
    plt.ylabel('Text Length (characters)')
    plt.xticks(rotation=45)
    plt.show()

def show_plots_text_length(entries_df):
    # Define bins for grouping text lengths
    bins = [0, 50, 150, 300, 500, float('inf')]
    labels = ['Very Short (0-50)', 'Short (51-150)', 'Medium (151-300)', 'Long (301-500)', 'Very Long (500+)']
    entries_df['LengthGroup'] = pd.cut(entries_df['TextLength'], bins=bins, labels=labels)

    # Plot 1: Histogram of text lengths
    plot_text_length_histogram(entries_df)

    # Plot 2: Count of entries in each length group
    plot_text_length_group_count(entries_df)

    # Plot 3: Boxplot of text lengths
    plot_text_length_boxplot(entries_df)

def show_plots_likes(entries_df):
    # Define bins for grouping number of likes
    bins = [0, 50, 100, 500, float('inf')]
    labels = ['0-50', '51-100', '101-500', '500+']
    entries_df['LikeGroup'] = pd.cut(entries_df['LikeCount'], bins=bins, labels=labels)

    # Plot 1: Histogram of number of likes
    plt.figure(figsize=(10, 6))
    sns.histplot(entries_df['LikeCount'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Likes')
    plt.xlabel('Number of Likes')
    plt.ylabel('Frequency')
    plt.show()

    # Plot 2: Count of entries in each like group
    plt.figure(figsize=(10, 6))
    sns.countplot(x='LikeGroup', data=entries_df, palette='Set2')
    plt.title('Count of Entries by Number of Likes Group')
    plt.xlabel('Number of Likes Group')
    plt.ylabel('Number of Entries')
    plt.xticks(rotation=45)
    plt.show()

    # Plot 3: Boxplot of number of likes by group
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='LikeGroup', y='LikeCount', data=entries_df, palette='Set3')
    plt.title('Boxplot of Number of Likes by Group')
    plt.xlabel('Number of Likes Group')
    plt.ylabel('Number of Likes')
    plt.xticks(rotation=45)
    plt.show()

### -------------------- Flows

def initial_analysis(entries_df):
    # Calculate the length of each 'Text'
    entries_df['TextLength'] = entries_df['Text'].apply(lambda x: len(str(x)))

    print('Basic Analysis of Text Lengths before Outlier Removal:')
    basic_analysis(entries_df)
    print('-----------------------------------')

    # Step 2: Remove outliers based on z-scores
    entries_df = remove_outliers_zscores(entries_df, 'TextLength')

    print('Basic Analysis of Text Lengths after Outlier Removal:')
    basic_analysis(entries_df)
    print('-----------------------------------')

    # Step 3: Plotting
    show_plots_text_length(entries_df)

### -------------------- Main code

merged_entries = load_merged_entries(sampled=True)
# Step 1: Create a DataFrame from merged_entries to analyze the 'Text' property
df = pd.DataFrame(merged_entries)
df['TextLength'] = df['Text'].apply(lambda x: len(str(x)))

#initial_analysis(df)

# Step 2: Remove outliers based on z-scores on the 'TextLength' column
df = remove_outliers_zscores(df, 'TextLength')

df_entries_comments_likes = add_likes_comments(df)

df_entries_comments_likes_hours = add_hours_by_timestamp(df_entries_comments_likes)

# Remove entries with less than 50 likes
df_entries_comments_likes_hours = df_entries_comments_likes_hours[df_entries_comments_likes_hours['LikeCount'] > 50]

# Build plots for number of likes
show_plots_likes(df_entries_comments_likes_hours)



# Show correlation matrix of text length, likes, and comments
# correlation_matrix = df_entries_comments_likes[['TextLength', 'LikeCount', 'CommentCount']].corr()
# print(correlation_matrix)

# Perform clustering on the data
# apply_kmeans(df_entries_comments_likes, n_clusters=6)
apply_db_scan(df_entries_comments_likes_hours, eps=0.5, min_samples=5)

# I would apply clustering on hour of the day and number of likes / comments / both and maybe text length. Can use a small epsilon for DBSCAN to group similar entries together.
# Is there correlation between no likes and no comments? - it is some kind of correlation, but not very strong
