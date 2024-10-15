import pandas as pd
import data_loader
import numpy as np
from ydata_profiling import ProfileReport

users = data_loader.load_users("users.csv")

entries_1 = data_loader.load_entries("entries1.csv")
entries_2 = data_loader.load_entries("entries2.csv")
entries_3 = data_loader.load_entries("entries3.csv")
merged_entries = pd.concat([entries_1, entries_2, entries_3])

likes = data_loader.load_likes("likes.csv")
comments = data_loader.load_comments("commentAugSept.csv")


# Step 1: Count the number of likes for each PostID
like_count = likes.groupby('PostID').size().reset_index(name='LikeCount')
# Step 2: Merge the likes count with the entries DataFrame
entries_with_likes = pd.merge(entries_1, like_count, on='PostID', how='left')
# Step 3: Replace NaN in NumLikes with 0
entries_with_likes['LikeCount'] = entries_with_likes['LikeCount'].fillna(0)
# Step 4: Count the number of comments for each PostID
comment_count = comments.groupby('EntryID').size().reset_index(name='CommentCount')
# Step 5: Merge the comments count with the entries_with_likes DataFrame
entries_with_comments_and_likes = pd.merge(entries_with_likes, comment_count, left_on='PostID', right_on='EntryID', how='left')
# Step 6: Replace NaN in CommentCount with 0
entries_with_comments_and_likes['CommentCount'] = entries_with_comments_and_likes['CommentCount'].fillna(0)
# Drop the 'EntryID' column, as it's now redundant
entries_with_comments_and_likes.drop(columns=['EntryID'], inplace=True)

print(users.shape)
print(entries_1.shape)
print(likes.shape)

print(entries_with_comments_and_likes.shape)


#display(entries_with_comments_and_likes)
entries_with_comments_and_likes.info()


# Sample a subset of  data (e.g., 10,000 rows)
#sampled_data = entries_with_comments_and_likes.sample(n=10000, random_state=1)

#Generate the profile report for the merged data
#profile = ProfileReport(sampled_data, title="Entries with Comments and Likes Profile Report")
#profile.to_file("eda.html")