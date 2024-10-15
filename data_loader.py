import pandas as pd

# Function to load user data
def load_users(filepath):
    user_headers = ['ID', 'Type', 'Name', 'Reserved field', 'Description']
    # Read the users data, some of the lines contain the delimeter | character so decided to skip them
    users = pd.read_csv(filepath, sep="|", names=user_headers, header=None, on_bad_lines='skip')
    # Drop the not needed columns
    users.drop(columns=['Reserved field'], inplace=True)
    return users

# Function to load entries data. Optionally, you can specify the number of rows to load to avoid loading the entire dataset for easier debugging
def load_entries(filepath, nrows=None):
    entry_headers = [
        'PostID', 'PostedBy', 'SourceName', 'SourceURL', 'GeoX', 'GeoY',
        'Timestamp', 'Text', 'NumImg', 'ImgURL', 'NumVid', 'VidURL'
    ]
    usecols = ['PostID', 'PostedBy', 'SourceName', 'Timestamp', 'Text', 'NumImg', 'NumVid']
    entries = pd.read_csv(filepath, sep="\t", names=entry_headers, header=None, nrows=nrows, usecols=usecols)
    return entries

# Function to load likes data
def load_likes(filepath):
    like_headers = ['userID', 'PostID', 'Timestamp']
    likes = pd.read_csv(filepath, sep="\t", names=like_headers, header=None)
    return likes

# Function to load comments data
def load_comments(filepath):
    comment_headers = [
        'PostID', 'EntryID', 'PostedBy', 'SourceName', 'SourceURL', 'GeoX', 'GeoY',
        'Timestamp', 'Text', 'NumImg', 'ImgURL', 'NumVid', 'VidURL'
    ]
    comments = pd.read_csv(filepath, sep="\t", names=comment_headers, header=None)
    # Drop the not needed columns
    comments.drop(columns=['SourceURL', 'GeoX', 'GeoY', 'ImgURL', 'VidURL'], inplace=True)
    return comments