from google.colab import drive
drive.mount('/content/drive')
path='drive/mydrive/dataset'
from sklearn.cluster import KMeans  # Note the capital 'K' and 'M'
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans  # Note the capital 'K' and 'M'
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np

# Read the uploaded dataset (replace 'your_dataset.csv' with your dataset's file name)
df = pd.read_csv('/content/drive/MyDrive/dataset/Book2.csv')
df.head()
df.tail()
def clean_tags(tag):
    tag = tag.strip()  # Remove leading and trailing spaces
    tag = tag.replace('[', '').replace(']', '')  # Remove square brackets
    tag = tag.replace(' ', '')  # Remove spaces within the tag
    return tag

# Apply the clean_tags function to the 'tags' column
df['tags'] = df['tags'].apply(clean_tags)

# Display the DataFrame
print(df)

word_to_label = {}
current_label = 1  # Start with label 1

# Iterate through the unique words in the "tags" column
for word in df['tags'].unique():
    word_to_label[word] = current_label
    current_label += 1

# Map the words to numerical labels and create a new "labels" column
df['tagslabels'] = df['tags'].map(word_to_label)
scaler=MinMaxScaler()
scaler.fit(df[['friendsCount']])
df['friendsCount']=scaler.transform(df[['friendsCount']])
scaler=MinMaxScaler()
scaler.fit(df[['tagslabels']])
df['']=scaler.transform(df[['tagslabels']])
arr=range(1,11)
sse=[]
for i in arr:
  km=KMeans(n_clusters=i)
  km.fit(df[['friendsCount','tagslabels']])
  sse.append(km.inertia_)
plt.plot(arr,sse)
km=KMeans(n_clusters=4)
km.fit(df[['friendsCount','tagslabels']])
predicted=km.fit_predict(df[['friendsCount','tagslabels']])
df['labels']=predicted
df1=df[df.labels==0]
df2=df[df.labels==1]
df3=df[df.labels==2]
df4=df[df.labels==3]
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],c='black',marker='*')
plt.scatter(df1['friendsCount'],df1['tagslabels'],c='pink')
plt.scatter(df2['friendsCount'],df2['tagslabels'],c='red')
plt.scatter(df3['friendsCount'],df3['tagslabels'],c='green')
plt.scatter(df4['friendsCount'],df4['tagslabels'],c='yellow')
# Create a dictionary to store the mapping from cluster to value
cluster_to_value = {}

# Generate values based on the number of occurrences of each cluster
for  cluster,count in df['labels'].value_counts().items():
    values = [0.1 + i * 0.1 for i in range(count)]
    cluster_to_value[cluster] = values

# Create a new column "cluster_count" to count occurrences within each cluster
df['cluster_count'] = df.groupby('labels').cumcount()

# Create a new column "cluster_value" by mapping cluster IDs and counts to values
df['cluster_value'] = df.apply(lambda row: cluster_to_value[row['labels']][row['cluster_count']], axis=1)

# Drop the temporary "cluster_count" column
df['clus']=df['labels']+df['cluster_value']

# Display the DataFrame with the new "cluster_value" column
cluster_mapping = {}
for cluster_id in df['labels'].unique():
    cluster_members = df[df['labels'] == cluster_id]['clus'].tolist()
    cluster_mapping[cluster_id] = cluster_members
# Simulate user input (replace with actual user input)
user_selected_cluster = 0
# Example: User selects cluster 0

# Get recommendations for the selected cluster
selected_cluster_members = cluster_mapping.get(user_selected_cluster, [])

# Exclude the user's own user_id from recommendations (if needed)
user_id_to_exclude = 0.2
number = user_id_to_exclude
rounded_number = round(number)
if user_selected_cluster==rounded_number:
  # Example: User's user_id
  if user_id_to_exclude in selected_cluster_members:
    selected_cluster_members.remove(user_id_to_exclude)

# Display recommendations
  if selected_cluster_members:
    print(f"Recommendations for Cluster {user_selected_cluster}:")
    for user_id in selected_cluster_members:
        # Retrieve the user's screenName using their user_id
        user_screen_name = df[df['clus'] == user_id]['screenName'].values[0]
        print({user_screen_name})
  else:
    print("No recommendations for Cluster {user_selected_cluster}.")
else:
  print("the cluster does not match with the user id ")
cluster_1_members = df[df["labels"] == 0]

# Print the screenName for each member in Cluster 1
print("Screen Names in Cluster 1:")
for index, row in cluster_1_members.iterrows():
    print(row["screenName"])
