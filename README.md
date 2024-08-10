# User Clustering and Recommendation System

## Overview

This project implements a user clustering and recommendation system using the K-means clustering algorithm. The dataset is preprocessed to generate tags and assign labels, which are then used to cluster users. Based on the clusters, the system can recommend other users who belong to the same cluster.

## Features

- **Data Preprocessing:** Clean and process the dataset, including removing unwanted characters from tags and mapping them to numerical labels.
- **K-means Clustering:** Apply the K-means algorithm to cluster users based on their `friendsCount` and `tagslabels`.
- **Cluster Visualization:** Visualize the clusters and their centroids using Matplotlib.
- **Cluster-based Recommendations:** Provide recommendations of users from the same cluster, excluding the selected user.
- **Custom Cluster Values:** Assign a unique value to each user within a cluster to differentiate them.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/user-clustering.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd user-clustering
    ```

3. **Install the required dependencies:**

    Make sure you have Python installed. Then, install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

4. **Upload the dataset:**

    Ensure that your dataset (e.g., `Book2.csv`) is uploaded to your Google Drive in the specified path.


## Project Structure

- **main_script.py:** The main script containing the logic for data preprocessing, clustering, and recommendation.
- **Book2.csv:** The dataset containing user data, including tags, friends count, and screen names.

## Key Components

- **Pandas:** For data manipulation and analysis.
- **Scikit-learn:** For implementing the K-means clustering algorithm.
- **Matplotlib:** For visualizing the clusters and centroids.
- **Google Colab:** For executing the script and accessing Google Drive.

## Data Preprocessing

- **Tags Cleaning:** The `clean_tags` function removes unwanted characters and spaces from tags.
- **Label Mapping:** Each unique tag is assigned a numerical label, which is then used for clustering.

## Clustering Process

1. **Scaling:** Normalize the `friendsCount` and `tagslabels` features using MinMaxScaler.
2. **Elbow Method:** Determine the optimal number of clusters using the elbow method.
3. **K-means Clustering:** Apply K-means to cluster the users.
4. **Cluster Visualization:** Plot the clusters and their centroids for visual inspection.

## Recommendations

- **Cluster-based:** Recommend users within the same cluster.
- **Exclusion Logic:** Exclude the selected user's ID from recommendations if it matches the cluster.

## Future Enhancements

- **Advanced Clustering Techniques:** Experiment with other clustering algorithms like DBSCAN or hierarchical clustering.
- **Improved Recommendation Logic:** Incorporate additional features or user behaviors to refine recommendations.
- **Web Interface:** Develop a web-based interface using frameworks like Streamlit to make the system more accessible.

## Contributing

Feel free to fork this repository, create a new branch, and submit a pull request with your contributions. All contributions are welcome!

## Acknowledgments

- **Google Colab:** For providing the environment to run the project.
- **Scikit-learn:** For the machine learning tools and algorithms used in the project.
