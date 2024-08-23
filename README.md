# K-Means with PySpark

This project implements the K-Means clustering algorithm using PySpark, a powerful tool for distributed data processing. The project leverages PySpark's MapReduce framework to efficiently handle large datasets, demonstrating the scalability and performance benefits of distributed computing. The K-Means algorithm is applied to various datasets, and its performance is evaluated using metrics such as the Calinski-Harabasz Index and Adjusted Rand Index. This project showcases the practical application of PySpark and MapReduce in solving real-world data clustering problems.

## Overview

The main focus of this project is to utilize PySpark for performing K-Means clustering on different datasets. The implementation leverages PySpark's MapReduce capabilities to distribute the computation across a cluster, making it scalable and efficient. The clustering results are evaluated to determine the quality of the clusters formed, and the results are saved for further analysis.

### Key Features

- **K-Means Clustering**: Implements the K-Means algorithm with support for varying the number of clusters (`K`) and number of experiments (`exp`).
- **PySpark Integration**: Utilizes PySpark for distributed data processing, enabling the algorithm to scale with large datasets.
- **MapReduce**: The K-Means algorithm is implemented using PySpark’s MapReduce framework. The `mapper` function assigns data points to the nearest cluster, and the `reducer` function recalculates the centroids by averaging the assigned points.
- **Evaluation Metrics**: Computes the Calinski-Harabasz Index and Adjusted Rand Index to evaluate the quality of the clustering.
- **Result Export**: Saves the clustering results and evaluation metrics to a CSV file for further analysis.

## Project Structure

- **`main.py`:** The main script that performs K-Means clustering on the datasets using PySpark, evaluates the results, and exports the metrics to `kmeans_results.csv`.
  - **MapReduce Implementation**: 
    - **Mapper Function**: Assigns each data point to the nearest cluster based on the current centroids.
    - **Reducer Function**: Recalculates the centroids by averaging the points assigned to each cluster.
- **Datasets:**
  - **`iris.csv`:** The Iris dataset.
  - **`glass.csv`:** The Glass dataset.
  - **`parkinsons.csv`:** The Parkinsons dataset.
- **`kmeans_results.csv`:** The CSV file where the clustering results and evaluation metrics are saved.

## How to Use

### Prerequisites

- **Python 3.x**
- **PySpark**: Ensure PySpark is installed and configured correctly on your system.

### Running the K-Means Clustering

To run the K-Means clustering algorithm on the datasets, execute the following command:

```bash
python main.py
```
This will process the datasets, apply K-Means clustering with varying K values, and save the results to kmeans_results.csv.

## Evaluation  
The results of the clustering are saved in kmeans_results.csv. The file contains the following columns:
- **Dataset name**: The name of the dataset used.
- **K**: The number of clusters used in the K-Means algorithm.
- **Average and std CH**: The average and standard deviation of the Calinski-Harabasz Index across experiments.
- **Average and std ARI**: The average and standard deviation of the Adjusted Rand Index across experiments.


## Installation  

1. **Clone the repository**
   ```bash
   git clone https://github.com/omriarie/spark-mapreduce.git

   ```

2. **Navigate to the project directory**
   ```bash
   cd spark-mapreduce
   ```

3. **Install required dependencies**
   ```bash
   pip install pyspark numpy pandas scikit-learn
   ```

4. **Run the K-Means clustering**
   ```bash
   python main.py
   ```
## Datasets
The datasets used in this project (iris.csv, glass.csv, and parkinsons.csv) are standard datasets commonly used for machine learning tasks. In this project, they are included primarily for testing and demonstrating the functionality of the K-Means algorithm implemented in PySpark. The datasets themselves are not the focus of the project but serve to validate the algorithm's implementation and its scalability.


## Contact
For any inquiries or issues, please open an issue on the GitHub repository or contact the maintainers directly:

Omri Arie – omriarie@gmail.com  
Project Link: https://github.com/omriarie/spark-mapreduce
