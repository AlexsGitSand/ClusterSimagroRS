import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
from tslearn.metrics import dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from scipy.spatial.distance import squareform
from sklearn_extra.cluster import KMedoids
from fastdtw import fastdtw
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

color_temp='#030303' #black
color_relHum='#66CD00' #green
color_rainfall='#1874CD' #blue
color_solarRad='#CD6600' #orange

color_C0='#006400' # dark green
color_C1='#A52A2A' # dark red


def transform_dfTS_to_3Darray_DOYsorted(df, columns):
    
    # Get the unique values of StationID
    unique_station_ids = df['StationID'].unique()
    
    # Create an empty list to hold the dataframes
    split_dfs = []

    # Loop over the unique values of StationID and split the dataset accordingly
    for station_id in unique_station_ids:
        df_temp = df[df['StationID'] == station_id]
        firstMonthsYear = df_temp[df_temp["doy"] <= 90] # until march
        lastMonthsYear = df_temp[df_temp["doy"] >= 274] # from october
        df_season = pd.concat([lastMonthsYear, firstMonthsYear])
        split_dfs.append(df_season[columns])
    
    
    # Get the length of the first two dimensions of the array
    dim1 = len(split_dfs)
    dim2 = split_dfs[0].shape[0]
    dim3 = split_dfs[0].shape[1]


    df_return = np.zeros((dim1, dim2, dim3))

    # Copy the values from the dataframes to the new array
    for i, data in enumerate(split_dfs):
        df_return[i, :, :] = data.values
    
    return df_return




def plot_first_TS(array, station_list):
    
    stations = pd.read_csv('SIMAGROStationsList.csv')
    names = stations[['StationName','StationID']]
    
    index = 0
    
    plt.figure(figsize=(20,3))
    stationId = station_list[index]
    result = names[names['StationID'] == stationId]
    stationName = result.iloc[0]['StationName']
    plt.title("Station "+str(stationId)+" - "+stationName+" - (Temperature, Rel. Humidity, Rainfall, Solar Radiation")
    
    var_temp = []
    var_relHum = []
    var_rain = []
    var_solRad = []
    
    item = array[index]

    for i in item: # 182 itens
        var_temp.append(i[0])
        var_relHum.append(i[1])
        var_rain.append(i[2])    
        var_solRad.append(i[3])
    

    plt.plot(var_temp, '-',color=color_temp)
    plt.plot(var_relHum, '-',color=color_relHum)
    plt.plot(var_rain, '-',color=color_rainfall)
    plt.plot(var_solRad, '-',color=color_solarRad)
    
    

        
def plot_all_TS(array, station_list):
    
    stations = pd.read_csv('SIMAGROStationsList.csv')
    names = stations[['StationName','StationID']]
    
    for index, item in enumerate(array):
        plt.figure(figsize=(20,3))
        stationId = station_list[index]
        result = names[names['StationID'] == stationId]
        stationName = result.iloc[0]['StationName']
        plt.title("Station "+str(stationId)+" - "+stationName)
    
    
        for i in item:
            plt.plot(item, '-',c=(np.random.random(), np.random.random(), np.random.random()))
    

def plot_TS_by_index(array, station_list, index):
    
    stations = pd.read_csv('SIMAGROStationsList.csv')
    names = stations[['StationName','StationID']]
    
    
    plt.figure(figsize=(10,3))
    stationId = station_list[index]
    result = names[names['StationID'] == stationId]
    stationName = result.iloc[0]['StationName']
    plt.title("Station "+str(stationId)+" - "+stationName)#+" - Temperature (black), Relative Humidity (green), Rainfall (blue), Solar Radiation (orange)")
    
    
    var_temp = []
    var_relHum = []
    var_rain = []
    var_solRad = []
    
    item = array[index]

    for i in item: # 182 itens
        var_temp.append(i[0])
        var_relHum.append(i[1])
        var_rain.append(i[2])    
        var_solRad.append(i[3])
    
    plt.xlabel("Days")
    plt.margins(x=0.01)
    plt.plot(var_temp, '-',color=color_temp)
    plt.plot(var_relHum, '-',color=color_relHum)
    plt.plot(var_rain, '-',color=color_rainfall)
    plt.plot(var_solRad, '-',color=color_solarRad)


        
def plot_TS_by_index_separated_variables(array, station_list, index):
    
    stations = pd.read_csv('SIMAGROStationsList.csv')
    names = stations[['StationName','StationID']]
    
    stationId = station_list[index]
    result = names[names['StationID'] == stationId]
    stationName = result.iloc[0]['StationName']
    
    var_temp = []
    var_relHum = []
    var_rain = []
    var_solRad = []
    
    item = array[index]

    for i in item: # 182 itens
        var_temp.append(i[0])
        var_relHum.append(i[1])
        var_rain.append(i[2])    
        var_solRad.append(i[3])
    
    station_name_temp = stationName.rsplit(' (')[0]
    plt.figure(figsize=(10,3))
    plt.title("Station "+str(stationId)+" - "+station_name_temp+" - Temperature")
    plt.xlabel("Days")
    plt.ylabel("°C")
    plt.margins(x=0.01)
    plt.plot(var_temp, '-',color=color_temp)
    
    plt.figure(figsize=(10,3))
    plt.title("Station "+str(stationId)+" - "+station_name_temp+" - Relative Humidity")
    plt.xlabel("Days")
    plt.ylabel("%")
    plt.margins(x=0.01)
    plt.plot(var_relHum, '-',color=color_relHum)
    
    plt.figure(figsize=(10,3))
    plt.title("Station "+str(stationId)+" - "+station_name_temp+" - Rainfall")
    plt.xlabel("Days")
    plt.ylabel("mm")
    plt.margins(x=0.01)
    plt.plot(var_rain, '-',color=color_rainfall)
    
    plt.figure(figsize=(10,3))
    plt.title("Station "+str(stationId)+" - "+station_name_temp+" - Solar Radiation")
    plt.xlabel("Days")
    plt.ylabel("W/m²")
    plt.margins(x=0.01)
    plt.plot(var_solRad, '-',color=color_solarRad)

        

def sil_and_elbow_scores_TSKMeans(df_array, max_cluster, metric_c, n_init_c, max_iter_c):
    
    sil_scores = []
    elbow_plot_distances = []

    K = range(1,max_cluster)

    for k in K:
        # model instantiation
        model = TimeSeriesKMeans(n_clusters=k, metric=metric_c, n_init=n_init_c, max_iter=max_iter_c)
        y_pred = model.fit_predict(df_array)
    
        # silhouette score
        if (k>1):
            score=silhouette_score(df_array, y_pred, metric=metric_c)
            sil_scores.append(score)
    
        # inertia, for elbow plot
        elbow_plot_distances.append(model.inertia_)

    c=2
    for i in sil_scores:
        print('Clusters = '+str(c)+'  Silhouette Score: %.3f' % i)
        c+=1
    
    
    plt.plot(K, elbow_plot_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k - SoftDTW')
    plt.show()
    
    
def cluster_labels_TSKmeans(df_array, n_clusters_c, metric_c, n_init_c, max_iter_c):
    model = TimeSeriesKMeans(n_clusters=n_clusters_c, metric=metric_c, n_init=n_init_c, max_iter=max_iter_c)
    y_pred = model.fit_predict(df_array)
    return y_pred


def sil_score_TSHierarcClustering(df_array):
    
    # Calculate the DTW distance matrix
    positions = df_array.shape[0]
    dtw_distance_matrix = np.zeros((positions, positions))
    for i in range(positions):
        for j in range(positions):
            dtw_distance_matrix[i, j] = dtw(df_array[i], df_array[j])
            
    
    # Convert the distance matrix to condensed form
    dtw_distance_condensed = squareform(dtw_distance_matrix)

    # Perform hierarchical clustering on the condensed DTW distance matrix
    linkage_matrix = linkage(dtw_distance_condensed, method='ward')

    sil_scores = []
 
    for i in range(2,dtw_distance_matrix.shape[0]):
    
        n_clusters = i # Change this to the desired number of clusters
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        score=silhouette_score(dtw_distance_matrix, cluster_labels, metric="precomputed")
        sil_scores.append(score)

    c=2
    for i in sil_scores:
        print('Clusters = '+str(c)+'  Silhouette Score: %.3f' % i)
        c+=1

        
def dendogram_TSHierarcClustering(df_array, num_clusters, list_names_dendogram):
    
    # Calculate the DTW distance matrix
    positions = df_array.shape[0]
    dtw_distance_matrix = np.zeros((positions, positions))
    for i in range(positions):
        
        for j in range(positions):
            dtw_distance_matrix[i, j] = dtw(df_array[i], df_array[j])

            
    # Convert the distance matrix to condensed form
    dtw_distance_condensed = squareform(dtw_distance_matrix)

    # Perform hierarchical clustering on the condensed DTW distance matrix
    linkage_matrix = linkage(dtw_distance_condensed, method='ward')

    # Perform clustering by cutting the dendrogram at a suitable height
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

    #print(cluster_labels)
    
    plt.figure(figsize=(13, 10))
    dendrogram(
            linkage_matrix,
            orientation='right',
            labels=list_names_dendogram,
            distance_sort='descending',
            show_leaf_counts=False
          )
    plt.show()

    return cluster_labels, linkage_matrix


def plot_map_clustering_names(stations, cluster_labels):

    stations_coord = stations[['StationName','StationID','latitude', 'longitude']]
        
    df_plot = stations_coord.sort_values('StationID')

    BBox = ((-57.766, -49.351, #longitude
          -33.962, -26.800)) #latitude
    
    df_plot['cluster'] = cluster_labels


    n_clusters=df_plot['cluster'].unique()

    dfs_clusters=[]

    for i in range(0,len(n_clusters)):
        df_temp = df_plot.loc[df_plot['cluster'] == n_clusters[i]]
        dfs_clusters.append(df_temp)

    plot_map = plt.imread('rs_map_white.jpeg')
    fig, ax = plt.subplots(figsize = (8,7))

    for i in range(0,len(n_clusters)):
        if (i==0):
            color = color_C0
        else:
            color = color_C1
        ax.scatter(dfs_clusters[i].longitude, dfs_clusters[i].latitude, zorder=1, alpha=1, c=color, s=40)

    
    ax.set_title('Weather Stations clusters on RS Map')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])

    for index, row in df_plot.iterrows():
        st_name = row['StationName'].rsplit(' (')[0]
        if (st_name == "Itaqui") | (st_name=="Cachoeira do Sul") | (st_name=="Piratini"):
            plt.text(row['longitude'],row['latitude']+0.1, st_name, horizontalalignment='center', size='medium', color='black')            
        else: 
            plt.text(row['longitude']+0.1,row['latitude']-0.25, st_name, horizontalalignment='center', size='medium', color='black')

        
    ax.imshow(plot_map, zorder=0, extent = BBox, aspect= 'equal')
    
    
    
    
def plot_map_clustering(stations, cluster_labels):

    stations_coord = stations[['StationName','StationID','latitude', 'longitude']]
        
    df_plot = stations_coord.sort_values('StationID')

    BBox = ((-57.766, -49.351, #longitude
          -33.962, -26.800)) #latitude
    
    df_plot['cluster'] = cluster_labels


    n_clusters=df_plot['cluster'].unique()

    dfs_clusters=[]

    for i in range(0,len(n_clusters)):
        df_temp = df_plot.loc[df_plot['cluster'] == n_clusters[i]]
        dfs_clusters.append(df_temp)

    plot_map = plt.imread('rs_map_white.jpeg')
    fig, ax = plt.subplots(figsize = (8,7))

    for i in range(0,len(n_clusters)):
        if (i==0):
            color = color_C0
        else:
            color = color_C1
        ax.scatter(dfs_clusters[i].longitude, dfs_clusters[i].latitude, zorder=1, alpha=1, c=color, s=40)

    
    ax.set_title('Weather Stations clusters on RS Map')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
        
    ax.imshow(plot_map, zorder=0, extent = BBox, aspect= 'equal')
    
    
    

# Custom DTW distance function
def dtw_distance(x, y):
    distance, _ = fastdtw(x, y, dist=lambda x, y: np.linalg.norm(x - y))
    return distance
    
    
def cluster_labels_TSKMedoids(df_array, n_clusters_c, max_iter_c):
    
    # Reshape the data to (n_samples, n_timestamps, n_features)
    n_samples, n_timestamps, n_features = df_array.shape
    X = df_array.reshape((n_samples, n_timestamps, n_features))

    # Perform k-medoids clustering with DTW as the distance metric
    kmedoids = KMedoids(n_clusters=n_clusters_c, metric="euclidean", random_state=0, max_iter=max_iter_c)
    y_pred = kmedoids.fit_predict(X.reshape(n_samples, -1))

    # Now, y_pred contains the cluster assignments for each time series

    # Print cluster assignments
    #for cluster_idx in range(n_clusters):
    #    cluster_samples = np.where(y_pred == cluster_idx)[0]
    #print(f"Cluster {cluster_idx}: {len(cluster_samples)} time series")

    # To access cluster medoids, reshape them back to the original format
    cluster_medoids_indices = kmedoids.medoid_indices_
    cluster_medoids = X[cluster_medoids_indices]
    
    return y_pred, cluster_medoids


def sil_and_elbow_scores_TSKMedoids(df_array, max_cluster, max_iter_c):
    
    
    n_samples, n_timestamps, n_features = df_array.shape
    X = df_array.reshape((n_samples, n_timestamps * n_features))

    # Range of cluster numbers to try
    K = range(1,max_cluster)

    # wcss for elbow plot
    elbow_plot_distances = []
    
    # Calculate silhouette scores for each k
    silhouette_scores = []
    
    for k in K:
        kmedoids = KMedoids(n_clusters=k, metric="euclidean", random_state=0, max_iter=max_iter_c)
        y_pred = kmedoids.fit_predict(X)

        if (k>1):
            silhouette_scores.append(silhouette_score(X, y_pred, metric="euclidean"))

        elbow_plot_distances.append(kmedoids.inertia_)
        
    c=2
    for i in silhouette_scores:
        print('Clusters = '+str(c)+'  Silhouette Score: %.3f' % i)
        c+=1
        

    # Plot the elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(K, elbow_plot_distances, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal Cluster Number')
    plt.show()
    
    
    

# flatten 3D array to 2D pandas dataframe
def flatten_multivariate_time_series(data):
    # Get the shape of the input array
    num_stations, num_timestamps, num_features = data.shape

    # Create a list of column names based on your specified format
    column_names = [
        f"station{station}_feature{feature}"
        for station in range(num_stations)
        for feature in range(num_features)
    ]

    # Reshape the data into a 2D array
    reshaped_data = data.transpose(1, 0, 2).reshape(num_timestamps, -1)

    # Create a DataFrame with column names
    flattened_df = pd.DataFrame(reshaped_data, columns=column_names)

    return flattened_df


def predict_and_evaluate_all_stations_rmse(data_df):
    num_stations = len(data_df.columns) // 4
    total_rmse = 0.0
    
    for station_index in range(num_stations):
        # Calculate the starting and ending column indexes for the current station
        start_col = station_index * 4
        end_col = (station_index + 1) * 4
        
        # Extract the data for the current station
        station_data = data_df.iloc[:, start_col:end_col]
        
        # Remove the data for the current station from the dataset
        reduced_df = data_df.drop(columns=station_data.columns)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            reduced_df, station_data, test_size=0.2, random_state=42
        )

        # Train a prediction model (e.g., Linear Regression)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test data
        predictions = model.predict(X_test)

        # Calculate the Root Mean Squared Error (RMSE) for the station
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        print(f"RMSE for station {station_index}: {rmse}")

        # Add the RMSE to the total
        total_rmse += rmse

    # Calculate the average RMSE across all stations
    average_rmse = total_rmse / num_stations
    #print(f"Average RMSE across all stations: {average_rmse}")
    
    return average_rmse


def calculate_cluster_rmse(data_df, cluster_labels):
    unique_clusters = np.unique(cluster_labels)
    total_cluster_rmse = 0.0
    
    for cluster_label in unique_clusters:
        # Find stations in the current cluster
        cluster_indices = np.where(cluster_labels == cluster_label)[0]
        print("Cluster "+str(cluster_label))
        # Extract the data for stations in the current cluster
        cluster_data = data_df.iloc[:, [i * 4 + j for i in cluster_indices for j in range(4)]]
        #print(cluster_data)
        # Calculate RMSE within the cluster
        cluster_rmse = predict_and_evaluate_all_stations_rmse(cluster_data)
        
        print(f"RMSE for Cluster {cluster_label}: {cluster_rmse}")
        
        # Add cluster RMSE to the total
        total_cluster_rmse += cluster_rmse

    # Calculate the average RMSE across all clusters
    average_cluster_rmse = total_cluster_rmse / len(unique_clusters)
    print(f"Average Cluster RMSE: {average_cluster_rmse}")