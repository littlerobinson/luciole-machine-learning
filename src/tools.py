import numpy as np
import pandas as pd

import xgboost as xgb

import plotly.express as px

from pympler import asizeof

from sklearn.datasets import load_diabetes


def haversine(lon_1: float, lon_2: float, lat_1: float, lat_2: float):
    """
    The haversine formula determines the great-circle distance between two points on a sphere given their longitudes and latitudes.
    
    Arguments:
    lon_1 -- start longitude
    lon_2 -- target longitude
    lat_1 -- start longitude
    lat_2 -- target latitude
    """

    # earth radius: 6371km
    earth_radius = 6371
    
    lon_1, lon_2, lat_1, lat_2 = map(np.radians, [lon_1, lon_2, lat_1, lat_2])  # Convert degrees to Radians
    
    
    diff_lon = lon_2 - lon_1
    diff_lat = lat_2 - lat_1
    

    distance_km = 2*earth_radius*np.arcsin(np.sqrt(np.sin(diff_lat/2.0)**2 + np.cos(lat_1) * np.cos(lat_2) * np.sin(diff_lon/2.0)**2)) # earth radius: 6371km
    
    return distance_km


def ouliers_viewver(dataset, columns=[]):
    """
    Display outliers from Pandas dataset.

    Parameters:
    dataset (pd.DataFrame): Pandas dataset
    columns (list): list of the columns in dataset to check outliers. All by default. 
    
    Returns:
    Void
    """
    outliers_count = {}
    if len(columns) < 1:
        columns = dataset.columns
        
    for col in columns:
        mean = dataset[col].mean()
        std = dataset[col].std()
        
        # 3 sigmas rules
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

        #print(f"For col {col}, lower is {lower_bound} and upper is {upper_bound}")
        
        # Create mask
        outliers = (dataset[col] < lower_bound) | (dataset[col] > upper_bound)
        outliers_count[col] = outliers.sum()

    outliers_df = pd.DataFrame(list(outliers_count.items()), columns=['Column', 'Outliers'])
    fig = px.bar(outliers_df, x='Column', y='Outliers', title='Outliers count by column')
    fig.show()


def delete_ouliers(dataset, columns=[]):
    """
    Delete outliers from Pandas dataset.

    Parameters:
    dataset (pd.DataFrame): Pandas dataset
    columns (list): list of the columns in dataset to check outliers. All by default. 
    
    Returns:
    pd.DataFrame: clean dataset
    """
    masks = []
    if len(columns) < 1:
        columns = dataset.columns
        
    for col in columns:
        mean = dataset[col].mean()
        std = dataset[col].std()
        
        # 3 sigmas rules
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        #print(f"For col {col}, lower is {lower_bound} and upper is {upper_bound}")
        
        # Create mask
        mask = (dataset[col] >= lower_bound) & (dataset[col] <= upper_bound)
        masks.append(mask)

    # Apply mask in all columns
    # example: 
    # row1 = [0,1,1] -> [0]
    # row2 = [1,1,1] -> [1]
    final_mask = pd.concat(masks, axis=1).all(axis=1)
    filtered_df = dataset.loc[final_mask, :]
    return filtered_df

    

def test_xgb_finds_gpu(capsys):
    """Check if XGBoost finds the GPU."""
    dataset = load_diabetes()
    X = dataset["data"]
    y = dataset["target"]
    xgb_model = xgb.XGBRegressor(
        # If there is no GPU, the tree_method kwarg will cause either
        # - an error in `xgb_model.fit(X, y)` (seen with pytest) or
        # - a warning printed to the console (seen in Spyder)
        # It's unclear which of the two happens under what circumstances.
        tree_method="hist"
    )
    xgb_model.fit(X, y)


def get_size_in_mb(obj):
    """
    Calcule la taille d'un objet en mémoire en mégaoctets (Mo) en utilisant pympler.

    Paramètres :
    obj : L'objet dont vous souhaitez calculer la taille.

    Retourne :
    float : La taille de l'objet en Mo.
    """
    size_in_bytes = asizeof.asizeof(obj)
    size_in_mb = size_in_bytes / (1024 ** 2)
    return size_in_mb

