import numpy as np

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
