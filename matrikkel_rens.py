import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 1. Les inn CSV-filen
df = pd.read_csv("matrikkel_adresse.csv", sep=";")


# 2. Lag geometrikolonne fra Øst/Nord (X/Y)
# Bytt ut kolonnenavnene hvis de heter noe annet i din fil
gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df["Øst"], df["Nord"])],
    crs="EPSG:25833"  # UTM sone 33N (meter)
)

# 3. Konverter til lat/lon (WGS84)
gdf = gdf.to_crs("EPSG:4326")

# 4. Ekstraher lat/lon til egne kolonner
gdf["longitude"] = gdf.geometry.x
gdf["latitude"] = gdf.geometry.y

# 5. Lagre til ny CSV
gdf.drop(columns="geometry").to_csv("matrikkel_adresse_latlon.csv", index=False)

print("Ferdig! Fil lagret som 'matrikkel_adresse_latlon.csv'")
