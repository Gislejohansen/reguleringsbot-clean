import geopandas as gpd
import fiona

# Se hvilke lag som finnes
layers = fiona.listlayers("teig.gml")
print("Tilgjengelige lag:", layers)

# Hent ut og last inn spesifikt lag
gdf = gpd.read_file("teig.gml", layer="Teig")

# Se kolonner som faktisk er med
print("Tilgjengelige kolonner:", gdf.columns)

# Forhåndsvis noen rader
print(gdf.head())
