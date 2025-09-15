import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
import os
from tqdm import tqdm

def georeference_png_to_geotiff(png_path, shapefile_path, output_geotiff_path):
    """
    Georeferences a PNG image using a shapefile and saves it as a GeoTIFF.

    Args:
        png_path (str): Path to the input PNG image.
        shapefile_path (str): Path to the shapefile (.shp) containing geospatial reference data.
        output_geotiff_path (str): Path to save the output GeoTIFF file.
    """
    # Step 1: Read the shapefile to extract the spatial extent and CRS
    gdf = gpd.read_file(shapefile_path)
    if len(gdf) != 1:
        raise ValueError("Shapefile should contain exactly one feature for georeferencing.")
    bounds = gdf.total_bounds  # Returns [minx, miny, maxx, maxy]
    crs = gdf.crs

    # Step 2: Open the PNG to get dimensions and read image data
    with rasterio.open(png_path) as src:
        width = src.width
        height = src.height
        count = src.count  # Number of bands (e.g., 3 for RGB)
        dtype = src.dtypes[0]  # Data type (e.g., uint8)
        image_data = src.read()  # Read all bands

    # Step 3: Create the affine transform using the bounds and image dimensions
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

    # Step 4: Define the GeoTIFF profile
    profile = {
        'driver': 'GTiff',
        'width': width,
        'height': height,
        'count': count,
        'dtype': dtype,
        'crs': crs,
        'transform': transform,
        'nodata': None  # Adjust if your image has a specific nodata value
    }

    # Step 5: Write the image data to the GeoTIFF file
    with rasterio.open(output_geotiff_path, 'w', **profile) as dst:
        dst.write(image_data)

# Example usage
# png_path = 'Src/Habiganj.png'
# shapefile_path = 'Src/ADM2_EN_Habiganj.shp'
# output_geotiff_path = 'Src/georeferenced_Habiganj_image.tif'

cwd = os.getcwd()
root_path = os.path.join(os.getcwd() + "/Base_patches")
distNames = sorted([name for name in os.listdir(root_path) if os.path.isfile(os.path.join(root_path, name))])

for district in tqdm(distNames, desc="Processing districts"):
    print("Processing Start of " + district, end="\n")
 
    png_path = os.path.join(root_path + "/" + district)
    name = district[:-4]
    shapefile_path = cwd + "/64_shapes" + "/ADM2_EN_" + name + ".shp" 
    output_geotiff_path = cwd + "/geotiff_districts/geotiff_" + name + ".tif"

    # print(png_path, shapefile_path, output_geotiff_path, sep='##')
    georeference_png_to_geotiff(png_path, shapefile_path, output_geotiff_path)
    print("Processing Completed of " + district, end="\n")


# not_done = [name for name in os.listdir(root_path) if os.path.isfile(os.path.join(root_path, name)) and name not in distNames]
# print(not_done)


