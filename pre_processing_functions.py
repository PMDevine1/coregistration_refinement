import pdal
import geopandas as gpd
from shapely import wkt
import rasterio
from rasterio.mask import mask
import numpy as np
from rasterio.features import shapes
import fiona
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

def get_lidar_boundary(las_path, epsg):
    #Run PDAL hexbin filter and return largest polygon geometry
    
    pipeline = (
        pdal.Reader.las(filename=str(las_path), override_srs=f"EPSG:{epsg}")
        | pdal.Filter.hexbin(edge_length=2.0, smooth=True)
    )
    
    pipeline.execute()
    metadata = pipeline.metadata
    
    boundary_wkt = metadata['metadata']['filters.hexbin']['boundary']
    geom = wkt.loads(boundary_wkt)

    # Keep largest polygon if multipolygon
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)

    return geom


def save_geometry_to_shapefile(geom, epsg, output_path):
    #Save shapely geometry to shapefile.
    
    gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[geom], crs=f"EPSG:{epsg}")
    gdf.to_file(output_path)
    
    return gdf


def clip_raster_with_shape(raster_path, gdf, output_path):
    #Clip raster using GeoDataFrame geometry.
    
    with rasterio.open(raster_path) as src:
        # Reproject geometry to raster CRS
        gdf = gdf.to_crs(src.crs)

        out_image, out_transform = mask(src, gdf.geometry, crop=True)
        out_meta = src.meta.copy()

    # Update metadata
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    # Write output
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

def compute_ndvi(input_raster, output_path, red_band=5, nir_band=7, nodata_value=np.nan):
    # Compute NDVI on raster
    with rasterio.open(input_raster) as src:
        red = src.read(red_band).astype("float32")
        nir = src.read(nir_band).astype("float32")

        # Avoid divide warnings
        np.seterr(divide='ignore', invalid='ignore')

        ndvi = (nir - red) / (nir + red)

        # Handle invalid pixels
        invalid_mask = (nir + red) == 0
        ndvi[invalid_mask] = nodata_value

        # Prepare output metadata
        meta = src.meta.copy()
        meta.update({
            "count": 1,
            "dtype": "float32"
        })

        # Write output
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(ndvi, 1)

    print(f"NDVI saved to: {output_path}")

def extract_strong_vegetation(ndvi_raster, output_path):
    # Extract the veg from the raster
    with rasterio.open(ndvi_raster) as src:
        ndvi = src.read(1)

        # Threshold: >= 0.5 → 1, else 0
        mask = (ndvi >= 0.45).astype("uint8")

        # Copy metadata and update for single-band uint8 output
        meta = src.meta.copy()
        meta.update({
            "count": 1,
            "dtype": "uint8"
        })

        # Write output
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(mask, 1)

def raster_to_polygons(binary_raster, output_shp):
    # Convert raster veg to canopy polygons
    with rasterio.open(binary_raster) as src:
        image = src.read(1)
        transform = src.transform
        crs = src.crs

        results = (
            {"geometry": shape(geom), "properties": {"value": int(value)}}
            for geom, value in shapes(image, transform=transform)
        )

        schema = {
            "geometry": "Polygon",
            "properties": {"value": "int"}
        }

        with fiona.open(output_shp, "w",
                        driver="ESRI Shapefile",
                        crs=crs,
                        schema=schema) as dst:
            for feature in results:
                dst.write({
                    "geometry": mapping(feature["geometry"]),
                    "properties": feature["properties"]
                })

    print(f"Polygon shapefile saved to: {output_shp}")