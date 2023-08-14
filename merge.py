# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 22:56:12 2023

@author: sudonghn_14513887214
"""
from osgeo_utils.ogrmerge import ogrmerge

src_datasets = [
    r"D:\DEM\subdem\SRTM0.shp",
    r"D:\DEM\subdem\SRTM1.shp"
]



r = ogrmerge(
    src_datasets=src_datasets,
    dst_filename=r"D:\DEM\s12.shp",
    single_layer=True,
    lco=["ENCODING=UTF-8"]
)

print(r)