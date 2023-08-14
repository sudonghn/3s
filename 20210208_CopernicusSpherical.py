#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Alternative to OneStopShop.
# Copernicus DEM based, speherical coordinates.
# Adriaan van Natijne <A.L.vanNatijne@tudelft.nl>

# Recommended Conda environment:
# conda create -n sidx python=3.7 numpy geopandas rasterio numba matplotlib notebook

# Make it into a script:
# jupyter nbconvert --to python 20210208_CopernicusSpherical.ipynb
# And run
# python -OO 20210208_CopernicusSpherical.py


# In[1]:


import numpy as np
import geopandas as gpd
import rasterio as rio, rasterio.merge
import numba
from pathlib import Path
from matplotlib import pyplot as plt
from contextlib import ExitStack
import multiprocessing
from functools import partial
import datetime
import base64
import traceback


# ## Earth parameters

# In[3]:


# WGS84 ellipsoid parameters.
a = 6378137.0 # Equitorial radius[m]
b = 6356752.314245 # Polar radius [m]

# In[4]:


# Apperant local radius (at latitude phi).
# Source: https://en.wikipedia.org/wiki/Earth_radius#Geocentric_radius
R_E = lambda phi: np.sqrt(((a**2*np.cos(phi))**2 + (b**2*np.sin(phi))**2)/
                          ((a*np.cos(phi))**2+(b*np.sin(phi))**2))


# In[5]:


# Minimum slope, susceptible to landslides.
MIN_SLOPE = np.deg2rad(5)


# ## Satellite parameters

# In[6]:


# Satellite (Sentinel-1A) parameters
# Acquired 2021-02-08 from https://celestrak.com/satcat/tle.php?CATNR=39634 .
i = np.deg2rad(98.1810) # Inclination
k = 14.59198664         # Mean Motion (revolutions per day), daily frequency


# In[7]:


# Approximate ground track angle
# Capderou (2005)
heading = lambda phi: np.arctan((np.cos(i)-np.cos(phi)**2/k) /
                                np.sqrt(np.cos(phi)**2-np.cos(i)**2))


# In[8]:


# Incidence angles to evaluate the algorithm for.
INCIDENCE = (np.deg2rad(29.1), np.deg2rad(46)) # Sentinel-1


# ## Parameter check

# In[9]:


fig, ax = plt.subplots(1, 2, sharey=True, constrained_layout=True)
lat = np.linspace(-np.pi/2, np.pi/2, 100)

# Earth radius
ax[0].plot(R_E(lat), np.rad2deg(lat))
ax[0].set_title('Earth radius')
ax[0].set_xlabel('Radius [m]')
ax[0].set_ylabel(r'Latitude [$\degree$]')
ax[0].grid(True)

# Ground track angle
ax[1].plot(np.rad2deg(heading(lat)), np.rad2deg(lat), label='Ascending')
ax[1].plot(180-np.rad2deg(heading(lat)), np.rad2deg(lat), label='Descending')
ax[1].set_title('Satellite heading')
ax[1].set_xlabel(r'Heading [$\degree$]')
ax[1].legend(loc='best')
ax[1].grid(True)

del lat


# ## Data input/requirements
# 
# * Copernicus DEM <span style="color:red;">Will SRTM work as well?</span>
# * A tile index, readable by GeoPandas. The tile index should contain the extents of each SRTM tile available and reference the file in the `location` variable.
#   This index can be generated using `gdaltindex`.
#   Unfortunately, there are too many tiles to be handeled as arguments by bash.
#   ```bash
#   gdaltindex COP-DEM_GLO-30-DTED_PUBLIC/2019_1_S.shp COP-DEM_GLO-30-DTED_PUBLIC/2019_1/Copernicus_DSM_10_S*.dt2
#   gdaltindex COP-DEM_GLO-30-DTED_PUBLIC/2019_1_N.shp COP-DEM_GLO-30-DTED_PUBLIC/2019_1/Copernicus_DSM_10_N*.dt2
#   ogrmerge.py -single -o COP-DEM_GLO-30-DTED_PUBLIC/2019_1.shp COP-DEM_GLO-30-DTED_PUBLIC/2019_1_{N,S}.shp 
#   ```
#   
# Debugging:
# Is disabled by `python -O`.
# This will also disable the abundant `assert` statements that verify the consistent output between derivations.
# **Debugging will generate a lot (5× more) data and is much slower!**

# In[10]:

DEM_ROOT = Path('D:/DEM/')
print(DEM_ROOT.absolute())
DEM_TILES = gpd.read_file(DEM_ROOT / 's12.shp')
OUT_ROOT = DEM_ROOT / 'out'
print(DEM_TILES.columns)

assert DEM_ROOT.exists() and DEM_ROOT.is_dir()
assert isinstance(DEM_TILES, gpd.GeoDataFrame) and (len(DEM_TILES) > 0) and ('location' in DEM_TILES.columns)
assert OUT_ROOT.exists() and OUT_ROOT.is_dir()

print(1)
# In[11]:


def _tile_exists(f: str) -> bool:
    return (DEM_ROOT / f).exists()

with multiprocessing.Pool() as p:
    assert all(p.map(_tile_exists, DEM_TILES.location, chunksize=500))


# <span style="color:red;">Neighbourhood search documentation</span>
# Neighbourhood search does not wrap around dateline or poles!

# In[12]:

_buffer = DEM_TILES.copy()
print(11)
_buffer['geometry'] = _buffer.buffer(0.1, resolution=1)
print(11)
DEM_TILE_GROUPS = gpd.sjoin(_buffer, DEM_TILES, 'inner', 'intersects', lsuffix='buffer', rsuffix='tile')
print(11)

if __debug__:
    _buffer.to_file(OUT_ROOT / 'groups.gpkg', driver='GPKG', layer='buffer')
    DEM_TILE_GROUPS.to_file(OUT_ROOT / 'groups.gpkg', driver='GPKG', layer='groups')
    
DEM_TILE_GROUPS = DEM_TILE_GROUPS.groupby('location_buffer', sort=True)

print(11)
del _buffer

# In[13]:


@numba.njit(parallel=True, fastmath=True)
def _shadow(h: np.ndarray,
            mask: np.ndarray,
            gamma: np.ndarray,
            incidence: np.float64,
            R: np.ndarray,
            rlon: np.float64,
            rlat: np.float64,
            contaminate: bool = False
            ) -> np.ndarray:
    # Output shadow mask
    shadow = np.empty_like(h, dtype=np.int8)
    shadow.fill(-1)

    # Maximum height to search for (in this tile)
    h_max = np.max(h)

    # Loop over all pixels.
    for m in numba.prange(h.shape[0]):
#         m = int(m) # Numba indexing requirement, cast to int.

        for n in numba.prange(h.shape[1]):
#             n = int(n) # Numba indexing requirement, cast to int.
            print(666)

            # Not steep enough for a landslide. Skip.
            if mask[m, n] == False:
                shadow[m, n] = 0

            # Determine required resolution (in ratio of pixel)
    #         rlon = np.deg2rad(abs(merged[1].a))
    #         rlat = np.deg2rad(abs(merged[1].e))
            rs = min(rlon, rlat)

            # Direction of the satellite (in radians!)
            dE = np.sin(gamma[m, n])*rs
            dN = -np.cos(gamma[m, n])*rs
            # Approximate change in meters for each step
            dNE = R[m, n]*np.arccos(np.cos(dE)*np.cos(dN))
            dU = np.tan(incidence)*dNE

            # Step size in (fractional) pixels, for convenience.
            _dE = dE/rlon
            _dN = dN/rlat

            # Start looking for surface topography
            m_, n_ = 0.0, 0.0 # Offset (fractional pixels)
            h_ = h[m, n]      # Start height
            while (m+m_ > 0) and (n+n_ > 0) and (np.rint(m+m_) < h.shape[0] -1) and (np.rint(n+n_) < h.shape[1] -1):
                # 'Step' towards satellite
                m_ += _dN
                n_ += _dE
                h_ += dU
                # Add curvature correction
                # Calculate spherical degrees from origin, to compensate for horizon curvature.
#                 dw_ = np.cos(m_)*np.cos(n_)
#                 h_e = (R[m, n] +h[m, n])/np.cos(dw_)-R[m, n]
                h_e = 0 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                if h[int(np.rint(m +m_)), int(np.rint(n +n_))] >= (h_ +h_e):
                    shadow[m, n] = 1 # Shadow pixel
                    if contaminate:
                        shadow[int(np.rint(m +m_)), int(np.rint(n +n_))] = 1 # Cause
                    break
                elif (h_ +h_e) > h_max:
                    # Shadow could only be induced by an object higher than the highest object in the map.
                    # So, no shadow.
                    shadow[m, n] = 0
                    break
            else:
                if shadow[m, n] < 0:
                    # If the loop did not set the pixel, it must be free of shadow.
                    shadow[m, n] = 0

    return shadow


# In[85]:


def _job(args: tuple):
    # Expand variables.
    print(6)
    ti, (tile, neighbours) = args
    ti, (tile, neighbours) = (1, (tile, DEM_TILE_GROUPS.get_group(tile)))

    # Name template.
    tile_out_name = lambda prefix: OUT_ROOT / ('{!s}_{:s}.tiff'.format(prefix, tile.rpartition('/')[-1][-22:-11]))

    # Check if it exists already, unless debugging is active.
    if not __debug__ and tile_out_name('S').exists():
        return None

    print(16)
    # GDAL environment, including optimisations for DTED reading.
    with rio.Env(GDAL_CACHEMAX=2500,
                 GDAL_DTED_SINGLE_BLOCK=True,
                 REPORT_COMPD_CS=True,
                 NUM_THREADS='ALL_CPUS'), \
         ExitStack() as stack:
        # Open all (neighbouring) tiles.
        # The ExitStack provides a with() statement for all open files.
        tiles = [stack.enter_context(rio.open(DEM_ROOT / f)) for f in neighbours.location_tile]

        # Select the main tile from the series of neighbours.
        # This is later used to read the properties from the source file.
        i_t = np.nonzero((neighbours.index == neighbours.index_tile))[0]
        assert len(i_t) == 1
        i_t = i_t[0]

        # Intermitent and output profiles.
        # Profile for exports
        tile_profile = partial(rasterio.profiles.DefaultGTiffProfile,
                               count=1,
                               crs=tiles[i_t].profile['crs'],
                               width=tiles[i_t].profile['width'],
                               height=tiles[i_t].profile['height'],
                               dtype=np.float32, nbits=16,
                               transform=tiles[i_t].profile['transform'],
                               nodata=tiles[i_t].profile['nodata'],
                               compression=rio.enums.Compression.lzw)

        print(166)
        # Extract buffer area.
        buffer_bounds = neighbours.iloc[i_t].geometry.bounds

        # Combine with adjacent tiles.
        merged = rio.merge.merge(tiles,
                                 bounds=buffer_bounds,
                                 res=tiles[i_t].res,
    #                                  dtype=np.float32,
    #                                  indexes=(1,),
    #                                  resampling=rio.enums.Resampling.bilinear,
                                 nodata=tiles[i_t].profile['nodata'],
                                 precision=25, # https://github.com/mapbox/rasterio/issues/2048
    #                                  dst_path=tile_out_name('DEM') if __debug__ else None
                                )
        h = merged[0][0, ...] # alias

        print(1666)
        # Determine window spanning input tile.
        win = rio.windows.from_bounds(*tiles[i_t].bounds,
                                      transform=merged[1], height=tiles[i_t].height, width=tiles[i_t].width)
        win = (slice(int(np.rint(win.row_off)), int(np.rint(win.row_off+win.height))),
               slice(int(np.rint(win.col_off)), int(np.rint(win.col_off+win.width))))
        # Verify input tile and window are the same.
        assert np.allclose(tiles[i_t].read(1), h[win])

        if __debug__:
            # Output profile for merged raster, usefull for debugging.
            merged_profile = partial(rasterio.profiles.DefaultGTiffProfile,
                                     count=1,
                                     crs=tiles[i_t].profile['crs'],
                                     width=h.shape[1],
                                     height=h.shape[0],
                                     dtype=np.float32,
                                     transform=merged[1],
                                     nodata=tiles[i_t].profile['nodata'],
                                     compression=rio.enums.Compression.lzw)

        # TODO: Replace with dst_path from merge.
        if __debug__:
            with rio.open(tile_out_name('DEM'), 'w', **merged_profile(dtype=np.int16)) as o:
                o.write_band(1, h)

        # Create latitude grid.
        lon, lat = merged[1] *(np.mgrid[:h.shape[1], :h.shape[0]] +0.5)
        lat = np.deg2rad(lat.T)

        if __debug__:
            lon = np.deg2rad(lon.T)

            with rio.open(tile_out_name('LONLAT'), 'w', **merged_profile(count=2, nbits=16)) as o:
                o.write_band(1, np.rad2deg(lon).astype(np.float32))
                o.write_band(2, np.rad2deg(lat).astype(np.float32))

        del lon

        # Approximate local radius of the Earth.
        R = R_E(lat)

        if __debug__:
            with rio.open(tile_out_name('R'), 'w', **merged_profile()) as o:
                o.write_band(1, R.astype(np.float32))

        # Approximate DEM resolution.
        rN = R*np.deg2rad(abs(merged[1].e))
        rE = R*np.deg2rad(abs(merged[1].a))*np.cos(lat)

        if __debug__:
            with rio.open(tile_out_name('r'), 'w', **merged_profile(count=2, nbits=16)) as o:
                o.write_band(1, rN.astype(np.float32))
                o.write_band(2, rE.astype(np.float32))

        # Estimate satellite heading (ascending).
        gamma = heading(np.deg2rad(lat))

        if __debug__:
            with rio.open(tile_out_name('GAMMA'), 'w', **merged_profile(nbits=16)) as o:
                o.write_band(1, np.rad2deg(gamma).astype(np.float32))

        # Calculate gradient(s)
        p = np.zeros_like(h, dtype=np.float64)
        p[1:-1,1:-1] = ((h[:-2,2:]+2*h[1:-1,2:]+h[2:,2:])-(h[:-2,:-2]+2*h[1:-1,:-2]+h[2:,:-2]))/(8*rE[1:-1,1:-1])
        q = np.zeros_like(h, dtype=np.float64)
        q[1:-1,1:-1] = ((h[:-2,2:]+2*h[:-2,1:-1]+h[:-2,:-2])-(h[2:,:-2]+2*h[2:,1:-1]+h[2:,2:]))/(8*rN[1:-1,1:-1])

        if __debug__:
            with rio.open(tile_out_name('pq'), 'w', **merged_profile(count=3, nbits=16)) as o:
                o.write_band(1, p.astype(np.float32))
                o.write_band(2, q.astype(np.float32))
                # Export length of downslope vector (should always be 1).
                o.write_band(3, np.sqrt((p**2+q**2+(p**2+q**2)**2)/np.abs((1+p**2+q**2)*(p**2+q**2))).astype(np.float32))

        # Veriy downslope vector is a unit-vector.
        # TODO: verify only non-NaN pixels
    #     assert np.allclose(np.sqrt((p**2+q**2+(p**2+q**2)**2)/np.abs((1+p**2+q**2)*(p**2+q**2))), 1, equal_nan=True)

        # Slope filter
        # Will only mark slopes within the tile boundaries, to accelerate shadow/layover search.
        slope_mask = np.zeros_like(h, dtype=np.bool)
        slope_mask[win] = np.hypot(p[win], q[win]) >= np.tan(MIN_SLOPE)

        if __debug__:
            # Calculate slope (only for debugging).
            beta = np.arctan(np.hypot(p, q))

            # Calculate aspect (only for debugging).
            alpha = np.arctan2(q, p)
            alpha = (alpha-np.pi/2)%(2*np.pi) # wrt. North

            with rio.open(tile_out_name('sa'), 'w', **merged_profile(count=2, nbits=16)) as o:
                o.write_band(1, np.rad2deg(beta).astype(np.float32))
                o.write_band(2, np.rad2deg(alpha).astype(np.float32))

        print(16666)
        # Quick check on radar geometry.
        assert np.allclose(np.linalg.norm(np.dstack([np.cos(gamma)*np.sin(INCIDENCE[0]),
                                                     -np.sin(gamma)*np.sin(INCIDENCE[0]),
                                                     -np.cos(INCIDENCE[0])*np.ones_like(gamma)]), axis=2), 1)

        # Calculate the sensitivity, based on the algebraic expressions.
        # First, calculate basic building blocks to accelerate the process.
        p2, q2 = p**2, q**2
        S_denom = np.sqrt((1+p2+q2)*(p2+q2)) # = sqrt(1+p2+q2)*sqrt(p2+q2)
        S_a = (p2+q2)/S_denom
        del p2, q2 # Save some memory
        # S_b = (-p*np.cos(gamma)+q*np.sin(gamma))/S_denom # Asc/Dsc specific
        # This way, the equation is 'simplified' to:
        # S = (S_a*np.cos(theta)+S_b*np.sin(theta))

        # Calculate sensitivity, ascending.
        S_b = (-p*np.cos(gamma)+q*np.sin(gamma))/S_denom
        S_asc = np.min(np.abs(np.dstack([S_a*np.cos(_i)+S_b*np.sin(_i) for _i in INCIDENCE])), axis=2)

        assert np.nanmax(S_asc) <= 1 and np.nanmin(S_asc) >= 0

        # Calculate sensitivity, descending.
        S_b = (-p*np.cos(np.pi-gamma)+q*np.sin(np.pi-gamma))/S_denom
        S_dsc = np.min(np.abs(np.dstack([S_a*np.cos(_i)+S_b*np.sin(_i) for _i in INCIDENCE])), axis=2)

        assert np.nanmax(S_dsc) <= 1 and np.nanmin(S_dsc) >= 0

        del S_a, S_b, S_denom # Save some memory

        print(166666)
        if __debug__:
            with rio.open(tile_out_name('S_RAW'), 'w', **merged_profile(count=3, nbits=16)) as o:
                o.write_band(1, S_asc.astype(np.float32))
                o.write_band(2, S_dsc.astype(np.float32))
                # Tentative maximum (not accounting for shadow/layover).
                o.write_band(3, np.maximum(S_asc, S_dsc).astype(np.float32))

        # Estimate shadow.
        ShL_asc = _shadow(h, slope_mask, gamma+(3/2)*np.pi, np.pi/2-np.max(INCIDENCE), R,                           np.deg2rad(abs(merged[1].a)), np.deg2rad(abs(merged[1].e)), False)
        ShL_dsc = _shadow(h, slope_mask, np.pi-gamma+(3/2)*np.pi, np.pi/2-np.min(INCIDENCE), R,                           np.deg2rad(abs(merged[1].a)), np.deg2rad(abs(merged[1].e)), False)

        # Estimate layover.
        ShL_asc += 2*_shadow(h, slope_mask, gamma+np.pi/2, np.pi/2-np.max(INCIDENCE), R,                              np.deg2rad(abs(merged[1].a)), np.deg2rad(abs(merged[1].e)), True)
        ShL_dsc += 2*_shadow(h, slope_mask, np.pi-gamma+np.pi/2, np.pi/2-np.max(INCIDENCE), R,                              np.deg2rad(abs(merged[1].a)), np.deg2rad(abs(merged[1].e)), True)

        if __debug__:
            with rio.open(tile_out_name('ShL'), 'w', **merged_profile(count=2, dtype=np.int8, nodata=0)) as o:
                o.write_band(1, ShL_asc)
                o.write_band(2, ShL_dsc)

        # Set sensitivity in shadow/layover regions to zero.
        S_asc[ShL_asc > 0] = 0
        S_dsc[ShL_dsc > 0] = 0

        # Remove areas flatter than the minimum slope.
        S_asc[~slope_mask] = np.nan
        S_dsc[~slope_mask] = np.nan

        # Clip to original raster, combine results.
        S = np.maximum(S_asc[win], S_dsc[win])

        # Save results.
        with rio.open(tile_out_name('S'), 'w', **tile_profile(count=3, dtype=np.float32, nbits=16)) as o:
            o.write_band(1, S_asc[win].astype(np.float32))
            o.write_band(2, S_dsc[win].astype(np.float32))
            o.write_band(3, S.astype(np.float32))

            # Calculate histogram/statistics.
            bins = np.linspace(0, 1, 51)
            r2   = rN[win]*rE[win]
            _to_str = lambda a: base64.standard_b64encode(a.tobytes()).decode('ASCII')
            o.update_tags(ns='SIDX',
                          bins=_to_str(np.linspace(0, 1, 51)),
                          S=_to_str(np.histogram(S, bins=bins, weights=r2, density=False)[0]),
                          S_asc=_to_str(np.histogram(S_asc[win], bins=bins, weights=r2, density=False)[0]),
                          S_dsc=_to_str(np.histogram(S_dsc[win], bins=bins, weights=r2, density=False)[0]),
                          Sh_asc=(ShL_asc[win] & 1).mean(),
                          L_asc=(ShL_asc[win] & 2).mean(),
                          Sh_dsc=(ShL_dsc[win] & 1).mean(),
                          L_dsc=(ShL_dsc[win] & 2).mean(),
                          ShL=((ShL_asc[win] > 0) & (ShL_dsc[win] > 0)).mean(),
                          generated=datetime.datetime.now().astimezone().isoformat())

            # Decode:
            # with rio.open(tile_out_name('S'), 'r') as o:
            #   print(np.frombuffer(base64.standard_b64decode(o.tags(ns='SIDX')['bins'])))
            
    return tile

def _multi_job(args: tuple):
    # Wrapper for multiprocessing jobs (ignore errors and warnings).
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            return _job(args)
    except Exception as e:
        print(args[1][0], e)
        print(traceback.format_exc())
        return None


# In[86]:


if __debug__:
    # Process single tile, for testing.
    tile = 'COP-DEM_GLO-30-DTED_PUBLIC/2019_1/Copernicus_DSM_10_N47_00_E011_00_DEM.dt2'
    _job((1, (tile, DEM_TILE_GROUPS.get_group(tile))))
else:
    # Massive processing.
    with multiprocessing.Pool(4, maxtasksperchild=100) as p:
        for p, r in enumerate(p.imap_unordered(_multi_job, enumerate(DEM_TILE_GROUPS), chunksize=5)):
            print('{: 6d}/{:d} {!s}'.format(p, len(DEM_TILE_GROUPS), r))

