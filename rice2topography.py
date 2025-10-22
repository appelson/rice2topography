# --- import essentials -----------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.affinity import scale, rotate
from shapely.ops import unary_union
from shapely.prepared import prep
from scipy.stats import gaussian_kde
from scipy.ndimage import distance_transform_edt, gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import os

# --- parameters / global setup --------------------------------------------
num_rice = 1000              # total grains
map_size = (1000, 1000)      # map dimensions
rice_length = 30
rice_width = 8
length_jitter = 0.3
rotation_jitter = 180        # full spin range
outline_radius = 15
num_clusters = 30
cluster_spread = 55          # how wide clusters scatter

# --- step 1: define world boundary ----------------------------------------
global_boundary = Polygon([
    (0, 0),
    (map_size[0], 0),
    (map_size[0], map_size[1]),
    (0, map_size[1])
])
prep_global = prep(global_boundary)

# --- step 2: generate cluster centers + rice grain seeds ------------------
cluster_centers = np.random.rand(num_clusters, 2) * map_size
points = []
for _ in range(num_rice * 2):
    cx, cy = cluster_centers[np.random.randint(0, num_clusters)]
    px = np.random.normal(cx, cluster_spread)
    py = np.random.normal(cy, cluster_spread)
    points.append((px, py))
points = np.array(points)

# --- step 3: build actual rice grains inside the world --------------------
grains = []
for px, py in points:
    attempts = 0
    while attempts < 50:
        attempts += 1
        grain = Point(px, py).buffer(1)
        # random scaling + rotation for organic variation
        length_scale = rice_length / 2 * (1 + np.random.uniform(-length_jitter, length_jitter))
        width_scale  = rice_width  / 2 * (1 + np.random.uniform(-length_jitter, length_jitter))
        angle = np.random.uniform(0, rotation_jitter)
        grain = scale(grain, xfact=length_scale, yfact=width_scale)
        grain = rotate(grain, angle, origin=(px, py))
        grain_outline = grain.buffer(outline_radius)

        # keep only if entire outline fits inside map
        if prep_global.contains(grain_outline):
            grains.append(grain)
            break

        # otherwise try new random location
        px, py = np.random.rand(2) * map_size

    if len(grains) >= num_rice:
        break

# --- step 4: merge all grains + generate land outline ---------------------
merged = unary_union(grains)
outline = merged.buffer(outline_radius)

# --- step 5: density-based height map (KDE) -------------------------------
x, y = np.array([g.centroid.x for g in grains]), np.array([g.centroid.y for g in grains])
kde = gaussian_kde([x, y])
xi, yi = np.mgrid[0:map_size[0]:200j, 0:map_size[1]:200j]
zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

# --- step 6: mask + fade edges for smooth land-water transition -----------
prep_outline = prep(outline)
mask = np.zeros_like(zi)
for i in range(xi.shape[0]):
    for j in range(yi.shape[1]):
        if prep_outline.contains(Point(xi[i, j], yi[i, j])):
            mask[i, j] = 1

dist_inside  = distance_transform_edt(mask)
dist_outside = distance_transform_edt(1 - mask)
fade_inside, fade_outside = 15, 50

inside_fade  = np.sqrt(np.clip(dist_inside / fade_inside, 0, 1))
outside_fade = np.exp(-dist_outside / fade_outside)
blend = np.clip(mask * inside_fade + (1 - mask) * outside_fade, 0, 1)
blend = gaussian_filter(blend, sigma=2)
zi = zi * blend

# --- step 7: visualize raw grain geometry ---------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, map_size[0]); ax.set_ylim(0, map_size[1])
ax.set_aspect('equal'); ax.axis('off')

for grain in grains:
    xg, yg = grain.exterior.xy
    ax.fill(xg, yg, color="#d1c29f", alpha=0.6)
plt.show()

# --- step 7b: 2D terrain-style heatmap ------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, map_size[0]); ax.set_ylim(0, map_size[1])
ax.set_aspect('equal'); ax.axis('off')

ax.imshow(np.rot90(zi), extent=[0, map_size[0], 0, map_size[1]], cmap='terrain', alpha=0.7)
ax.contour(xi, yi, zi, levels=8, colors='black', linewidths=0.4, alpha=0.4)

# outline the landmass
if outline.geom_type == "Polygon":
    xo, yo = outline.exterior.xy
    ax.fill(xo, yo, color="#e6d7b3", alpha=0.4, edgecolor="black", linewidth=1)
else:
    for geom in outline.geoms:
        xo, yo = geom.exterior.xy
        ax.fill(xo, yo, color="#e6d7b3", alpha=0.4, edgecolor="black", linewidth=1)
plt.show()

# --- step 8: build shaded 3D topographic surface --------------------------
z_scaled = gaussian_filter(zi, sigma=3)
z_scaled = z_scaled / z_scaled.max() * 60   # exaggerate vertical relief
land_fade = np.sqrt(np.clip(dist_inside / 15, 0, 1))
z_scaled[mask == 1] *= land_fade[mask == 1]
z_scaled[mask == 0] = 0

# define elevation color bands
water_color = np.array([0.18, 0.38, 0.70])
sand_color  = np.array([0.88, 0.80, 0.60])
grass_color = np.array([0.32, 0.50, 0.28])
rock_color  = np.array([0.42, 0.42, 0.42])
snow_color  = np.array([0.93, 0.93, 0.93])

# assign colors per height
colors = np.zeros(z_scaled.shape + (3,))
for i in range(z_scaled.shape[0]):
    for j in range(z_scaled.shape[1]):
        h = z_scaled[i, j]
        if mask[i, j] == 0:
            colors[i, j] = water_color
        elif h < 5:
            colors[i, j] = sand_color
        elif h < 30:
            colors[i, j] = grass_color
        elif h < 55:
            colors[i, j] = rock_color
        else:
            colors[i, j] = snow_color

# soft lighting for terrain shading
ls = LightSource(azdeg=315, altdeg=45)
shaded_rgb = ls.shade_rgb(colors, z_scaled, blend_mode='soft', vert_exag=1.2)

# plot the 3D landscape
fig = plt.figure(figsize=(12, 9))
ax3d = fig.add_subplot(111, projection='3d')

ax3d.plot_surface(
    xi, yi, z_scaled,
    rstride=1, cstride=1,
    facecolors=shaded_rgb,
    linewidth=0, antialiased=False, shade=False
)

# view + frame tuning
ax3d.set_xlim(0, map_size[0]); ax3d.set_ylim(0, map_size[1]); ax3d.set_zlim(0, 200)
ax3d.set_box_aspect((1, 1, 0.4))
ax3d.view_init(elev=35, azim=135)
ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([]); ax3d.set_axis_off()

plt.tight_layout()
plt.show()
