import numpy as np
import pyvista as pv
import os

# Grid
p =  2
q = 3
s = 2
x  = np.linspace(-p, p, 400)
y = np.linspace(-q, q, 400)
z = np.linspace(-s, s, 200)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
l1 = 2
l2 = 3
n = 0.038
m = 0.986
a = 2
a_1 = 2
a_2 = 1.7
a_3 = 2.63
a_4 = 1.45
f_1 = 2*Z**2 - 2*m**(a*np.cosh(a_1*X) + np.cosh(a_2*Y))
s_1 = n**((X - l1)**2 + (Y - l2)**2 ) + 0.1*(np.sin(((X - l1)**2 + (Y - l2)**2 + Z) * np.exp(0.1*(X+Y)))*np.exp(0.1*(X+Y)))
s_2 = n**((X + l1)**2 + (Y - l2)**2 ) + 0.1*(np.sin(((X + l1)**2 + (Y - l2)**2 + Z) * np.exp(0.1*(-X+Y)))*np.exp(0.1*(-X+Y)))
s_3 = n**((X - l1)**2 + (Y + l2)**2 ) + 0.1*(np.sin(((X - l1)**2 + (Y + l2)**2 + Z) * np.exp(0.1*(X-Y)))*np.exp(0.1*(X-Y)))
s_4 = n**((X + l1)**2 + (Y + l2)**2 ) + 0.1*(np.sin(((X + l1)**2 + (Y + l2)**2 + Z) * np.exp(0.1*(-X-Y)))*np.exp(0.1*(-X-Y)))

s_t = s_1 + s_2 + s_3 + s_4

with np.errstate(invalid='ignore'): b_1 = np.arcsin(0.001*X**8 + 0.0001*Y**8 + 0.0001*Z**8)


o_1 = np.sin(-3 * np.cos(2*X*Y*Z) * np.cos(0.5*X*Y)) * np.arcsin( np.cos((X**2)*(Y**2)*(Z)) * np.cos(0.1*X*Y))


o_2 = (Z)*(np.sin(a_3*(X**2) - a_4*(Y**2)))*(np.sin(-a_3*(X**2) + a_4*(Y**2))) 




F =  -3*Z + 2*(Z**2) - Z**3 - s_t + 5*f_1**2 + 0.1*b_1 - 5*(o_2**2)*(o_1)


#--- APPLY MASK (for only region output) ---
mask = (np.abs(X) <= p) & (np.abs(Y) <= 9)
F = np.where(mask, F, np.nan)   # use NaN, NOT 0

# Create grid
grid = pv.StructuredGrid(X, Y, Z)
grid["F"] = F.ravel(order="F")

# Gradient
grid = grid.compute_derivative(scalars="F", gradient=True)

# Extract surface
#surface = grid.contour([0]) #.... level set 0
region = grid.threshold(value=0, scalars="F", invert=True)

# EXPORT STL !!!
#region_surface = region.extract_surface()
    # Clean it up before exporting — important for Rhino
#region_surface = region_surface.clean()          # remove duplicate points
#region_surface = region_surface.triangulate()    # Rhino prefers all triangles

#region_surface.save("D:\implicit\output_obj\Rec_slab_1.stl")

# X-gradient
#surface["grad_x"] = surface["gradient"][:, 0] #.... level set 0
region["grad_z"] = region["gradient"][:, 2]


# Plot (with out the scale)
#surface.plot(scalars="grad_x", cmap="coolwarm", show_scalar_bar=False,) #.... level set 0
#region.plot(scalars="grad_x", cmap="coolwarm", show_scalar_bar=False,)

# Plot
plotter = pv.Plotter()

plotter.add_mesh(
    region,
    scalars="grad_z",
    cmap="seismic",
    show_scalar_bar=False,
    )

# Coordinate grid / axes
plotter.show_grid(
    xtitle="X",
    ytitle="Y",
    ztitle="Z",
    font_size=10,
    grid="back",        # 'back', 'front', or 'both'
    location="outer",   # 'outer', 'inner', or 'all'
    ticks="outside",    # 'outside', 'inside', or 'both'
    n_xlabels=5,
    n_ylabels=5,
    n_zlabels=5,
)

# Optional: add XYZ axis widget in corner
plotter.add_axes(
    xlabel="X", ylabel="Y", zlabel="Z",
    line_width=3,
)

plotter.show()
