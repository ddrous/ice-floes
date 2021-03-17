"""
Options file for the plot.
"""

size_edges = 0.5
triangles = {'linewidth':size_edges}

size_fractured_nodes = 2
plot_mid_nodes = False
mid_nodes = {'marker':'D', 'color':'red', 'markersize':size_fractured_nodes}
plot_tip_nodes = False
tip_nodes = {'marker':'s', 'color':'yellow', 'markersize':size_fractured_nodes}

size_refinement_marker = 0.3
size_refinement_edges = 1
plot_fractured_triangles = False
local_refinement_left = {'marker':'x', 'color':'blue', 'markersize':size_refinement_marker} 
local_refinement_right = {'marker':'x', 'color':'green', 'markersize':size_refinement_marker}
local_refinement_edges = {'linewidth':size_refinement_edges, 'color':'m'}

size_fracture = 1.5
fracture_segments = {'color':'red', 'linewidth':size_fracture}
fracture_points = {'marker':'+', 'color':'k'}
plot_fracture_points = False

circle_inclusion = {'color':'r', 'alpha':0.5}

savefig = {'dpi':600, 'frameon':False, 'bbox_inches':'tight', 'pad_inches':0, 'transparent':True}
