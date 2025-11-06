import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

#SKILL
if False:
    print('skills')
    # Define the colors for minimum, central, and maximum values
    min_color = '#40004b'
    central_color = '#dfdfdf' #'#f5f5f5'
    max_color = '#00441b'
    ## Create a discrete divergent colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', [min_color, central_color, max_color])
    # Define the number of colors
    bounds = [-0.45,-0.35,-0.25,-0.15,-0.05,0.05,0.15,0.25,0.35,0.45]
    n_colors = len(bounds)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 4))

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    # Get the colors from the colormap
    colors = [cmap(i) for i in np.linspace(0, 1, norm.N+1)]
    # Convert the colors to hex
    hex_colors = [mcolors.to_hex(color) for color in colors]
    # Print the hex colors
    print("Hex colors used in the legend:")
    print(hex_colors)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation='horizontal',
                 label="Discrete intervals with extend='both' keyword")


    plt.show()
    plt.close()
# # P dry
if False:
    print('p dry')
    # # Define the colors for minimum, central, and maximum values
    min_color = '#FFFF00'
    max_color = '#FF0000'
    ## Create a discrete divergent colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', [min_color, max_color])
    # Define the number of colors
    bounds = [.4,.5,.6,.7,.8,.9,1]
    n_colors = len(bounds)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 4))

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')
    # Get the colors from the colormap
    colors = [cmap(i) for i in np.linspace(0, 1, norm.N-1)]
    # Convert the colors to hex
    hex_colors = [mcolors.to_hex(color) for color in colors]
    # Print the hex colors
    print("Hex colors used in the legend:")
    print(hex_colors)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation='horizontal',
                 label="Discrete intervals with extend='neither' keyword")
    plt.show()
    plt.close()

# # P norm
if False:
    print('p normal')
    # # Define the colors for minimum, central, and maximum values
    min_color = '#dfdfdf'
    max_color = '#000000'
    ## Create a discrete divergent colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', [min_color, max_color])
    # Define the number of colors
    bounds = [.4, .5, .6, .7, .8, .9, 1]
    n_colors = len(bounds)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 4))

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')
    # Get the colors from the colormap
    colors = [cmap(i) for i in np.linspace(0, 1, norm.N - 1)]
    # Convert the colors to hex
    hex_colors = [mcolors.to_hex(color) for color in colors]
    # Print the hex colors
    print("Hex colors used in the legend:")
    print(hex_colors)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation='horizontal',
                 label="Discrete intervals with extend='neither' keyword")
    plt.show()
    plt.close()

# # P wet
if False:
    print('p normal')
    # # Define the colors for minimum, central, and maximum values
    min_color = '#7DF9FF'
    max_color = '#0000FF'
    ## Create a discrete divergent colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', [min_color, max_color])
    # Define the number of colors
    bounds = [.4, .5, .6, .7, .8, .9, 1]
    n_colors = len(bounds)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 4))

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')
    # Get the colors from the colormap
    colors = [cmap(i) for i in np.linspace(0, 1, norm.N - 1)]
    # Convert the colors to hex
    hex_colors = [mcolors.to_hex(color) for color in colors]
    # Print the hex colors
    print("Hex colors used in the legend:")
    print(hex_colors)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation='horizontal',
                 label="Discrete intervals with extend='neither' keyword")
    plt.show()
    plt.close()