import matplotlib.pyplot as plt
import ternary

from viz.plots import *

fontsize = 12
offset = 0.2
scale = 100
data = generate_heatmap_data(scale)
figure, tax = ternary.figure(scale=scale)
tax.gridlines(color="white", multiple=10)
tax.heatmap(data, style="hexagonal", use_rgba=True, colorbar=False)
tax.boundary()
tax.right_corner_label("", fontsize=fontsize, offset=offset)
tax.top_corner_label("", fontsize=fontsize, offset=offset)
tax.left_corner_label("", fontsize=fontsize, offset=offset)
tax.left_axis_label("P(algorithm$_2$ wins)", fontsize=fontsize, offset=offset)
tax.right_axis_label("P(ROPE)", fontsize=fontsize, offset=offset)
tax.bottom_axis_label("P(algorithm$_1$ wins)", fontsize=fontsize, offset=offset)
tax.horizontal_line(90, linewidth=2., color='white')
tax.left_parallel_line(90, linewidth=2., color='white')
tax.right_parallel_line(90, linewidth=2., color='white')
tax.get_axes().text(4, 3, "S", color='white')
tax.get_axes().text(49, 35, "I", color='white')
tax.get_axes().text(49, 80, "E", color='white')
tax.get_axes().text(94, 3, "L", color='white')

tax.set_title("")
tax.get_axes().axis('off')
tax.ticks(axis='lbr', multiple=10, linewidth=1, offset=0.025)
plt.show()

plt.savefig('./figures/ternary_plot.png', dpi=350)