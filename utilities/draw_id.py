#=== vẽ label chữ trên OBB
import open3d 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops

def set_equal_aspect(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    max_range = max(
        abs(xlim[1] - xlim[0]),
        abs(ylim[1] - ylim[0]),
        abs(zlim[1] - zlim[0])
    ) / 2

    mid_x = np.mean(xlim)
    mid_y = np.mean(ylim)
    mid_z = np.mean(zlim)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def visualize_matplotlib(current_list_of_visuals, current_list_of_names, folder, frameId):

    fig = plt.figure(figsize=(20, 20))
    plt.style.use('dark_background')
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.dist = 7  # zoom out
    ax.set_xlim(0,20)
    ax.set_ylim(0,20)
    ax.set_zlim(0,25)
    #set_equal_aspect(ax)
    

    ax.view_init(elev=30, azim=-30)
    # Plot point clouds
    for item in current_list_of_visuals:
        if isinstance(item, (open3d.geometry.PointCloud, open3d.cuda.pybind.geometry.PointCloud)):
            pts = np.asarray(item.points)
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=0.5, c='white')

    edges = [
    (0,1),(1,7),(7,2),(2,0),   # bottom
    (3,6),(6,4),(4,5),(5,3),   # top
    (0,3),(1,6),(7,4),(2,5)    # verticals
]

    name_index = 0

    # Plot OBBs + labels
    for item in current_list_of_visuals:
        if isinstance(item, (open3d.geometry.OrientedBoundingBox, open3d.cuda.pybind.geometry.OrientedBoundingBox)):

            obb = item
            corners = np.asarray(obb.get_box_points())
           # print("corners:", corners)
            if 'p' in current_list_of_names[name_index]:
                color = 'cyan'
            elif 'v' in current_list_of_names[name_index]:
                color = 'yellow'
            else:
                color = 'red'
            for s, e in edges:
                p1 = corners[s]
                p2 = corners[e]
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color,
                    linewidth=1.5
                )


            # Label
            top_center = corners[4:8].mean(axis=0)
            label_pos = top_center + np.array([0, 0, 0.5])

            ax.text(label_pos[0], label_pos[1], label_pos[2],
                    str(current_list_of_names[name_index]),
                    color=color, fontsize=14, fontweight='bold')

            name_index += 1
    # --- Draw GLOBAL coordinate axes ---
    # origin = np.array([0, 0, 0])
    # axis_len = 5.0   # adjust as needed

    # # X axis (red)
    # ax.plot([origin[0], axis_len],
    #         [origin[1], 0],
    #         [origin[2], 0],
    #         color='red', linewidth=3)

    # # Y axis (green)
    # ax.plot([origin[0], 0],
    #         [origin[1], axis_len],
    #         [origin[2], 0],
    #         color='green', linewidth=3)

    # # Z axis (blue)
    # ax.plot([origin[0], 0],
    #         [origin[1], 0],
    #         [origin[2], axis_len],
    #         color='blue', linewidth=3)

# # Optional: label axes
#     ax.text(axis_len, 0, 0, 'X', color='red', fontsize=14)
#     ax.text(0, axis_len, 0, 'Y', color='green', fontsize=14)
#     ax.text(0, 0, axis_len, 'Z', color='blue', fontsize=14)


    plt.savefig(f"{folder}/frame_{frameId}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
