# Visualization
import glooey
import pyglet
import trimesh
import trimesh.transformations as tf
import trimesh.viewer
import numpy as np

# From https://github.com/wkentaro/octomap-python/blob/main/examples/insertPointCloud.py
def labeled_scene_widget(scene, label):
    vbox = glooey.VBox()
    vbox.add(glooey.Label(text=label, color=(255, 255, 255)), size=0)
    vbox.add(trimesh.viewer.SceneWidget(scene))
    return vbox

# From https://github.com/wkentaro/octomap-python/blob/main/examples/insertPointCloud.py
def visualize(
    occupied, empty, K, width, height, resolution, aabb
):
    print("Make window")
    window = pyglet.window.Window(
        width=int(1920 * 0.9), height=int(1080 * 0.9)
    )

    @window.event
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.Q:
                window.on_close()

    gui = glooey.Gui(window)
    hbox = glooey.HBox()
    hbox.set_padding(5)

    print("Set camera")
    camera = trimesh.scene.Camera(
        resolution=(width//2, height), focal=(K[0, 0], K[1, 1])
    )
    camera_marker = trimesh.creation.camera_marker(camera, marker_height=0.1)

    # initial camera pose
    camera_transform = np.array(
        [
#            [0.73256052, -0.28776419, 0.6168848, 0.66972396],
#            [-0.26470017, -0.95534823, -0.13131483, -0.12390466],
#            [0.62712751, -0.06709345, -0.77602162, -0.28781298],
#            [0.0, 0.0, 0.0, 1.0],
            [1, 0, 0, 10],
            [0, 1, 0, 10],
            [0, 0, 1, 10],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )

    print("Create box_outline")
    aabb_min, aabb_max = aabb
    bbox = trimesh.path.creation.box_outline(
        aabb_max - aabb_min,
        tf.translation_matrix((aabb_min + aabb_max) / 2),
    )

    print("Rendering 1")
    geom = trimesh.voxel.ops.multibox(
        occupied, pitch=resolution, colors=[1.0, 0, 0, 0.5]
    )
    scene = trimesh.Scene(camera=camera, geometry=[bbox, geom, camera_marker])
    scene.camera_transform = camera_transform
    hbox.add(labeled_scene_widget(scene, label="occupied"))

    print("Rendering 2")
    geom = trimesh.voxel.ops.multibox(
        empty, pitch=resolution, colors=[0.5, 0.5, 0.5, 0.5]
    )
    scene = trimesh.Scene(camera=camera, geometry=[bbox, geom, camera_marker])
    scene.camera_transform = camera_transform
    hbox.add(labeled_scene_widget(scene, label="empty"))

    print("Show the GUI window")
    gui.add(hbox)
    pyglet.app.run()


def update_freespace_by_subtraction(octree, aabb_min, aabb_max, resolution):
    offset = resolution / 2
    for x in np.arange(aabb_min[0], aabb_max[0], resolution):
        for y in np.arange(aabb_min[1], aabb_max[1], resolution):
            for z in np.arange(aabb_min[2], aabb_max[2], resolution):
                key = octree.coordToKey(np.array([x+offset, y+offset, z+offset]))
                node = octree.search(key)
                try:
                    label = octree.isNodeOccupied(node)
                except Exception as e:
                    octree.updateNode(key, False)

    return octree