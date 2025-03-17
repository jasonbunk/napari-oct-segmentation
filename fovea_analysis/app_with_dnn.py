# Author: Jason Bunk
import napari
from app_classic import (
    build_on_layers_inserted_for_thread_loop,
    on_viewer_cursor_position,
    on_viewer_mouse_over_canvas,
    on_viewer_slice_step_change,
)
from threading_work import thread_work_loop_neural_network

viewer = napari.Viewer()
viewer.window.qt_viewer.dockLayerList.setVisible(False) # de-clutter interface by hiding layer controls tab... makes interface cleaner since so many layers get created
viewer.window.qt_viewer.dockLayerControls.setVisible(False)
viewer.layers.events.inserted.connect(build_on_layers_inserted_for_thread_loop(thread_work_loop_neural_network))
viewer.cursor.events.position.connect(on_viewer_cursor_position)
viewer.events.mouse_over_canvas.connect(on_viewer_mouse_over_canvas)
viewer.dims.events.current_step.connect(on_viewer_slice_step_change)
napari.run()