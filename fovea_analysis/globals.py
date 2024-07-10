# global variables that can be shared across files

current_filename = None

# latest_detected_layers is an array of shape (ImageWidth, NumRetinaLayerCategories)
# where NumRetinaLayerCategories = 4 for RPE, ILM, OPL, CSJ
# For each column (from 0 to ImageWidth-1), the value is the height/row of that layer in that column.
latest_detected_layers = None
