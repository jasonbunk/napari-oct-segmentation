from enum import Enum, IntEnum

class RetinaLayers(IntEnum):
    RPE = 1
    ILM = 2
    OPL = 3
    CSJ = 4

class RetinaLayerColors(Enum):
    RPE = (0, 0, 1) # blue 
    ILM = (0, 1, 0) # green
    OPL = (1, 0, 0) # red
    CSJ = (1, 0.894, 0.71) # light orange

class OCTAnnotationMode(Enum):
    Trace_Layers = 1
    Click_Fovea_Points = 2

machine_labels_opacity = 1
user_label_opacity = 0.2

metrics_table_rows = [
    "foveal_thickness",
    "foveal_thickness_inner",
    "foveal_thickness_outer",
    "parafoveal_thickness_inner",
    "parafoveal_thickness_outer",
    "parafoveal_thickness_inner_left",
    "parafoveal_thickness_outer_left",
    "parafoveal_thickness_inner_right",
    "parafoveal_thickness_outer_right",
    "foveal_angle",
    "choroid_foveal_thickness",
    "choroid_parafoveal_thickness",
    "choroid_parafoveal_thickness_left",
    "choroid_parafoveal_thickness_right",
    "f_p_ratio_inner",
    "f_p_ratio_outer",
    "f_p_ratio_inner_left",
    "f_p_ratio_outer_left",
    "f_p_ratio_inner_right",
    "f_p_ratio_outer_right",
    ]

# small radius around fovea (and around left parafovea, and around right parafovea) where metrics are measured
radius1_cal = 5

# values from ROMEP data OCT scans dated 2015.08.25
typical_image_width_pixels = 1000
typical_image_width_mm = 12
typical_image_height_pixels = 1024
typical_image_height_mm = 2.4566001761193066

# distance in millimeters from fovea center to each side where measurements are made
parafovea_radius_mm = 2.5