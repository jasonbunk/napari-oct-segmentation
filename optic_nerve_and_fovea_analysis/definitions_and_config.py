from enum import Enum, IntEnum

class RetinaLayers(IntEnum):
    ILM = 1
    RNFLGCL = 2
    GCLIPL = 3
    IPLINL = 4
    OPL = 5
    RPE = 6

class TraceableRetinaLayers(IntEnum):
    ILM = 1
    RNFLGCL = 2
    GCLIPL = 3
    IPLINL = 4
    # no OPL
    RPE = 6

class DiskRetinaLayers(IntEnum):
    ILM = 1
    RNFLGCL = 2
    RPE = 6

class RetinaLayerColors(Enum):
    ILM = (0, 1, 0) # green
    RNFLGCL = (0.741, 0.1333, 1) # purple
    GCLIPL = (0.8, 0.8, 0.2) # yellow  (0.3, 0.9, 0.9) # cyan
    IPLINL = (0.3, 0.9, 0.9) # cyan    (0.52, 0.32, 0.16) # brownish orange
    OPL = (1, 0, 0) # red
    RPE = (0.1, 0.1, 1) # blue 
    CSJ = (1, 0.894, 0.71) # light orange

class OCTAnnotationMode(IntEnum):
    Trace_Layers = 1
    Click_Points = 2

class DiscClickMode(IntEnum):
    RPE_Endpoints = 1
    Cup_Bottom = 2

class DiskClickColors(Enum):
    RPE_Endpoints = (1, 1, 1) # white
    Cup_Bottom = (1, 1, 0) # yellow

class IndicatorLineTypes(IntEnum):
    RPE_Endpoints = 0
    Cup_Rim = 1
    Cup_Depth = 2
    Rim_Left = 3
    Rim_Right = 4

class IndicatorLineTypeColors(Enum):
    RPE_Endpoints = (1, 1, 1) # white
    Cup_Rim = (1, 0, 0) # red
    Cup_Depth = (1, 1, 0) # yellow
    Rim_Left = (0, 0, 0.7) # dark blue
    Rim_Right = (0, 0, 0.70001) # dark blue but it cant be exactly the same

machine_labels_opacity = 1
user_label_opacity = 0.6

disc_metrics_table_rows = [
    "disc_diameter",
    "cup_diameter",
    "max_cup_depth",
    "neural_rim_left",
    "neural_rim_right",
    "neural_height_left",
    "neural_height_right",
    "RNFL_avgthk_peri_left",
    "RNFL_avgthk_peri_right",
    ]

layers = ["RNFL", "GCL", "IPL", "RPE-ILM"]

retinalayers_metrics_table_rows = []
for ly in layers:
    retinalayers_metrics_table_rows += [ly+"_left", ly+"_center", ly+"_right"]

# values from ROMEP data OCT scans dated 2015.08.25
typical_image_width_pixels = 1000
typical_image_width_microns = 12000
typical_image_height_pixels = 1024
typical_image_height_microns = 2456.6001761193066

disc_paraside_microns = 1400
disc_paraside_radius1cal_microns = 200 # thickness of measurement region on each side of disc

# distance in microns from fovea center to each side where measurements are made
parafovea_radius_microns = 1250
# radius around fovea (and around left parafovea, and around right parafovea) where metrics are measured
radius_parafovea_calculations = 750
# radius around fovea center where metrics are measured
radius_foveacenter_calculations = 250
