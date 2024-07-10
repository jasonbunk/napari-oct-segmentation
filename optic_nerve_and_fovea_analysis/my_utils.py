import os
import numpy as np
from qtpy.QtWidgets import QFileDialog
import globals


def uint8clip(arr):
    return np.uint8(np.round(np.clip(arr, a_min=0, a_max=255)))


def uint8norm(arr):
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    return uint8clip((np.nan_to_num(arr)-amin)*255.0/(amax-amin))


# assumes user would never open an image that's only 4 pixels wide...
def make_3or4channel_image_grayscale(image):
    if image.shape[-1] > 4:
        return image, (len(image.shape) == 2)
    if len(image.shape) == 3:
        return image[:, :, 1], True
    if len(image.shape) == 4:
        return image[:, :, :, 1], False
    print(f"(1) warning: cant understand image shape {image.shape}")


def image_shape_to_2d_label_shape(image_shape):
    if len(image_shape) == 2:
        return image_shape
    if image_shape[-1] <= 4 and len(image_shape) in (3,4):
        return image_shape[:-1]
    print(f"(2) warning: cant understand image shape {image_shape}")
    return image_shape


# Create callback function which will save the data in "spreadsheet_data"
def build_spreadsheet_saver_function(spreadsheet_data, table_row_names):
    def save_metrics_to_spreadsheet():
        path, _ = QFileDialog.getSaveFileName(None, "Save Spreadsheet", "", "CSV File (*.csv)")
        if path:
            if "." not in os.path.basename(path):
                path = path+".csv"
            # save spreadsheet vertically
            #with open(path,'w') as outfile:
            #    for idx,(value,) in enumerate(spreadsheet_data):
            #        outfile.write(f"{table_row_names[idx]},{value}\n")
            # save spreadsheet horizontally
            line1 = ["filename",]; line2 = [str(globals.current_filename),]
            for idx,(value,) in enumerate(spreadsheet_data):
                line1.append(str(table_row_names[idx]))
                line2.append(str(value))
            with open(path,'w') as outfile:
                outfile.write(",".join(line1)+"\n"+",".join(line2)+"\n")
            print(f"saved metrics to {path}")
        else:
            print("user didnt provide a valid path to save metrics??")

    return save_metrics_to_spreadsheet


def describe(name, arr):
    print(f"{name}: {arr.shape}, {arr.dtype}: min {arr.min()}, max {arr.max()}, mean {arr.mean()}, std {arr.std()}, median {np.median(arr)}")
