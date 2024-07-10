# Napari OCT layer segmentation application

Author: Jason Bunk

## Installation

You can use one of the scripts in the "install_helpers" folder, depending on your operating system.

#### For Mac OS:

 * Install Anaconda
 * Create an environment
 * Use that environment to open a terminal
 * Change directory to this workspace
 * Run the script: ```sh install_helpers/mac_install_from_conda_terminal.sh.```

#### For Windows, there are two options:

The first option doesn't require Anaconda or any command line terminal... just install Napari version 0.4.18 from github, and drag and drop the napari shortcut from your desktop onto the script "Windows - Drag and drop napari shortcut onto this to create customized version". That will create a new script file which can be double clicked to launch the app.

The second option is similar to the Mac installation above:

 * Install Anaconda
 * Create an environment
 * Use that environment to open a terminal
 * Change directory to this workspace
 * Run the script: ```windows_install_from_conda_terminal.bat```


## Running the app

If you used Anaconda to install the app, you can use your Anaconda environment to open a terminal and change directory to this workspace.

The main application file is "main_app_classic.py" which can be run with python.
To use the version with the neural network layer detection algorithm, you can run "main_app_dnn.py".

The measurement metrics will be blank (zero) until you click 3 fovea points. Those 3 points will be used to measure the fovea angle. The parafovea will be shown as two vertical yellow lines.

Some configuration parameters are defined in "definitions_and_config.py".
In particular, the parafovea width, and the expected width and height of the images in millimeters...
you should check the numbers to make sure the calculations will be correct!
Not every OCT scan has the same width and height!
There is no way to know that information from the TIFF image file, you'll need to verify with the original Bioptigen data files!

## App details

The metrics are calculated in a background thread, and re-calculated every time you scribble a correction...
Having the calculations done in the background allows for smooth interaction, to avoid freezing the app.

The metrics measurement calculations are done in ```threading_work.py```, you can see how the measurements are made.
Currently the fovea angle is measured assuming the pixels are square, but that's not typically true!

The detection algorithms are specified in ```oct_detection.py``` and are done in the background thread (they are utilized in ```threading_work.py```).
One algorithm for producing smooth line detections is ```seamsloped.py``` which is based on dynamic programming.
