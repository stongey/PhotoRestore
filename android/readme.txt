This is an android version of pyrestore3.py.  Unfortunately the Python Imaging 
Library is not available for android and so all the image processing has to 
be done in python and is very slow.   Nevertheless the app runs on a Samsung 
Galaxy S2 tablet and would be usable by people who want to try it or restore a 
few important pictures.

The user interface is experimental; there are three buttons; the one 
marked "list" opens a file browser and one can select a single image file to
restore; the button marked "open" displays the image.  The default directory 
is "/sdcard/Pictures" which exists on the two tablets I have checked..
The "restore" button starts the processing.  There is a progress bar to give 
encouragement that something is happening and the result is displayed at the 
end.

The "swap" button allows one to compare the original and restored versions.

The restored picture is put in subdirectory "restored" off the original one.
If further picture are restored the "swap" button cycles through all the 
pictures.

