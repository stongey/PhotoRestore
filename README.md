GIMP PLUG-INS FOR RESTORING FADED SLIDES AND PRINTS

NOTE - The original author, Geoff Daniell, has given me (stongey) permission to publish
this repository on GitHub in his name. Below is Geoff's original Read Me text.

Tested on GIMP 2.8 on a Mac. We will be working on testing with GIMP 2.10, other platforms, and Python 3 coming up.
Your experience may vary. Please report issues. 

-----

A retirement project was scanning all my old colour slides and it turned out that many of the colours had deteriorated over the years. After spending a whole wet afternoon trying to improve them with image processing software it seemed that the best result was using the 'AUTO' button. Since I used to work on image processing (but not this sort) I was sure I could do better than an 'AUTO' button. The result is:


| ![Deteriorated Original](http://www.lionhouse.plus.com/original7.jpg) | ![Restored using my algorithm](http://www.lionhouse.plus.com/restored7.jpg) |
| :---         | :---         |
| Deteriorated Original | Restored using my algorithm |


Some years ago I released a GIMP plug-in for automatically restoring scanned 
slides or prints that had deteriorated with age.  This work is described in 
the file "restore1.pdf" and the file Restore1.py is the corresponding plug-in, 
(renamed from the old restore.py).  Several people have used this and commented favourably
on it performance.

I have now devised a new algorithm which, I believe, should perform 
better.  The technical details and some examples are given in the file 
restore2.pdf and there are two versions of the plug-in.  At the present time 
Restore2.py is the one to use.  The reason Restore3.py is provided is because
there has been some discussion on removing the GIMP procedure for 
conversion of an image to color indexed mode and Restore3.py contains its own
code for the operation.  The results are not identical but are very similar and
it is unlikely that one will perform systematically better than the other, so
comparisons are waste of time.

I also regard a systematic comparison of the performance of the old and new 
algorithms as rather pointless because the justification for new is its much 
more solid theoretical basis.  It is shown to work well in several examples and
would be expected to do well on other images.  I have nevertheless kept the 
original version as inevitably there will be the occasional image for which 
this performs better and if the results of the new one are poor it is worth 
trying the old. 

The plug-ins listed above operate on a single picture opened in GIMP. The 
batch_restore versions are similar but operate on all the photographs in a 
directory.  The results are put in sub-directory "restored", which must exist.

To use these plug-ins copy them to your local directory ./gimp-2.x/plug-ins.
You may need to change the permissions to make them executable.  The
plug-ins will be automatically loaded when you start gimp and they appear on 
the menus under "restore" and "filters"; information is also loaded into the
plug-in database.  They also work on a Mac running OSX except that it is
hard to find the correct folder to put them in because it is hidden!  On 
Windows you are on your own!

