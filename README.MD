This was a one day "hackathon" project to track a car across a sequence of images.

Results / debugging output can be seen here: https://youtu.be/92AjZ-sZN7A

Project attempted: to track the car in the image sequence

Open source code used: OpenCV, used for:
* image loading, modifying and saving
* SIFT feature detection

To build and run:
* Put image sequence into data/00000001.jpg
* Build with cmake:
$ mkdir build
$ cd build && cmake .. && make
* To run:
$ ./car

Files:
main.cpp - has most of the code functionality
* main function loads image sequence
* each image is processed by detect_cars

dataTypes.hpp - file for data types
* contains ImageData class, used to store images and associated data

Algorithmic approach:
* Start with previous car position in image
* Look up set of all SIFT features that belong to the car in the previous image
* Detect all SIFT features in new image
* Match features from new image, to each of the "car features" from the previous image
* Find best location for car's bounding box in the new image, maximising the number of common features
* If we can't match anything good (eg occluded scenario), keep box where it is and hope for the best
* Repeat
