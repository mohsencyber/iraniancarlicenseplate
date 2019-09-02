import os
import sys
import glob
from PIL import Image
import dlib

# In this example we are going to train a plate  detector based on the small
# plate dataset in the car_plates/ directory.  This means you need to supply
# the path to this plates folder as a command line argument so we will know
# where it is.
if len(sys.argv) != 2:
    print(
        "Give the path to the car plates directory as the argument to this "
        "program."
        "execute this program by running:\n"
        "    ./read_car_plates_dlib.py car_plate/")
    exit()
plate_folder = sys.argv[1]


options = dlib.simple_object_detector_training_options()

options.add_left_right_image_flips = True

options.C = 50
options.epsilon=0.00001
options.num_threads = 4
options.be_verbose = True

training_xml_path = os.path.join(plate_folder, "training.xml")
testing_xml_path = os.path.join(plate_folder, "testing.xml")

print(options)
#dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)

print("")  # Print blank line to create gap from previous output
#print("Training accuracy: {}".format(
#    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))
#

#print("Testing accuracy: {}".format(
#    dlib.test_simple_object_detector(testing_xml_path, "detector.svm")))

detector = dlib.simple_object_detector("detector.svm")

# We can look at the HOG filter we learned.  It should look like a plate.  Neat!
#win_det = dlib.image_window()
#win_det.set_image(detector)

print("Showing detections on the images in the plates folder...")
#win = dlib.image_window()
for f in glob.glob(os.path.join(plate_folder, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    print (type(img))
    p_image=Image.fromarray(img,'RGB')
    print(type(p_image))
    p_image=p_image.convert('L',dither=Image.NONE)
    p_image=p_image.filter(ImageFilter.GaussianBlur())
    p_image=p_image.filter(ImageFilter.SHARPEN)    
    p_image.show()
    dets = detector(p_image)
    print("Number of plate detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom())
        img = p_image.crop(d.top(),d.left(),d.bottom(),d.right())
        img.show()

    #win.clear_overlay()
    #win.set_image(img)
    #win.add_overlay(dets)
    dlib.hit_enter_to_continue()


