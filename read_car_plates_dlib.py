import os
import sys
import glob
from PIL import Image
import dlib

# In this example we are going to train a face detector based on the small
# faces dataset in the examples/faces directory.  This means you need to supply
# the path to this faces folder as a command line argument so we will know
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

options.C = 5

options.num_threads = 4
options.be_verbose = True

training_xml_path = os.path.join(plate_folder, "training.xml")
testing_xml_path = os.path.join(plate_folder, "testing.xml")

print(options)
dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)

print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))
#

print("Testing accuracy: {}".format(
    dlib.test_simple_object_detector(testing_xml_path, "detector.svm")))

detector = dlib.simple_object_detector("detector.svm")

# We can look at the HOG filter we learned.  It should look like a face.  Neat!
win_det = dlib.image_window()
win_det.set_image(detector)

print("Showing detections on the images in the faces folder...")
win = dlib.image_window()
for f in glob.glob(os.path.join(plate_folder, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    p_image=Image.fromarray(img,'RGB')
    #p_image.show()
    dets = detector(img)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()


detector1 = dlib.fhog_object_detector("detector.svm")
# In this example we load detector.svm again since it's the only one we have on
# hand. But in general it would be a different detector.
detector2 = dlib.fhog_object_detector("detector.svm")
# make a list of all the detectors you want to run.  Here we have 2, but you
# could have any number.
detectors = [detector1, detector2]
image = dlib.load_rgb_image(plate_folder + '/car01.jpg')
[boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectors, image, upsample_num_times=1, adjust_threshold=0.0)
for i in range(len(boxes)):
    print("detector {} found box {} with confidence {}.".format(detector_idxs[i], boxes[i], confidences[i]))

# Finally, note that you don't have to use the XML based input to
# train_simple_object_detector().  If you have already loaded your training
# images and bounding boxes for the objects then you can call it as shown
# below.

# You just need to put your images into a list.
images = [dlib.load_rgb_image(plate_folder + '/car08.jpg'),
          dlib.load_rgb_image(plate_folder + '/car07.jpg')]
# Then for each image you make a list of rectangles which give the pixel
# locations of the edges of the boxes.
boxes_img1 = ([dlib.rectangle(left=329, top=78, right=437, bottom=186),
               dlib.rectangle(left=224, top=95, right=314, bottom=185),
               dlib.rectangle(left=125, top=65, right=214, bottom=155)])
boxes_img2 = ([dlib.rectangle(left=154, top=46, right=228, bottom=121),
               dlib.rectangle(left=266, top=280, right=328, bottom=342)])
# And then you aggregate those lists of boxes into one big list and then call
# train_simple_object_detector().
boxes = [boxes_img1, boxes_img2]

detector2 = dlib.train_simple_object_detector(images, boxes, options)
# We could save this detector to disk by uncommenting the following.
#detector2.save('detector2.svm')

# Now let's look at its HOG filter!
win_det.set_image(detector2)
dlib.hit_enter_to_continue()


print("\nTraining accuracy: {}".format(
    dlib.test_simple_object_detector(images, boxes, detector2)))

