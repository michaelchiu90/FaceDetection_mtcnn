# face detection with mtcnn on a photograph
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

# Currently the ‘memory growth’ option should be the same for all GPUs.
# You should set the ‘memory growth’ option before initializing GPUs.
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# draw an image with detected objects
def draw_image_with_boxed(filename,result_list):
    # load image from file
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for result in result_list:
        # get corrdinated
        x,y,width,height = result['box']
        # create the shape
        rect = Rectangle((x,y), width, height,fill = False, color='red')
        # draw the box
        ax.add_patch(rect)

        for key , value in result['keypoints'].items():
            dot = Circle(value, radius=2,color='red')
            ax.add_patch(dot)

    pyplot.show()

def draw_faces(filename, result_list):
    #load the image
    data = pyplot.imread(filename)
    # plot each face as a subplot
    for i in range(len(result_list)):
        #get coordinates (x1 , y1 :Bottom-left-hand-corner)
        x1, y1 , width , height = result_list[i]['box']
        x2 , y2 = x1 + width , y1 + height
        # define subplot
        pyplot.subplot(1 , len(result_list) , i+1)
        pyplot.axis('off')
        # plot face
        pyplot.imshow(data[y1:y2, x1:x2])

    pyplot.show()

filename = "test2.jpg"
pixels = pyplot.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)

#draw_image_with_boxed(filename,faces)
draw_faces(filename, faces)