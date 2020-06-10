import os
import glob
import argparse
import matplotlib
import sys
sys.path.insert(0, "./")

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, save_image, to_multichannel,display_image,evaluate
from matplotlib import pyplot as plt
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 

def get_args(parser=None):
    # Argument Parser
    if parser is None:
        parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
    parser.add_argument('--input', default='../Data/monodepth/input/*.png', type=str, help='Input filename or folder.')
    args = parser.parse_args()
    return args
    

#input images are stored in 'input/*.png'
def prediction():
    args = get_args()
    # Input images
    inputs = load_images(glob.glob(args.input))
    print(inputs.shape)

    # Compute results
    #outputs = predict(model, inputs)
    #print("Prediction done!")

    #save the images
    #display_image(outputs,inputs)

class MonoDepth:
    def __init__(self, base_path="", parser=None):
        args = get_args(parser)
        self.args = args
        self.args.model = base_path + self.args.model

        # Custom object needed for inference and training
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

        print('Loading model...')

        # Load model into GPU / CPU
        self.model = load_model(args.model, custom_objects=custom_objects, compile=False)
        print('\nModel loaded ({0}).'.format(args.model))
        
    def prediction(self):
        # Input images
        inputs = load_images(glob.glob(self.args.input))
        print(inputs.shape)

        # Compute results
        outputs = predict(model, inputs)
        print("Prediction done!")

        #save the images
        display_image(outputs, inputs)
        
    def forward(self, x):
        return predict(self.model, x)
    
        

if __name__ == '__main__':
	# Example usage
    prediction()
    model = MonoDepth()
    
