import os
import glob
import argparse
import matplotlib

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, save_image, to_multichannel,display_image,evaluate
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='../Data/monodepth/input/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

#input images are stored in 'input/*.png'
def prediction():
    print('\nModel loaded ({0}).'.format(args.model))

    # Input images
    inputs = load_images( glob.glob(args.input) )

    # Compute results
    outputs = predict(model, inputs)
    print("Prediction done!")

    #save the images
    display_image(outputs,inputs)



if __name__ == '__main__':
	# Example usage
    prediction()
