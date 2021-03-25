Files

config.py is the configuration file of the project, including batch_size, eopch, image_size, class and other parameters
data.py is the data processing function of the project, which feeds the data needed by the network.
train.py is the training function, including the framework construction and training functions.
test.py is the output function, which generates predictions.txt  output files.
test_acc.py is the  function that outputs the accuracy on different labels and average accuracy.
The weight folder contains the weight file (vgg16_weights.h5) required for pre-training vgg16 


Dependencies

keras
opencv
