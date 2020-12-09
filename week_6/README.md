You have to implement an image classification neural network for the same dataset as before. Feel free to reduce the number of examples for each class in your dataset, if it takes too long to train on your hardware.

For this task, you should make at least two neural networks: the fully connected one that works on the same extracted features as before and another one convolutional with class prediction at the end.

You can do it either in Keras or Pytorch. Better to do both.

As an output, you should provide your code, trained model files (2 pcs. at least), your dataset, and the same precision metrics [calculated on test images] as you did before.

Your code should provide 3 execution modes/functions: train (for training new model), test (for testing the trained and saved model on the test dataset), and infer (for inference on a particular folder with images or a single image).
