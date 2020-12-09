This time you have to implement a concept of transfer learning on the MNIST dataset.


You should train your CNN classifier on the MNIST dataset with all images rotated by 90 degrees in one direction. As an architecture, you can use any desirable solution. It would be better if you experiment with multiple versions of convolutions and convolutional blocks (like resnet/inception/xception). We also encourage you to use simple data augmentations, batch normalization and dropout.


Once you have trained your best CNN on those rotated data and saved the model, you should switch back to normal (not rotated) data and apply transfer learning. You should load coefficients from the previously trained model and retrain different parts of the CNN. You should try a few options:
1. freeze the entire model except for the last layer and retrain only it.
2. freeze all convolutional layers that extract features and retrain a classifier part (if it is bigger than a single layer).
3. don’t freeze any layers, but start training of the new CNN with weights of the previous model.


You should provide your code and a separate file with the results of your experiments. Here you should specify the accuracy of the next CNNs:
* “rotated” CNN on a rotated test dataset
* “rotated” CNN on a normal test dataset
* retrained CNN a) on a normal test dataset
* retrained CNN b) on a normal test dataset
* retrained CNN c) on a normal test dataset



## RESULTS
* Rotated model accuracy on normal test is :0.4429999887943268 
* Rotated model loss on normal test is 2.470109462738037Rotated model accuracy on rotated test is :0.982699990272522
* Rotated model loss on rotated test is 0.056300971657037735
* retrained CNN a):  accuracy on normal test is :0.77920001745224
* retrained CNN a):  loss on normal test is 0.6721389889717102
* retrained CNN c):  accuracy on normal test is :0.9702000021934509
* retrained CNN c):  loss on normal test is 0.09733608365058899
