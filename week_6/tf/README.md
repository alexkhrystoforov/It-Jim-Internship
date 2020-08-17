1_model.h5 - CNN
2_model.h5 - NN
trained on kaggle notebooks GPU


I chose accuracy metric because we have balanced classes.

CNN on test:
Accuracy :  0.5375000238418579
Loss :  1.838739037513733

CNN on val:
Loss : ~0.9
Accuracy : ~0.7


We have strong overfit due to small train set, but image augmentation,
early stopping, dropout and batch normalization  have helped a lot.
Use KFold for validation.
Our CNN architecture is very similar VGG16.


NN on test:
Accuracy :  0.31562501192092896
Loss :  2.286170721054077


NN on val:
Test loss: 1.961135983467102
Test accuracy: 0.3921568691730499