You have to implement you own image retrieval solution. There is a dataset with images of a few classes. Your code should take image filename as an input parameter, search for most similar images over the whole dataset and visualize input image + 5 top matches. Feel free to use any classic features/descriptors (histograms, Gabor, HOG etc.) except neural networks stuff. Select matches by the minimal distance.


Only color features doesn't work good, especially on 84x84 images.

Sometimes you need to delete 'ore' and S_Store file in datasets folder...


## achitecture
![architecture](https://github.com/alexkhrystoforov/It-Jim-Internship/blob/master/week_4/architecture.png)
