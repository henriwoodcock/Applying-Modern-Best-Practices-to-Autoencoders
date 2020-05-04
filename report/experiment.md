# Experiment

## Model and Baselines

## Measures
MSE, MAE and accuracy on a classifier trained on the original images.

## Results
All results are based on the testing data supplied by the datasets, and so this data has been unseen by the model in training. Each model was trained for a total of 15 epochs.

Optimal learning rates for each model were found through calculating the loss with different learning rates for a batch from the dataset. This means that different models require different learning rates. Using the same learning rate for all would not be a fair assessment due to some models have more parameters to learn.
### MNIST
| Method        | MAE           | MSE   | Classifier Accuracy | Training Time |
| :-----------: |:-------------:| :----:|:-------------------:| :--------:|
| Original Image | n/a | n/a | 0.9912 | n/a |
| Baseline      | ........ | **0.000854** | **0.9900** | 24m 43s |
| Pixel Shuffle | 0.011906 | 0.001174 | 0.9870 | 27m 40s |
| Progressive Resizing | 0.011891 | 0.001168 | 0.9877 | 24m 44s |
| Pretraining | 0.018001 | 0.003003 | 0.9804 | 25m 12s |
| Resnet w/o pretrain | 0.018034 | 0.003589 | 0.9654 | 25m 30s |
| Feature Loss | 0.075097 | 0.011058 | 0.9886 | 26m 52s |

### Cifar10
| Method        | MAE           | MSE   | Classifier Accuracy | Training Time |
| :-----------: |:-------------:| :----:|:-------------------:| :--------:|
| Original Image | n/a | n/a | 0.8179 | n/a |
| Baseline      | 0.046274 | 0.004345 | 0.4558 | 23m 47s |
| Pixel Shuffle | 0.046642 | 0.004385 | 0.4759 | 27m 6s |
| Progressive Resizing | **0.046132** | 0.004310 | 0.4608 | 23m 5s |
| Pretraining | 0.076558 | 0.011385 | 0.2333 | 24m 12s |
| Resnet w/o pretrain | 0.075097 | 0.011058 | 0.2236 | 26m 6s |
| Feature Loss | 0.046206 | **0.004301** | **0.6209** | 28m 34s |

The best model for the MNIST dataset was the _baseline_ model for all three measurements. For the Cifar-10 dataset, the best model for MSE and Classifier Accuracy was the _Feature Loss_ model, however the _Pixel Shuffle_ model achieved the lowest MAE. Overall making the Feature Loss model the best for Cifar-10.

## Discussion
The first thing to note that due to the _simplicity_ of the MNIST images, it appears that any of the new training techniques becomes overkill for the MNIST images and the baseline (most simple) model works best for this dataset. Due to this it is hard to do a comparison of the techniques apart from saying that for simple images only a simple model is required.

The conclusion from the MNIST dataset is however that images of size 3*32*32=3072 (the input size) can be cut down to an array of size 1000, meaning these images can be stored at 1/3rd of the size and still achieve a similar accuracy on a classifier (0.9912 compared to 0.9900). However an issue with this is that the original images are of size 1*32*32 and are extended to 3*32*32 to make the models comparable between both datasets. This means the images are not compressed as much as it seems if you take the size of the original images (1024 pixels).

When looking at the Cifar-10 results, it can be seen the more complex methods become more beneficial for more complex images. However, the results are still far from perfect. The best measure to see the "usability" of the model is the classifier accuracy as this shows how the images can still be recognised. Looking at this assessment Feature Loss was the best technique which shows when using an autoencoder trying to recreate features is better than trying to recreate each individual pixel.

The downside to using the Feature Loss model is the time it takes to train, taking over 5 minutes longer than the shortest training time for Cifar-10. However, when compared to other methods it shows the extra training is worth it as the next highest classifier accuracy is as low as 0.4759.

The next subsections will show testing set images for each model and discuss what was learned from using each technique. Each image has the input on the left and the model output on the right.

### Feature Loss
![cifarFL](/images/cifar10-featureloss.png "Cifar10 Feature Loss Results")
![mnistFL](/images/MNIST-featureloss.png "MNIST Feature Loss Results")

### Pretraining
![cifarPT](/images/cifar10-pretrained.png "Cifar10 Pretrained Results")
![cifarresnet](/images/cifar10-pretrained.png "Cifar10 Resnet Results")

![mnistPT](/images/MNIST-pretrained.png "MNIST Pretrained Results")
![mnistresnet](/images/MNIST-pretrained.png "MNIST Resnet Results")

Using a pretrained model achieves the best results.

### Pixel Shuffle
![cifarPS](/images/cifar10-pixelshuffle.png "Cifar10 Pixel Shuffle Results")
![mnistPS](/images/MNIST-pixelshuffle.png "MNIST Pixel Shuffle Results")

Pixel shuffle achieves better results in certain situations when measuring with MSE. When looking for visually pleasing images, it achieves similar to the upsampling technique except contains "defects" instead of blur. Overall this gives a less visually pleasing result. However both results appear unusable.

### Progressive Resizing
![cifarPR](/images/cifar10-progresize.png "Cifar10 Progressive Resizing Results")

![mnistPR](/images/MNIST-progresize.png "MNIST Progressive Resizing Results")

# Conclusion
Overall when attempting to use an autoencoder as a compression system, visual pleasing results are difficult to achieve and may not be possible.

One aspect of an autoencoder that was not mentioned here was the information held in the encoder. In many situations the encoded format is the most important part and so it is best to optimise this instead of the overall output from the autoencoder.
