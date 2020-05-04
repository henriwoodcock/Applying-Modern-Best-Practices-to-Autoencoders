# Experiment

## Model and Baselines

## Measures
MSE, MAE and accuracy on a classifier trained on the original images.

## Results

### MNIST
| Method        | MAE           | MSE   | Classifier Accuracy | Training Time |
| :-----------: |:-------------:| :----:|:-------------------:| :--------:|
| Original Image | n/a | n/a |  | n/a |
| Baseline      | ........ | 0.000854 | 0.9900 | 24m 43s |
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
| Progressive Resizing | 0.046132 | 0.004310 | 0.4608 | 23m 5s |
| Pretraining | 0.076558 | 0.011385 | 0.2333 | 24m 12s |
| Resnet w/o pretrain | 0.075097 | 0.011058 | 0.2236 | 26m 6s |
| Feature Loss | 0.046206 | 0.004301 | 0.6209 | 28m 34s |


### Pretraining
Using a pretrained model achieves the best results.

### Pixel Shuffle
Pixel shuffle achieves better results in certain situations when measuring with MSE. When looking for visually pleasing images, it achieves similar to the upsampling technique except contains "defects" instead of blur. Overall this gives a less visually pleasing result. However both results appear unusable.

### Feature Loss
cell 94
### Progressive Resizing
Progressive resizing appears to lead to better generalisation. After being initially trained on smaller images, when trained on larger images, the train and validation loss appear closer in earlier epochs suggesting that this technique leads to more general solutions. However, this effect becomes unnoticeable after 20 epochs, suggesting that more data is still the more powerful solution.

To further this research the model trained on both 16x16 and 32x32 images should be used to recreate a 16x16  and 32x32 image, then compare this with the model purely trained on 32x32. This could show that the first leads to a general solution to the problem.

# Conclusion
Overall when attempting to use an autoencoder as a compression system, visual pleasing results are difficult to achieve and may not be possible.

One aspect of an autoencoder that was not mentioned here was the information held in the encoder. In many situations the encoded format is the most important part and so it is best to optimise this instead of the overall output from the autoencoder.
