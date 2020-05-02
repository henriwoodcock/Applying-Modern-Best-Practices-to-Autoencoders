# Experiment

## Model and Baselines

## Measures
MSE, MAE and accuracy on a classifier trained on the original images.

## Results

### MNIST
| Method        | MAE           | MSE   | Classifier Accuracy | Training Time |
| :-----------: |:-------------:| :----:|:-------------------:| :--------:|
| Baseline      | 000000 | 0000 | 00000 | 0m 0s |
| Pixel Shuffle | 000000 | 0000 | 00000 | 0m 0s |
| Progressive Resizing | 000000 | 0000 | 00000 | 0m 0s |
| Pretraining | 000000 | 0000 | 00000 | 0m 0s |
| Feature Loss | 000000 | 0000 | 00000 | 0m 0s |

### Cifar10
| Method        | MAE           | MSE   | Classifier Accuracy | Training Time |
| :-----------: |:-------------:| :----:|:-------------------:| :--------:|
| Baseline      | 000000 | 0000 | 00000 | 0m 0s |
| Pixel Shuffle | 000000 | 0000 | 00000 | 0m 0s |
| Progressive Resizing | 000000 | 0000 | 00000 | 0m 0s |
| Pretraining | 000000 | 0000 | 00000 | 0m 0s |
| Feature Loss | 000000 | 0000 | 00000 | 0m 0s |


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
