# Experiment

## Model and Baselines

## Results

### Pretraining
Using a pretrained model achieves the best results.

### Pixel Shuffle
Pixel shuffle achieves better results in certain situations when measuring with MSE. When looking for visually pleasing images, it achieves similar to the upsampling technique except contains "defects" instead of blur. Overall this gives a less visually pleasing result. However both results appear unusable.

### Feature Loss

### Combining methods

#### Pixel shuffle and Pretraining
Combining pixel shuffle and pretraining achieves far better results than standard upsampling.


# Conclusion
Overall when attempting to use an autoencoder as a compression system, visual pleasing results are difficult to achieve and may not be possible.

One aspect of an autoencoder that was not mentioned here was the information held in the encoder. In many situations the encoded format is the most important part and so it is best to optimise this instead of the overall output from the autoencoder.
