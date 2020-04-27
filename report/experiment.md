# Experiment

## Model and Baselines

## Results

### Pretraining
Using a pretrained model achieves the best results.

### Pixel Shuffle
Pixel shuffle achieves better results in certain situations when measuring with MSE. When looking for visually pleasing images, it achieves similar to the upsampling technique except contains "defects" instead of blur. Overall this gives a less visually pleasing result. However both results appear unusable.

### Combining methods

#### Pixel shuffle and Pretraining
Combining pixel shuffle and pretraining achieves far better results than standard upsampling.
