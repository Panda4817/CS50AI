# CS50AI 2020 - Traffic

Implementation of a convolutional neural network using TensorFlow to recognize traffic signs in pictures.

## Experiments

My model started with 1 convolutional layer, 1 pooling layer and 1 hidden layer with 0.5 dropout.

Below are the experiments I tried before finding a model that gave the highest accuracy and lowest loss.

### Increasing convolutional and max-pooling layers

I tried 2 convolutional layers and 2 max-pooling layers which gave me a test accuracy of 0.96 and loss of 0.18.

I then tried 3 layers of each, which gave me a lower test accuracy and loss than 2 layers. So I decided on 2 convolutional layers and 2 max-pooling layers.

### Changing numbers and sizes of filters for convolutional layers

I tried a 4x4 kernel to learn 32 filters for 1 layer and 3x3 for the other.
This gave me a test accuracy of 0.94 and loss of 0.24. I also tried a 3x3 kernel and a 2x2 kernel. The accuracy was lower.

I tried 100 filters with 3x3 and 32 filters with 3x3. This was slower to run and accuracy was lower. I also tried 20 filters with 3x3 and 32 with 3x3. This gave me a test accuracy of 0.98 and loss of 0.09. So I decided on 20 filters using 3x3 and 32 filters using 3x3.

### Changing pool sizes for pooling layers

I tried 1x1 sizes for both pooling layers. The accuracy  was very low.
Then I tried 2x2 for first pooling layer and 1x1 for second pooling layer. The test accuracy was 0.98 and the loss was 0.11. This was slightly better than having 2x2 for both layers so I decided on 2x2 and 1x1 pooling layers.

### Changing the numbers and sizes of hidden layers

I tried two hidden layers with 128 units each. This gave me a test accuracy of 0.96 and loss of 0.17. The accuracy is slightly lower than 1 hidden layer. 

I tried two hidden layers with lower number of units each. The accuracy was worse. 

I tried one hidden layer with higher number of units (200). This achieved high accuracy by epoch 5. The test accuracy was 0.98 and the loss was 0.09. So I decided on this hidden layer configuration.

### Changing the dropout value

A dropout of 0.1 reaches high accuracy by epoch 2. But the test accuracy was lower and loss was very high.

A dropout of 0.7 gives similar results to dropout of 0.5. The accuracy was 0.98 and the loss was 0.13. So I stuck with 0.5 dropout.

## Usage

Install required packages by running `pip install -r requirements.txt`.
Download and unzip the German Traffic Sign Recognition Benchmark (GTSRB) [data set](https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip). 
Then run `python traffic.py data_directory`.  The `data_directory` will most likely be `gtsrb`.

## Attribution

Data provided by J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011.