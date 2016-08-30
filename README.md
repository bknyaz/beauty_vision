# beauty_vision
Recognition of human faces attractiveness (the SCUT-FBP dataset [1])

### Overview

The SCUT-FBP dataset [1] contains 500 samples (images), for each image there is rating in the range (1,5) measuring beauty of an Asian female face.

I train Support vectors regression (SVR) on top of different features (in some cases projected by PCA with 50 components).
Average Pearson correlation (PC) for 5 independent 10-fold cross validation tests is reported as in [1].

In all experiments images are first resized to (224,294), then central crop (224,224) is taken.


### Results

Model						| Code	 						|  Avg PC for 5 tests
-------						|  :--------:						|  :--------:
Combined features + PCA + SVR [1] 		| - 							|  0.6433
ConvNet [1] 					| - 							|  0.8187
16 random filters + PCA50+ rbf SVR 		| [beauty_baseline_random] (beauty_baseline_random.py) 	|  0.642
16 random filters + linear SVR 			| [beauty_baseline_random] (beauty_baseline_random.py) 	|  0.646
24 random filters + linear SVR 			| [beauty_baseline_random] (beauty_baseline_random.py) 	|  0.660
24 Gabor filters + PCA50+ rbf SVR 		| [beauty_baseline_gabors] (beauty_baseline_gabors.py) 	|  0.638
24 colored Gabor filters + PCA50 + rbf SVR 	| [beauty_baseline_gabors] (beauty_baseline_gabors.py) 	|  0.614
Vgg-ImageNet (pool5+fc6) [2] + linear SVR	| [beauty_vgg_imagenet] (beauty_vgg_imagenet.py) 	|  0.804
Vgg-Face (pool5+fc6) [3] + linear SVR 		| [beauty_vgg_face] (beauty_vgg_face.py) 		|  0.856


### Example of prediction

![vgg_face_prediction_example](https://raw.githubusercontent.com/bknyaz/beauty_vision/master/figs/vgg_face_prediction_example.png)


### References

[1] Xie, Duorui, Lingyu Liang, Lianwen Jin, Jie Xu, and Mengru Li. "SCUT-FBP: A Benchmark Dataset for Facial Beauty Perception." In Systems, Man, and Cybernetics (SMC), 2015 IEEE International Conference on, pp. 1821-1826. IEEE, 2015.
[2] https://gist.github.com/ksimonyan/fd8800eeb36e276cd6f9
[3] http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
