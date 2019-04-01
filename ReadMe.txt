To run the codes in these files, u need download MNIST data first (u'd better name the file as MNIST_data.file and by this u can run the code straightly)
PaperModel_v1.0 compares ‘0’ and ‘1’ classes
PaperModel_v1.1 compares ‘0’ and ‘5’ classes, which are more likely mixed up than ‘0’ and ‘1’. 
PaperModel_v2.0 tries to generate several weak learning machines, and  combine these weak ones as a strong learning machines. But the result shows this binary data model is not sensitive to kind-of random forest method. The accuracy seldom increase, no matter how to tune the parameters
PaperModel_v3.0 tries to use several kernels to deepen G matrix to transform original data to higher dimension, which hopefully can be linearly classified more easily. But the fact is that the accuracy is even worse. It’s reasonable because the kernel in this model has independent identically distributed standard Gaussian entries. With more random generated kernel, the original information in data will be diluted.
PaperModel_v4.0 implement a function of setting different layers. It’s obviously from results that with more layers, higher the accuracy is. But there still exist a deadly problem: with more layers, the number of layer pattern perform a feature of exponential explosion. As the number of layer up to 6, CPU cannot be qualified for this model.
PaperModel_v5.0 compares this model with other traditional successful model, such as SVM, Random Forest and CNN. These models all perform better than binary data model in the aspect of accuracy.

And here are some result of this model:
Original model
m=100, p=50 [15, 12, 12, 13, 15, 16, 14, 7, 11, 17] 13.2
m=100, p=40  [11, 16, 8, 20, 18, 10, 15, 24, 8, 20] 15

weak learning machines:
m=100, p=50, tt=5, sub_p=40 [16, 11, 12, 13, 13, 9, 16, 11, 18, 21] 14.0
m=100, p=50, tt=35, sub_p=40 [10, 16, 13, 19, 13, 16, 19, 13, 10, 17] 14.6
m=100, p=50, tt=65, sub_p=40 [11, 7, 22, 15, 15, 11, 11, 20, 16, 14] 14.2
m=100, p=50, 55=95, sub_p=40 [12, 8, 18, 17, 14, 11, 14, 15, 17, 9] 13.5

K kernels:
m=100, p=50, k=1 [13, 16, 11, 9, 18, 13, 11, 17, 11, 9, 16, 15, 12, 11, 17, 18, 9, 8, 12, 12], average = 12.9
m=100, p=50, k=2 [18, 18, 14, 15, 14, 27, 22, 18, 19, 26, 29, 13, 18, 22, 26, 27, 21, 20, 19, 16], average = 20.1
m=100, p=50, k=3 [16, 28, 25, 23, 25, 30, 33, 20, 29, 30, 20, 32, 21, 18, 23, 23, 20, 31, 22, 27], average = 24.8
m=100, p=50, k=4 [28, 44, 21, 27, 31, 26, 34, 26, 34, 34, 33, 40, 31, 27, 25, 34, 32, 38, 36, 30], average = 31.55
m=100, p=50, k=5 [35, 40, 46, 36, 38, 39, 39, 30, 26, 35, 36, 38, 44, 35, 44, 33, 41, 37, 37, 37], average = 37.3
m=100, p=50, k=6 [32, 45, 40, 48, 47, 51, 51, 40, 37, 45, 47, 38, 41, 51, 44, 43, 43, 46, 43, 42], average = 43.7
m=100, p=50, k=7 [38, 38, 41, 51, 35, 51, 35, 41, 49, 42, 43, 50, 55, 39, 49, 53, 42, 40, 37, 43], average = 43.6
m=100, p=50, k=8 [45, 45, 43, 49, 47, 46, 51, 34, 42, 46, 49, 45, 44, 43, 42, 44, 42, 53, 53, 38], average = 45.05
m=100, p=50, k=9 [44, 39, 43, 52, 44, 49, 53, 43, 54, 58, 47, 47, 59, 49, 49, 53, 47, 45, 48, 50], average = 48.65
m=100, p=50, k=10 [51, 44, 43, 52, 56, 45, 55, 43, 48, 54, 46, 45, 52, 51, 37, 41, 45, 44, 61, 56], average = 48.45
m=100, p=50, k=11 [48, 50, 59, 52, 47, 51, 56, 50, 48, 49, 55, 52, 47, 51, 50, 51, 46, 49, 37, 54], average = 50.1
m=100, p=50, k=12 [49, 52, 54, 44, 44, 50, 41, 43, 51, 37, 50, 43, 51, 42, 56, 47, 49, 44, 48, 54], average = 47.45
m=100, p=50, k=13[57, 49, 49, 53, 54, 51, 54, 48, 53, 51, 55, 56, 46, 46, 48, 48, 48, 49, 49, 51], average = 50.75

different layers:
m=100, p=50, l=1  average = 13.3
m=100, p=50, l=2 average = 13.61
m=100, p=50, l=3 average = 12.45
m=100, p=50, l=4 average = 12.05
m=100, p=50, l=5, average= 11.0

SVM 0.94
RF 0.97
CNN  0.96
