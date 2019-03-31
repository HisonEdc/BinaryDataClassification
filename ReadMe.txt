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
m=100, p=50, k=3 [22, 32, 27, 28, 30, 19, 16, 22, 33, 33] 26.2
m=100, p=50, k=5 [37, 43, 39, 46, 36, 42, 30, 44, 39, 40] 39.6

different layers:
m=100, p=50, l=1  average = 13.3
m=100, p=50, l=2 average = 13.61
m=100, p=50, l=3 average = 12.45
m=100, p=50, l=4 average = 12.05
m=100, p=50, l=5, average= 11.0

SVM 0.94
RF 0.97
CNN  0.96
