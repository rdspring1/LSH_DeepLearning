## Scalable and Sustainable Deep Learning via Randomized Hashing

**Look for major updates around ICLR 2018 deadline this November!**

# Abstract
Current deep learning architectures are growing larger in order to learn from complex datasets. These architectures require giant matrix multiplication operations to train millions of parameters. Conversely, there is another growing trend to bring deep learning to low-power, embedded devices. The matrix operations, associated with the training and testing of deep networks, are very expensive from a computational and energy standpoint. We present a novel hashing-based technique to drastically reduce the amount of computation needed to train and test neural networks. Our approach combines two recent ideas, Adaptive Dropout and Randomized Hashing for Maximum Inner Product Search (MIPS), to select the nodes with the highest activation efficiently. Our new algorithm for deep learning reduces the overall computational cost of the forward and backward propagation steps by operating on significantly fewer nodes. As a consequence, our algorithm uses only 5% of the total multiplications, while keeping within 1% of the accuracy of the original model on average. A unique property of the proposed hashing-based back-propagation is that the updates are always sparse. Due to the sparse gradient updates, our algorithm is ideally suited for asynchronous, parallel training, leading to near-linear speedup, as the number of cores increases. We demonstrate the scalability and sustainability (energy efficiency) of our proposed algorithm via rigorous experimental evaluations on several datasets.

# Future Work
1. Implement a general LSH framework for GPUs with API support for TensorFlow, PyTorch, and MXNet
2. Build Scalable One-Shot Learning using Locality-Sensitive Hashing [https://github.com/RUSH-LAB/LSH_Memory]
3. Demonstrate Tera-Scale machine learning on a single machine using our algorithm, tailored for sparse, high-dimensional datasets (See Netflix VectorFlow)

# References
1. [Scalable and Sustainable Deep Learning via Randomized Hashing (KDD 2017 - Oral)](http://dl.acm.org/citation.cfm?id=3098035)
2. [Efficient Class of LSH-Based Samplers](https://arxiv.org/abs/1703.05160)
3. [Learning to Remember Rare Events](https://arxiv.org/abs/1703.03129)
4. [Netflix VectorFlow](https://medium.com/@NetflixTechBlog/introducing-vectorflow-fe10d7f126b8)