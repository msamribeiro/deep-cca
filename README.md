# deep-cca

### Overview
**Deep Canonical Correlation Analysis** (DCCA) implementation using Theano

DCCA is a nonlinear extension of Canonical Correaltion Analysis (CCA). Given two views of the same data, DCCA learns transformations that are maximally correlated (Galen et al. 2013). This implementation adopts a stochastic optimization approach (Wang et al. 2015) via SGD. The data used here is the MNIST dataset. Each image is divided into its left and right halves, and we let that be the two views of the same data (check Galen et al, 2013) for details.

Many thanks to [Herman Kamper](https://github.com/kamperh) for various resources, comments, and discussions.

### References:
 - Andrew, Galen, et al. "Deep Canonical Correlation Analysis." ICML (3). 2013.
 - Wang, Weiran, et al. "Unsupervised learning of acoustic features via deep canonical correlation analysis." 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2015.


### Running

```sh
$ python ./deep_cca.py
```
If you wish to run on a GPU, you might want to try something like
```sh
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,force_device=True python ./deep_cca.py 
```