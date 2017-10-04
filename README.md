# MachineLearningFromScratch
**G**radient **B**oosting **D**ecision **T**ree, **S**upport **V**ector **M**achine and **N**eural **N**etwork are arguably the three best machine learning algorithms that has gone through the test of time.

This project implements the three algorithms with simple and neat python code. Those toy codes may not compare to other mature packages such as `xgboost` and `sklearn` in terms of speed and memory consumption, but could help illustrate how those algorithms work.

## Dependence

All three algorithms are implemented in `Python 3.6`. All three algorithms are build from scratch, which means that the implementation is purely based on `numpy`, and there is no dependence on any other machine learning package.

- [NumPy](https://github.com/numpy/numpy)

## Construction in Progress

The implementation of GBDT has been finished, while SVM and NN are still construction in progress.

Tutorial of GBDT is provided below.

## GBDT

This implementation of GBDT supports most of the core features of `xgboost`. Briefly, it supports:

- **Built-in loss**: Mean squared loss for regression task and log loss for classfication task.
- **Customized loss**: Other loss are also supported. User should provide the link function, the gradient, and the hessian.
- **Hessian information**: It uses Newton Method for boosting, thus makes full use of the second-order derivative information. 
- **Regularization**: lambda and gamma, as in `xgboost`.
- **Multi-processing**: It uses the python `Pool` module for multi-processing.

To keep the code neat, some features of `xgboost` are not implemented. For example, it does not handle missing value, and randomness is not supported.

A quick start is provided below.

**Import the module**

```python
from gbdt import GBDT
```

**Initialize model**
```python
model = GBDT(n_threads=None,loss='mse',max_depth=3,min_sample_split=10,reg_lambda=1,gamma=0,
learning_rate=0.1,n_estimators=100)
```
* `n_threads`: Number of threads for multiprocessing. `None` to use all.
* `loss`: Loss function for gradient boosting. `'mse'`  is mean squared error for regression task and `'log'` is log loss for classification task. Pass a child class of the `loss` class to use customized loss. See [source code](https://github.com/drop-out/MachineLearningFromScratch/blob/master/gbdt.py#L7) for details.
* `max_depth`: The maximum depth of a tree.
* `min_sample_split`: The minimum number of samples required to further split a node.
* `reg_lambda`: The regularization coefficient for leaf score, also known as lambda.
* `gamma`: The regularization coefficient for number of tree nodes, also know as gamma.
* `learning_rate`: The learning rate of gradient boosting.
* `n_estimators`: Number of trees.

**Train**
```python
model.fit(train,target)
```
All inputs should be numpy arrays. `train` should be 2D array and `target` should be 1D array.

**Predict**
```python
model.predict(test)
```
Return predictions as numpy array.

**Customized loss**

Define a class that inheritates the `loss` class (see [source code](https://github.com/drop-out/MachineLearningFromScratch/blob/master/gbdt.py#L7)), which specifies the link function, the gradients and the hessian.

```python
class customized_loss(loss):
	def link(self,score):
		return 1/(1+np.exp(-score))
	def g(self,true,score):
		pred=self.link(score)
		return pred-true
	def h(self,true,score):
		pred=self.link(score)
		return pred*(1-pred)
```

And the class could be passed when initializing the model.

```python
model = GBDT(loss=customized_loss,learning_rate=0.1,n_estimators=100)
```



