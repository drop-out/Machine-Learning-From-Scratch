from abc import ABCMeta, abstractmethod
from multiprocessing import Pool
from functools import partial
import numpy as np


class loss(metaclass=ABCMeta):
	'''
	The absctract base class for loss function.
	Three things should be specified for a loss,
	namely link function, gradient and hessian.
	link() is the link function, which takes scores as input, and returns predictions.
	g() is the gradient, which takes true values and scores as input, and returns gradient.
	h() is the heassian, which takes true values and scores as input, and returns hessian.
	All inputs and outputs are numpy arrays.
	'''
	@abstractmethod
	def link(self,score):
		pass

	@abstractmethod
	def g(self,true,score):
		pass

	@abstractmethod
	def h(self,true,score):
		pass

class mse(loss):
	'''Loss class for mse. As for mse, link function is pred=score.'''
	def link(self,score):
		return score

	def g(self,true,score):
		return score-true

	def h(self,true,score):
		return np.ones_like(score)

class log(loss):
	'''Loss class for log loss. As for log loss, link function is logistic transformation.'''
	def link(self,score):
		return 1/(1+np.exp(-score))

	def g(self,true,score):
		pred=self.link(score)
		return pred-true

	def h(self,true,score):
		pred=self.link(score)
		return pred*(1-pred)


class GBDT(object):
	'''
	Parameters:
	----------
	n_threads: The number of threads used for fitting and predicting.
	loss: Loss function for gradient boosting.
		'mse' for regression task and 'log' for classfication task.
		A child class of the loss class could be passed to implement customized loss.
	max_depth: The maximum depth of a tree.
	min_sample_split: The minimum number of samples required to further split a node.
	reg_lamda: The regularization coefficient for leaf score, also known as lambda.
	gamma: The regularization coefficient for number of tree nodes, also know as gamma.
	learning_rate: The learning rate of gradient boosting.
	n_estimators: Number of trees.
	'''
	def __init__(self,
		n_threads=None,
		loss='mse',
		max_depth=3,min_sample_split=10,reg_lambda=1,gamma=0,
		learning_rate=0.1,n_estimators=100):
		self.n_threads=n_threads
		self.loss=loss
		self.max_depth=max_depth
		self.min_sample_split=min_sample_split
		self.reg_lambda=reg_lambda
		self.gamma=gamma
		self.learning_rate=learning_rate
		self.n_estimators=n_estimators

	def fit(self,train,target):
		self.estimators=[]
		if self.loss=='mse':
			self.loss=mse()
		self.score_start=target.mean()
		score=np.ones(len(train))*self.score_start
		for i in range(self.n_estimators):
			estimator=Tree(n_threads=self.n_threads,
				max_depth=self.max_depth,min_sample_split=self.min_sample_split,reg_lambda=self.reg_lambda,gamma=self.gamma)
			estimator.fit(train,g=self.loss.g(target,score),h=self.loss.h(target,score))
			self.estimators.append(estimator)
			score+=self.learning_rate*estimator.predict(train)
		return self

	def predict(self,test):
		score=np.ones(len(test))*self.score_start
		for i in range(self.n_estimators):
			score+=self.learning_rate*self.estimators[i].predict(test)
		return self.loss.link(score)


class TreeNode(object):
	'''
	The data structure that are used for storing trees.
	A tree is presented by a set of nested TreeNodes,
	with one TreeNode pointing two child TreeNodes,
	until a tree leaf is reached.

	Parameters:
	----------
	is_leaf: If is TreeNode is a leaf.
	score: The prediction (score) of a tree leaf.
	split_feature: The split feature of a tree node.
	split_threshold: The split threshold of a tree node.
	left_child: Pointing to a child TreeNode,
		where the value of split feature is less than the split threshold.
	right_child: Pointing to a child TreeNode,
		where the value of split features is greater than or equal to the split threshold.
	'''
	def __init__(self,
		is_leaf=False,score=None,
		split_feature=None,split_threshold=None,left_child=None,right_child=None):
		self.is_leaf=is_leaf
		self.score=score
		self.split_feature=split_feature
		self.split_threshold=split_threshold
		self.left_child=left_child
		self.right_child=right_child

class Tree(object):
	'''
	This is the building block for GBDT,
	which is a single decision tree,
	also known as an estimator.

	Parameters:
	----------
	n_threads: The number of threads used for fitting and predicting.
	max_depth: The maximum depth of the tree.
	min_sample_split: The minimum number of samples required to further split a node.
	reg_lamda: The regularization coefficient for leaf prediction, also known as lambda.
	gamma: The regularization coefficient for number of TreeNode, also know as gamma.
	'''
	def __init__(self,n_threads=None,max_depth=3,min_sample_split=10,reg_lambda=1,gamma=0):
		self.n_threads=n_threads
		self.max_depth=max_depth
		self.min_sample_split=min_sample_split
		self.reg_lambda=reg_lambda
		self.gamma=gamma

	def fit(self,train,g,h):
		'''
		All inputs must be numpy arrays.
		g and h are gradient and hessian respectively.
		'''
		self.estimator=self.construct_tree(train,g,h,self.max_depth)
		return self

	def predict(self,test):
		'''
		test must be numpy array.
		Return predictions (scores) as an array.
		Multiprocessing is supported for prediction.
		'''
		pool=Pool(self.n_threads)
		f=partial(self.predict_single,self.estimator)
		result=np.array(pool.map(f,test))
		pool.close()
		pool.join()
		return result

	def predict_single(self,treenode,test):
		'''
		The predict method for a single sample point.
		test must be numpy array.
		Return prediction (score) as a number.
		'''
		if treenode.is_leaf:
			return treenode.score
		else:
			if test[treenode.split_feature]<treenode.split_threshold:
				return self.predict_single(treenode.left_child,test)
			else:
				return self.predict_single(treenode.right_child,test)

	def construct_tree(self,train,g,h,max_depth):
		'''
		Construct tree recursively.
		First we should check if we should stop further splitting.
		The stopping conditions include:
		1. We have reached the pre-determined max_depth
		2. The number of sample points at this node is less than min_sample_split
		3. The best gain is less than gamma.
		4. Targets take only one value.
		5. Each feature takes only one value.
		By careful design, we could avoid checking condition 4 and 5 explicitly.
		In function find_threshold(), the best_gain is set to 0 initially.
		So if there are no further feature to split,
		or all the targets take the same value,
		the return value of best_gain would be zero.
		Thus condition 3 would be satisfied,
		and no further splitting would be done.
		To conclude, we need only to check condition 1,2 and 3.
		'''

		if max_depth==0 or len(train)<self.min_sample_split:
			return TreeNode(is_leaf=True,score=self.leaf_score(g,h))

		feature,threshold,gain=self.find_best_split(train,g,h)

		if gain<=self.gamma:
			return TreeNode(is_leaf=True,score=self.leaf_score(g,h))

		index=train[:,feature]<threshold
		left_child=self.construct_tree(train[index],g[index],h[index],max_depth-1)
		right_child=self.construct_tree(train[~index],g[~index],h[~index],max_depth-1)
		return TreeNode(split_feature=feature,split_threshold=threshold,left_child=left_child,right_child=right_child)

	def leaf_score(self,g,h):
		'''
		Given the gradient and hessian of a tree leaf,
		return the prediction (score) at this leaf.
		The score is -G/(H+λ).
		'''
		return -np.sum(g)/(np.sum(h)+self.reg_lambda)

	def leaf_loss(self,g,h):
		'''
		Given the gradient and hessian of a tree leaf,
		return the minimized loss at this leaf.
		The minimized loss is -0.5*G^2/(H+λ).
		'''
		return -0.5*np.square(np.sum(g))/(np.sum(h)+self.reg_lambda)

	def find_best_split(self,train,g,h):
		'''
		Return the best feature to split together with the corresponding threshold.
		Each feature is scanned by find_threshold(),
		a [threshold,best_gain] list is returned for each feature.
		Then we select the feature with the largest best_gain,
		and return index of that feature, the threshold, and the gain that is achieved.
		'''
		pool=Pool(self.n_threads)
		f=partial(self.find_threshold,g,h)
		thresholds=np.array(pool.map(f,train.T))
		pool.close()
		pool.join()
		feature=np.argmax(thresholds[:,1],axis=0)
		threshold=thresholds[feature,0]
		gain=thresholds[feature,1]
		return feature,threshold,gain

	def find_threshold(self,g,h,train):
		'''
		Given a particular feature,
		return the best split threshold together with the gain that is achieved.
		'''
		loss=self.leaf_loss(g,h)
		threshold=None
		best_gain=0
		unq=np.unique(train)
		for i in range(1,len(unq)):
			this_threshold=(unq[i-1]+unq[i])/2
			index=train<this_threshold
			left_loss=self.leaf_loss(g[index],h[index])
			right_loss=self.leaf_loss(g[~index],h[~index])
			this_gain=loss-left_loss-right_loss
			if this_gain>best_gain:
				threshold=this_threshold
				best_gain=this_gain
		return [threshold,best_gain]
