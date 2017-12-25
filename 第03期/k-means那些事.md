# k-means那些事

​	聚类，是机器学习的任务之一。同分类算法一样，聚类算法也被广泛的应用在各个领域，如根据话题，对文章、网页和搜索结果做聚类；根据社区发现对社交网络中的用户做聚类；根据购买历史记录对消费者做聚类。和分类算法不同的是，聚类算法的样本是没有标签的，也就是说，我们并不知道样本有哪些类别，算法需要根据样本的特征，对样本进行聚类，形成不同的聚类中心点。这篇文章，主要介绍比较著名的聚类算法——K-means算法。

​	首先，我们看一下基于目标来做聚类的算法定义:

> **Input** : A set S of n points, also a distance/dissimilarity measure specifying the distance d(x, y) between pairs (x, y). 
>
> **Goal**: output a partition of the data

​	基于这个定义，选择不同的距离计算公式，有以下三种具体的算法:

- **k-means**: find center partitions $c_1, c_2, …, c_k$ to minimize 
  $$ \sum min_{j \in\{i, …,k\}}d^2(x^i, c_j) $$ 
- **k-median**: find center partitions $c_1, c_2, …, c_k$ to minimize 
  $$ \sum min_{j \in\{i, …,k\}}d(x^i, c_j) $$ 
- **k-center**: find partition to minimize the maximum radius

## Euclidean k-means clustering

采用欧拉距离公式的k-means算法定义如下:

> **Input**: A set of n datapoints $x^1, x^2, …, x^n$ in $R^d$ (target #clusters k)
>
> **Output**: k representatives $c_1, c_2, …, c_k \in R^d$ 
>
> **Objective**: choose $c_1, c_2, …, c_k \in R^d$ to minimize  
> $$ \sum min_{j \in \{1,…,k\}}||x^i - c_j||^2 $$

求解该算法的最优解是一个NP难的问题，所有我们没有办法获得最优解，当然，当k=1或d=1这种特殊情况下，是可以获得最优解，有兴趣的可以自行推导一下， 这里不在赘述，这里我们主要介绍Lloyd's method[1]，该方法的核心算法如下:

> **Input**: A set of n datapoints $x^1, x^2, …, x^n$ in $R^d$
>
> **Initialize** centers $c_1, c_2, …, c_k \in R^d$ and clusters $C_1, C_2, …, C_k$ in any way.
>
> **Repeat** until there is no further change in the cost.
>   1. For each j: $C_j <- \{x \in S\ whose\ closest\ center\ is\ c_j\}$
>   2. For each j: $c_j <- mean\ of\ C_j $

对于该算法，难度不是特别大，最重要的地方在Repeat中的1，2两个步骤，其中，步骤1将固定住聚类中心$c_1, c_2, …, c_k$，更新聚类集$C_1, C_2, …, C_k$。步骤2固定住聚类集$C_1, C_2, …, C_k$，更新聚类中心$c_1, c_2, …, c_k$。

大部分学习k-means算法的人理解了步骤1和步骤2就觉得已经理解了k-means了，其实不然，先不说k-means中比较重要的聚类中心的初始化问题，任何一个机器学习算法，它要是有效的，必须证明其可收敛，也需要给出其时间复杂度和空间复杂度。

## Converges

- 目标函数的值在每一轮的迭代都会降低，这个特性由算法中步骤1和步骤2保证，因为对于每个样本点，我们每次都是选择最接近的聚类中心；而且，在每个聚类簇里，我们选择平均值作为其聚类中心。
- 目标函数有最小值0。

由于目标函数有最小值，而且在每一轮中都是值都是减少的，所有算法必然会收敛。

## Running Time

- O(nkd)  n为样本数 k为聚类中心数 d为维度

## Initialization

介绍完了整个算法过程、收敛性和时间复杂度之后，该算法的两个核心点需要我们思考: 1. 如何选择k的值; 2. 算法刚开始，并没有聚类中心，如何初始化聚类中心。对于问题1，我目前还没有过多的认识。这里主要介绍问题2，如何初始化聚类中心。

### 1. Random Initialization

在种方式是最简单的方式，就是随机选k个点作为聚类中心，虽然简单，但是会存在问题，我们看下面的这个例子:

![](https://github.com/neuclil/happy-algorithms/blob/master/%E7%AC%AC03%E6%9C%9F/images/k-means/random%20init.png?raw=true)

由于，我们采用了随机初始化的方式，对于这个样本，我们随机初始化的三个点如上图的绿、红、黑三个样本点，再后面的迭代中，我们最后的聚类簇如上图的箭头所示，这样的效果好吗？显然是不好的，为什么呢？因为很显然最左边三个、中间三个、最右边三个应该是被归为一个聚类簇的:

![](https://github.com/neuclil/happy-algorithms/blob/master/%E7%AC%AC03%E6%9C%9F/images/k-means/random%20init2.png?raw=true)

我们可以看到，聚类中心初始化得不好，直接影响我们最后聚类的效果，可能上面举的例子样本分布和初始化聚类中心太极端，不能说明问题， 我们现在假设样本的分布是多个高斯分布的情况下，聚类中心初始化不好导致的最后聚类的效果:

![](https://github.com/neuclil/happy-algorithms/blob/master/%E7%AC%AC03%E6%9C%9F/images/k-means/random%20init3.png?raw=true)

![](https://github.com/neuclil/happy-algorithms/blob/master/%E7%AC%AC03%E6%9C%9F/images/k-means/random%20init4.png?raw=true)

我们现在计算一下假设有k个高斯分布，我们随机初始化正确的概率有大(所谓正确是指任何两个随机初始化中心不在同一个高斯分布中):$\frac {k!}{k^k} \approx \frac {1}{e^k}$，当k增大时，这个概率会越来越低。

### 2. Furthest Point Heuristic

这种方法是一个中心点一个中心点依次进行初始化的，首先随机初始化$c_1$，然后选择距离$c_1$最远的点来初始化$c_2$，以此类推。

算法描述如下:

>Choose $c_1$ arbitrarily (or at random).
>
>For j = 2, …, k
>
>​	Pick $c_j$ among datapoints $x^1, x^2, …, x^n$ that is farthest from previously chosen $c_1, c_2, …, c_{j-1}$

这种方法解决了随机初始化高斯分布例子中的问题:

![](https://github.com/neuclil/happy-algorithms/blob/master/%E7%AC%AC03%E6%9C%9F/images/k-means/dist%201.png?raw=true)

![](https://github.com/neuclil/happy-algorithms/blob/master/%E7%AC%AC03%E6%9C%9F/images/k-means/dist%202.png?raw=true)

![](https://github.com/neuclil/happy-algorithms/blob/master/%E7%AC%AC03%E6%9C%9F/images/k-means/dist%203.png?raw=trueg)

但是，这种方法的问题是容易受噪声点干扰，请看下面的例子:

![](https://github.com/neuclil/happy-algorithms/blob/master/%E7%AC%AC03%E6%9C%9F/images/k-means/dist%204.png?raw=true)

所以这种方式进行初始化也是不行的，一旦出现噪声点，就极大的影响了最后聚类的结果。虽然实际上，几乎没有哪一个k-means算法会采用上面两种初始化方式，但是这里这样介绍是顺着我们的思维方式进行的，一般思考的方式都是从简单到复杂，所以下面，我们也顺理成章的引出`k-means++`这个初始化算法， 该算法很好的反映出随机化思想在算法中的重要性。

### 3. k-means++

算法描述如下:

>- Choose $c_1$ at random.
>
>- For j = 2, …, k
>
>  - Pick $c_j$ among $x^1, x^2, …, x^n$ according to the distribution
>
>    $ Pr(c_j = x^i) \propto min_{{j}' < j}\left \| x^i - c_{{j}`} \right \|^2 $

这就是k-means++的初始化过程，这个过程比较不好理解。关于这个过程，作以下几点说明:

- 这个初始化算法引入随机化，下一个被选为中心点的样本不是固定的，而是一个概率值，这个概率值正比于“离最近中心点的距离“。
- ”离最近中心点的距离“如何计算，实际上非常简单，就是当前样本距离各个中心点的距离中，最小的那个距离。
- 既然概率正比于 ”距离“ ，那么离群点的”距离“肯定是最大的，它的概率肯定是最大的，可是为什么算法不一定会选择它呢？举个例子说明，如果我们现在有一个聚类集合$S={x_1,x_2,x_3}$,和离群点$x_o$，假设选中 $x_o$的概率为 $1/3$ , 选中 $x_1, x_2, x_3$的概率分别为 $2/9$，这样看，即使$x_o$的概率很大，但是它只有1个，而 $x_1, x_2, x_3$ 即使每个概率不大，但是我们只要随便选中其中一个都是可以的(这是因为它们都在一个聚类簇中，只要选择聚类簇中任何一个点当聚类中心都可以)，所以可以把他们的概率相加，最后得到的概率就大于选中 $x_o$的概率。

## In Action

当然，在实际项目中，我们可能不会自己实现`k-means`算法， 一般我们都会用现成的比较好的一些机器学习库，我们这里结合`scikit-learn`来看一下，它是如何实现`k-means`算法的。

首先看一下，`sklearn.cluster.k_means`模块下的函数`k_means`方法:

```python
def k_means(X, n_clusters, init='k-means++', precompute_distances='auto',
            n_init=10, max_iter=300, verbose=False,
            tol=1e-4, random_state=None, copy_x=True, n_jobs=1,
            algorithm="auto", return_n_iter=False):
```

首先，我们看到参数有一个`init`，这里是指定k-means初始化方法，这里我们看下注释:

```python
"""
    init : {'k-means++', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': generate k centroids from a Gaussian with mean and
        variance estimated from the data.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.
"""
```

可以看到，`sklearn`实现了2种初始化算法，一个是随机初始化算法，另一个是`k-means++`算法，默认采用的是`k-means++`算法。然后，我们先看一下`sklearn`实现`k-means++`的代码:

```python

def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Init n_clusters seeds according to k-means++

    Parameters
    -----------
    X : array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    n_clusters : integer
        The number of seeds to choose

    x_squared_norms : array, shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : numpy.RandomState
        The generator used to initialize the centers.

    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers
```

该算法的是基于 k-means++:the advantages of careful seeding[2]实现的，有兴趣的可以看一下这篇论文。代码第49行，可以看到，第一个初始中心是随机初始化的。代码62行，通过循环，依次初始化其他的聚类中心。

# Reference

1. Lloyd, Stuart P. Least squares quantization in PCM[J]. IEEE Transactions on Information Theory, 1982, 28(2):129-137.
2. Arthur D, Vassilvitskii S. k-means++:the advantages of careful seeding[C]// Eighteenth Acm-Siam Symposium on Discrete Algorithms. Society for Industrial and Applied Mathematics, 2007:1027-1035.
