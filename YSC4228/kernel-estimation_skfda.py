
import matplotlib.pylab as plt
import numpy as np

import skfda
import skfda.preprocessing.smoothing.kernel_smoothers as ks
import skfda.preprocessing.smoothing.validation as val


dataset = skfda.datasets.fetch_phoneme()
fd = dataset['data'][:300]

fd[0:5].plot()
plt.show()


n_neighbors = np.arange(1, 24)

scale_factor = (
    (fd.domain_range[0][1] - fd.domain_range[0][0])
    / len(fd.grid_points[0])
)

bandwidth = n_neighbors * scale_factor

# K-nearest neighbours kernel smoothing.
knn = val.SmoothingParameterSearch(
    ks.KNeighborsSmoother(),
    n_neighbors,
)
knn.fit(fd)
knn_fd = knn.transform(fd)

# Local linear regression kernel smoothing.
llr = val.SmoothingParameterSearch(
    ks.LocalLinearRegressionSmoother(),
    bandwidth,
)
llr.fit(fd)
llr_fd = llr.transform(fd)

# Nadaraya-Watson kernel smoothing.
nw = val.SmoothingParameterSearch(
    ks.NadarayaWatsonSmoother(),
    bandwidth,
)
nw.fit(fd)
nw_fd = nw.transform(fd)





fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(
    n_neighbors,
    knn.cv_results_['mean_test_score'],
    label='k-nearest neighbors',
)
ax.plot(
    n_neighbors,
    llr.cv_results_['mean_test_score'],
    label='local linear regression',
)
ax.plot(
    n_neighbors,
    nw.cv_results_['mean_test_score'],
    label='Nadaraya-Watson',
)
ax.legend()
fig




fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Smoothing method parameter')
ax.set_ylabel('GCV score')
ax.set_title('Scores through GCV for different smoothing methods')

fd[10].plot(fig=fig)
knn_fd[10].plot(fig=fig)
llr_fd[10].plot(fig=fig)
nw_fd[10].plot(fig=fig)
ax.legend(
    [
        'original data',
        'k-nearest neighbors',
        'local linear regression',
        'Nadaraya-Watson',
    ],
    title='Smoothing method',
)
plt.show()

llr_fd[10]
