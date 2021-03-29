import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from sklearn.cluster import AgglomerativeClustering

def make_meshgrid(x1, x2):
    x1_min, x1_max = x1.min()-np.abs(x1.min()*0.05), x1.max()+np.abs(x1.max()*0.05)
    x2_min, x2_max = x2.min()-np.abs(x2.min()*0.05), x2.max()+np.abs(x2.max()*0.05)
    x1x1, x2x2 = np.meshgrid(np.arange(x1_min, x1_max, (x1_max-x1_min)/300),
                             np.arange(x2_min, x2_max, (x2_max-x2_min)/300))
    return x1x1, x2x2

def plot_contours(ax, clf, x1x1, x2x2, **params):
    Z = clf.predict(np.c_[x1x1.ravel(), x2x2.ravel()])
    Z = Z.reshape(x1x1.shape)
    out = ax.contourf(x1x1, x2x2, Z, **params)
    return out

# get unified pseudo label
def count_sort_pseudo_label(pseudo_label):
    label_count=[]
    _y = pseudo_label.copy()
    for unique in np.unique(pseudo_label):
        label_count.append(np.count_nonzero(pseudo_label == unique))

    sort_count = np.argsort(label_count)
    for i in range(len(sort_count)):
        _y[pseudo_label==sort_count[i]] = i
    return _y

def plot_dendrogram(model, cmap=None, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # set_link_color_palette
    set_link_color_palette(cmap)
    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)




def cubic_features(X):
    """
    Returns a new dataset with features given by the mapping
    which corresponds to the cubic kernel.
    """
    n, d = X.shape  # dataset size, input dimension
    X_withones = np.ones((n, d + 1))
    X_withones[:, :-1] = X
    new_d = 0  # dimension of output
    new_d = int((d + 1) * (d + 2) * (d + 3) / 6)

    new_data = np.zeros((n, new_d))
    col_index = 0
    for x_i in range(n):
        X_i = X[x_i]
        X_i = X_i.reshape(1, X_i.size)

        if d > 2:
            comb_2 = np.matmul(np.transpose(X_i), X_i)

            unique_2 = comb_2[np.triu_indices(d, 1)]
            unique_2 = unique_2.reshape(unique_2.size, 1)
            comb_3 = np.matmul(unique_2, X_i)
            keep_m = np.zeros(comb_3.shape)
            index = 0
            for i in range(d - 1):
                keep_m[index + np.arange(d - 1 - i), i] = 0

                tri_keep = np.triu_indices(d - 1 - i, 1)

                correct_0 = tri_keep[0] + index
                correct_1 = tri_keep[1] + i + 1

                keep_m[correct_0, correct_1] = 1
                index += d - 1 - i

            unique_3 = np.sqrt(6) * comb_3[np.nonzero(keep_m)]

            new_data[x_i, np.arange(unique_3.size)] = unique_3
            col_index = unique_3.size

    for i in range(n):
        newdata_colindex = col_index
        for j in range(d + 1):
            new_data[i, newdata_colindex] = X_withones[i, j]**3
            newdata_colindex += 1
            for k in range(j + 1, d + 1):
                new_data[i, newdata_colindex] = X_withones[i, j]**2 * X_withones[i, k] * (3**(0.5))
                newdata_colindex += 1

                new_data[i, newdata_colindex] = X_withones[i, j] * X_withones[i, k]**2 * (3**(0.5))
                newdata_colindex += 1

                if k < d:
                    new_data[i, newdata_colindex] = X_withones[i, j] * X_withones[i, k] * (6**(0.5))
                    newdata_colindex += 1

    return new_data