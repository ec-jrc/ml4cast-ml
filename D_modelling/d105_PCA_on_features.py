import re
import numpy as np
from sklearn.decomposition import PCA

def getPCA(self, feature_names, X):
    # Perform PCA data reduction ON SELECTED FEATURES: Only on NDVI, RAD, Temp (precipitation is excluded)

    # first: - no request of PCA on one single month should arrive here (skipped in c100)
    #        - z scaling is done already above
    # second: operate PCA on all var of group except RainSum

    # get list of feature type in feature group and exclude RainSum
    # Note: trend is unaffected because it is not in the feature group
    feature2PCA = [s for s in self.uset['feature_groups'] if s != 'RainSum']  # [f(x) for x in sequence if condition]
    for var2PCA in feature2PCA:
        # print(var2PCA)
        # print('shape in', X.shape)
        # print('varin', feature_names)
        idx2PCAlist = [i for i, item in enumerate(feature_names) if
                       re.search(var2PCA + 'M' + '[0-9]+', item)]
        var2PCAlist = list(np.array(feature_names)[idx2PCAlist])
        v = X[:, idx2PCAlist]
        # perform PCA and keep the required variance fraction
        n_comp = len(var2PCAlist) - 1  # at least one dimension less
        pca = PCA(n_components=n_comp, svd_solver='full')
        Xpca = pca.fit_transform(v)
        # retain components up to PCAprctVar2keep, if this is never reached take all
        if np.cumsum(pca.explained_variance_ratio_)[-1] <= self.uset['PCAprctVar2keep'] / 100:
            indexComp2retain = n_comp - 1
        else:
            indexComp2retain = np.argwhere(np.cumsum(pca.explained_variance_ratio_) > self.uset['PCAprctVar2keep'] / 100)[0][
                0]
            # print(pca.explained_variance_ratio_)
        # print(indexComp2retain)
        Xpca = Xpca[:, 0:indexComp2retain + 1]
        # now replace original columns with PCAs
        new_features_names = [var2PCA + '_PCA' + str(s) for s in range(1, indexComp2retain + 1 + 1)]
        X = np.delete(X, idx2PCAlist, 1)
        feature_names = list(np.delete(np.array(feature_names), idx2PCAlist))
        feature_names = feature_names + new_features_names
        X = np.concatenate((X, Xpca), axis=1)
    return X, feature_names