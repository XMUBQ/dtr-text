from sklearn.decomposition import PCA
from models.sklearn_plus.sk_plus import sk_plus


class sk_PCA(sk_plus):
    def eval(self, data_loader, **kwargs):
        pass

    def __init__(self, n_components=50, outside_name='', input_size=None, condition_size=None, output_size=None):
        super(sk_PCA, self).__init__(outside_name=outside_name)
        self.pca = PCA(n_components)

    def fit(self, x, y=None, **kwargs):
        assert y is None
        self.pca.fit(x)
        return self

    def transform(self, x, **kwargs):
        return self.pca.transform(x)

    def predict(self, x, **kwargs):
        return self.transform(x)
