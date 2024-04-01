from loader.load_coauthor import LoadCoauthor
from loader.load_baize import LoadBaize
from loader.load_dialcon import LoadDialcon

class DataLoaderFactory:
    loaders = {
        'coauthor': LoadCoauthor,
        'baize': LoadBaize,
        'dialcon': LoadDialcon
    }

    @staticmethod
    def get_loader(dataname, config, eval_dir, meta_path):
        loader_class = DataLoaderFactory.loaders.get(dataname)
        if loader_class:
            return loader_class(config, eval_dir, meta_path)
        else:
            raise ValueError(f"Unknown dataname: {dataname}")
