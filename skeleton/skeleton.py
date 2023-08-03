class skeleton:
    def __init__(self, config):
        pass

    def update(self, model, config, data_iter):
        raise NotImplementedError

    def eval(self, model, data_loader, print_result=True):
        raise NotImplementedError

    def test_counterfactual(self, model, config, data_iter):
        raise NotImplementedError

    def save_model(self, model, save_dir):
        model.save_model(save_dir)