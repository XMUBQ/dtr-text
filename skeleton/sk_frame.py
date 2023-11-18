from skeleton.skeleton import skeleton


class sk_frame(skeleton):
    def __init__(self, config):
        super(sk_frame, self).__init__(config)

    def test_counterfactual(self, model, config, data_iter):
        return model.test_counterfactual(config, data_iter)

    def update(self, model, config, data_iter):
        data_iter = model.rebuild_data(data_iter)
        _, _, _, train_learned_z = model.forward(data_iter, 'train')
        accumulate_pred, accumulate_true, prob, g_estimation, test_learned_z = self.eval(model, data_iter, print_result=False)
        return {'train_learned_z': train_learned_z, 'pred': accumulate_pred,
                'true': accumulate_true, 'article': data_iter['test']['article'], 'g_estimation': g_estimation,
                'prob': prob, 'test_learned_z': test_learned_z}

    def eval(self, model, data_loader, print_result=True):
        fold_pred, prob, g_estimation, learned_z = model.forward(data_loader, 'test')
        return fold_pred, data_loader['test']['label'], prob, g_estimation, learned_z
