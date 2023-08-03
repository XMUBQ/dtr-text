from skeleton.skeleton import skeleton
from sklearn.metrics import classification_report


class sk_frame(skeleton):
    def __init__(self, config):
        super(sk_frame, self).__init__(config)

    def test_counterfactual(self, model, config, data_iter):
        return model.test_counterfactual(config, data_iter)

    def update(self, model, config, data_iter):
        data_iter = model.rebuild_data(data_iter)
        _, _, _ = model.forward(data_iter, 'train')
        test_result, accumulate_pred, accumulate_true, prob, g_estimation = self.eval(model, data_iter,
                                                                                           print_result=False)
        return {'test_result': test_result, 'loss_list': None, 'pred': accumulate_pred,
                'true': accumulate_true, 'article': data_iter['test']['article'], 'g_estimation': g_estimation,
                'prob': prob}

    def eval(self, model, data_loader, print_result=True):
        fold_pred, prob, g_estimation = model.forward(data_loader, 'test')
        return classification_report(data_loader['test']['label'], fold_pred, output_dict=True), fold_pred, \
               data_loader['test']['label'], prob, g_estimation
