from models.g_adjust_model import g_adjust_model

class ModelFactory:
    models = {
        'vanilla': g_adjust_model,
        # TODO: future task implementation
    }

    @staticmethod
    def get_model(config):
        model_class = ModelFactory.models.get(config['model'])
        if model_class:
            return model_class(config)
        else:
            raise ValueError(f"Unknown model name: {config['model']}")
