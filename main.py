from loading import load_pipelines

if __name__ == "__main__":
    config, fold_data = load_pipelines()
    config['skeleton'].train(config, fold_data)