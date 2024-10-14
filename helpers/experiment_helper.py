import itertools
from random import choice
from sentence_transformers import SentenceTransformer

from model.ClusteringParameters import ClusterSettings
from model.UMAPParameters import UMAPSettings
from model.SentenceModel import TextModel
from model.Experiment import AnalysisExperiment

from helpers.file_helper import fetch_ott_negative_reviews


def generate_experiments():
    experiments = []

    negative_reviews = fetch_ott_negative_reviews()
    text_models = create_text_models(negative_reviews)
    umap_params = create_umap_params()
    cluster_params = create_clustering_params()

    for text_model in text_models:
        for umap_param in umap_params:
            for cluster_param in cluster_params:
                experiment = AnalysisExperiment(text_model, umap_param, cluster_param)
                experiments.append(experiment)

    return experiments


def create_text_models(data):
    pre_trained_model_names = ['paraphrase-mpnet-base-v2']
    text_models = []

    for model_name in pre_trained_model_names:
        text_models.append(build_text_model(model_name, data))

    return text_models


def create_umap_params():
    n_neighbors_values = range(10, 15)
    n_components_values = range(10, 15)
    metric_values = ['cosine', 'correlation']

    param_combinations = [n_neighbors_values, n_components_values, metric_values]
    params_list = []

    for param_set in itertools.product(*param_combinations):
        (n_neighbors, n_components, metric) = param_set
        umap_param = UMAPSettings(n_neighbors, n_components, metric)
        params_list.append(umap_param)

    return params_list


def create_clustering_params():
    min_cluster_size_values = range(50, 55)
    metric_values = ['chebyshev', 'euclidean', 'p']
    selection_method_values = ['eom', 'leaf']

    param_combinations = [min_cluster_size_values, selection_method_values, metric_values]
    params_list = []

    for param_set in itertools.product(*param_combinations):
        (min_cluster_size, selection_method, metric) = param_set
        cluster_param = ClusterSettings(min_cluster_size, metric, selection_method)
        params_list.append(cluster_param)

    return params_list


def build_text_model(model_name, data):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(data, show_progress_bar=True)

    text_model = TextModel(model_name, embeddings)
    return text_model


def generate_random_experiment():
    negative_reviews = fetch_ott_negative_reviews()
    random_text_model = generate_random_text_model(negative_reviews)
    random_umap_param = generate_random_umap()
    random_cluster_param = generate_random_cluster_param()

    experiment = AnalysisExperiment(random_text_model, random_umap_param, random_cluster_param)
    return experiment


def generate_random_text_model(data):
    pre_trained_models = [
        'paraphrase-mpnet-base-v2',
        'paraphrase-multilingual-mpnet-base-v2',
        'paraphrase-TinyBERT-L6-v2',
        'paraphrase-distilroberta-base-v2',
        'paraphrase-MiniLM-L12-v2',
        'paraphrase-MiniLM-L6-v2',
        'paraphrase-albert-small-v2',
        'paraphrase-multilingual-MiniLM-L12-v2',
        'paraphrase-MiniLM-L3-v2',
        'nli-mpnet-base-v2',
        'stsb-mpnet-base-v2',
        'distiluse-base-multilingual-cased-v1',
        'stsb-distilroberta-base-v2',
        'nli-roberta-base-v2',
        'stsb-roberta-base-v2',
        'nli-distilroberta-base-v2',
        'distiluse-base-multilingual-cased-v2',
        'average_word_embeddings_komninos',
        'average_word_embeddings_glove.6B.300d'
    ]

    random_model_name = choice(pre_trained_models)
    random_text_model = build_text_model(random_model_name, data)

    return random_text_model


def generate_random_umap():
    return choice(create_umap_params())


def generate_random_cluster_param():
    return choice(create_clustering_params())