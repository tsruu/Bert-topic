import umap
import hdbscan
import pandas as pd

class AnalysisExperiment:
    def __init__(self, text_model, umap_settings, cluster_settings):
        self.text_model = text_model
        self.neighbor_count = umap_settings.neighbor_count
        self.component_count = umap_settings.component_count
        self.umap_distance_metric = umap_settings.distance_metric
        self.minimum_cluster_size = cluster_settings.minimum_cluster_size
        self.clustering_metric = cluster_settings.clustering_metric
        self.selection_method = cluster_settings.selection_method

    def fetch_results(self):
        experiment_result = pd.DataFrame()
        experiment_result["Model Name"] = pd.Series(self.text_model.model_identifier)

        experiment_result["Umap Neighbors"] = pd.Series(self.neighbor_count)
        experiment_result["Umap Components"] = pd.Series(self.component_count)
        experiment_result["Umap Metric"] = pd.Series(self.umap_distance_metric)

        experiment_result["Cluster Size"] = pd.Series(self.minimum_cluster_size)
        experiment_result["Cluster Metric"] = pd.Series(self.clustering_metric)
        experiment_result["Cluster Selection Method"] = pd.Series(self.selection_method)

        try:
            cluster_output = self.execute_clustering()
            num_clusters = len(set(cluster_output.labels_))
        except:
            num_clusters = 0  # Indicates clustering error

        experiment_result["Number of Clusters"] = pd.Series(num_clusters)

        return experiment_result

    def apply_dimensionality_reduction(self):
        reducer = umap.UMAP(n_neighbors=self.neighbor_count, n_components=self.component_count, metric=self.umap_distance_metric)
        reduced_data = reducer.fit_transform(self.text_model.vector_embeddings)
        return reduced_data

    def execute_clustering(self):
        reduced_data = self.apply_dimensionality_reduction()
        cluster_output = hdbscan.HDBSCAN(min_cluster_size=self.minimum_cluster_size,
                                         metric=self.clustering_metric,
                                         cluster_selection_method=self.selection_method).fit(reduced_data)
        return cluster_output

    def display_info(self):
        return "Model: {}, " \
               "Umap Components: {}, " \
               "Umap Neighbors: {}, " \
               "Umap Metric: {}, " \
               "Cluster Size: {}, " \
               "Cluster Metric: {}, " \
               "Selection Method: {}".format(self.text_model.model_identifier,
                                             self.component_count,
                                             self.neighbor_count,
                                             self.umap_distance_metric,
                                             self.minimum_cluster_size,
                                             self.clustering_metric,
                                             self.selection_method
                                             )