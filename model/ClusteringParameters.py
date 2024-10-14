class ClusterSettings:
    def __init__(self, minimum_cluster_size, clustering_metric, selection_method):
        self.minimum_cluster_size = minimum_cluster_size
        self.clustering_metric = clustering_metric
        self.selection_method = selection_method

    def display_settings(self):
        return "Minimum Cluster: {}, " \
               "Metric: {}, " \
               "Selection Method: {}".format(self.minimum_cluster_size,
                                             self.clustering_metric,
                                             self.selection_method)