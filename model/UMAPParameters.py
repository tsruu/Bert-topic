class UMAPSettings:
    def __init__(self, neighbor_count, component_count, distance_metric):
        self.neighbor_count = neighbor_count
        self.component_count = component_count
        self.distance_metric = distance_metric

    def display(self):
        return "Neighbors: {}, " \
               "Components: {}, " \
               "Metric: {}".format(self.neighbor_count,
                                   self.component_count,
                                   self.distance_metric)