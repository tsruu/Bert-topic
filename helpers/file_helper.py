import pandas as pd

def load_ott_deceptive_reviews():
    """
    Attempts to read the OTT deceptive opinion dataset from different relative paths.
    """
    try:
        ott_path = "../local_datasets/deceptive-opinion.csv"
        ott_deceptive = pd.read_csv(ott_path, encoding='utf-8', sep=",", engine="python")
    except:
        ott_path = "./local_datasets/deceptive-opinion.csv"
        ott_deceptive = pd.read_csv(ott_path, encoding='utf-8', sep=",", engine="python")
    return ott_deceptive


def fetch_ott_negative_reviews():
    ott_reviews = load_ott_deceptive_reviews()
    negative_reviews = ott_reviews[ott_reviews['polarity'] == 'negative']
    return negative_reviews['text']