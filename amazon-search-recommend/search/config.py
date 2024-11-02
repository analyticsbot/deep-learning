# URLs for the Digital Music category
meta_url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Digital_Music.jsonl.gz"

filename = 'Digital_Music_Meta.csv'
# Specify sample size and directory for saving
sample_size = 100000
directory = '/Users/rshankar/Downloads/Projects/deep-learning/amazon-search-recommend/data'
bm_pickle_file = 'bm25_index.pkl'

#ES
es_index_name = 'es_index'

# qdrant
collection_name_text = "text_collection"