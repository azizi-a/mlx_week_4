import datasets

def download_data():
  try:
    datasets.load_from_disk("data_cache")
  except:
    ds = datasets.load_dataset("nlphuji/flickr30k", cache_dir="data_cache")
    ds.save_to_disk("data_cache")