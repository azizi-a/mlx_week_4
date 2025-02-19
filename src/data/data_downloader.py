import torch
import datasets
import numpy as np


class DataDownloader(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.ds = datasets.load_dataset(
            "nlphuji/flickr30k", cache_dir="data_cache", split="test"
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            items = self.ds[idx]
            # Convert the dictionary of lists into a list of dictionaries
            num_items = len(items["image"])
            return [
                {
                    "image": np.array(items["image"][i]),
                    "caption": items["caption"][i],
                    "split": items["split"][i],
                }
                for i in range(num_items)
            ]

        item = self.ds[idx]
        return {
            "image": np.array(item["image"]),
            "caption": item["caption"],
            "split": item["split"],
        }
