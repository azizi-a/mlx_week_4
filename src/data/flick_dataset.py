import torch
import datasets
import transformers


class Flick(torch.utils.data.Dataset):
  def __init__(self, split=True):
    super().__init__()
    rw_data = datasets.load_dataset("nlphuji/flickr30k", split="test", cache_dir="./data_cache/flickr")
    self.ds = rw_data.filter(lambda ex: ex["split"] == split)
    self.processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    self.tokenizer = transformers.CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    itm = self.ds[idx]
    img = itm["image"]
    txt = itm["caption"][0]
    enc = self.processor(text=txt, images=img, return_tensors="pt")
    img = enc["pixel_values"].squeeze()
    tkn = enc["input_ids"].squeeze()[:75]
    return tkn, img

  def collate(self, batch):
    txts = [itm[0] for itm in batch]
    imgs = [itm[1] for itm in batch]
    txts = torch.nn.utils.rnn.pad_sequence(txts, batch_first=True, padding_value=self.tokenizer.pad_token_id)
    imgs = torch.stack(imgs, dim=0)
    return txts, imgs


if __name__ == "__main__":
  ds = Flick(split="test")
  tkn, img = ds[0]
  print("len", len(ds))  # trn 29,000 val 1,014 tst 1,000
  print("txt", tkn)
  print("tkn", tkn.shape)  # torch.Size([19])
  print("img", img.shape)  # torch.Size([3, 224, 224])
  print("tkn", ds.tokenizer.decode(tkn))
