import json
from PIL import Image
import torch


class SynthDataset(torch.utils.data.Dataset):
    def __init__(self, dict_path, filter=None, transform=None):
        super().__init__()
        self.data =[]
        self.transform = transform
        with open(dict_path) as f:
            d = json.load(f)
        for folder, entries in d.items():
            for entry in entries:
                if filter and not filter(entry):
                    continue
                self.data.append((folder + entry['fname'], entry['text']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im_path, text = self.data[idx]
        im = Image.open(im_path)
        if self.transform:
            return self.transform(im, text)
        return im, text


@torch.inference_mode()
def estimate_metrics(model, ds_train, ds_val, n_iters=None):
    model.eval()

    device = next(model.parameters()).device

    metrics = {}
    for split, ds in zip(('train', 'val'), (ds_train, ds_val)):
        cum_loss, cum_acc, n = 0, 0, 0
        for n, ((x_im, x_ctx), target) in enumerate(ds):
            x_im, x_ctx = x_im.to(device), x_ctx.to(device)
            target = target.to(device)
            preds, loss = model(x_im, x_ctx, target)
            cum_loss += loss.item()
            cum_acc += torch.mean(preds.argmax(-1) == target, 
                                  dtype=torch.float32).item()

            if n_iters is not None and n + 1 >= n_iters:
                break
        metrics[f'{split}_loss'] = cum_loss / (n+1)
        metrics[f'{split}_acc'] = cum_acc / (n+1) 

    return metrics


class EMA:
    def __init__(self, initial=None, k=0.1):
        self.value = initial
        self.k = k

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = (1 - self.k) * self.value + self.k * x
