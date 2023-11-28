from models.ocrnet import OCRNet
from util import SynthDataset
from trainer import Trainer, gen_train_config

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
TFN = T.functional

import yaml, pprint, string, os


def entry_filter(entry):
    text, w = entry['text'], entry['w']
    if len(text) + 2 > n_dec_ctx:
        return False
    if any(c not in string.printable for c in text):
        return False
    return w < 512


def collate_fn(data):
    n = len(data)
    x_im = torch.vstack([data[i][0] for i in range(n)])[:, None, :, :]
    x_ctx = torch.vstack([data[i][1] for i in range(n)])
    target = torch.vstack([data[i][2] for i in range(n)])
    return (x_im, x_ctx), target


transforms = T.Compose([
    T.RandomAffine(degrees=0, scale=(0.9, 1.1), translate=(0.05, 0.05)),
])


def preprocess(im, text=None):
    im = TFN.to_dtype(TFN.to_image(im)) / 255
    im = 1 - im
    ar = im.size(-1) / im.size(-2)
    im = TFN.resize(im, (32, int(ar * 32)))
    dx = 512 - im.size(-1)
    im = TFN.pad(im, (0, 0, dx, 0), fill=0)

    if text is None:
        return im, None, None
    
    n_pad = n_dec_ctx - len(text) - 1
    toks = encode(bos + text + eos*n_pad)
    x_ctx = toks[:-1]
    target = toks[1:]
    
    return im, x_ctx, target 

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    with open('configs/charlevel-tiny.yaml') as f:
        cfg = yaml.load(f, yaml.Loader)['config']
    pprint.pp(cfg)

    n_vocab = cfg['decoder']['n_vocab']
    n_enc_ctx = cfg['encoder']['n_ctx']
    n_dec_ctx = cfg['decoder']['n_ctx']

    # define character-level tokenizer

    vocab = string.printable
    stoi = { ch:i for i, ch in enumerate(vocab) }
    itos = { i:ch for i, ch in enumerate(vocab) }
    bos, bos_id = '\r', stoi['\r']
    eos, eos_id = '\n', stoi['\n']
    assert len(vocab) == n_vocab

    encode = lambda s: torch.tensor([stoi[ch] for ch in s])
    decode = lambda x: ''.join(itos[i] for i in x.view(-1).tolist())

    def train_transform(im, text):
        x_im, x_ctx, target = preprocess(im, text)
        return transforms(x_im), x_ctx, target

    ds_train = SynthDataset('data/train.json', entry_filter, train_transform)
    print(len(ds_train), 'entries in train')
    ds_val = SynthDataset('data/val.json', entry_filter, preprocess)
    print(len(ds_val), 'entries in val')

    batch_size = 512
    train_cfg = gen_train_config(
            device='cuda', 
            batches_per_epoch=len(ds_train) // batch_size, 
            epochs=5,
            lr=3e-4,
            bs=batch_size)
    train_cfg.min_lr = 5e-5

    dl_train = DataLoader(ds_train, train_cfg.batch_size, shuffle=True, 
                          collate_fn=collate_fn, num_workers=2, drop_last=True)
    dl_val = DataLoader(ds_val, train_cfg.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=2, drop_last=True)

    ckpt_path = 'ckpts/best_model.pt'

    torch.manual_seed(42)
    net = OCRNet(cfg).to(train_cfg.device)
    count_params = lambda m: sum(p.numel() for p in m.parameters())
    print(count_params(net.backbone), 'CNN params')
    print(count_params(net.encoder), 'encoder params')
    print(count_params(net.decoder), 'decoder params')

    if os.path.isfile(ckpt_path):
        print(net.load_state_dict(torch.load(ckpt_path)))

    net = torch.compile(net)

    def on_best_model(_, model, epoch):
        torch.save(model.state_dict(), 
                   ckpt_path.replace('best', 'new_best'))

    trainer = Trainer(train_cfg)
    trainer.on_best_model = on_best_model
    trainer.train(net, dl_train, dl_val)
