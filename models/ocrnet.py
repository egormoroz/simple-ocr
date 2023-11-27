import torch
import torch.nn as nn, torch.nn.functional as F
from einops import rearrange

from .backbone import CNN
from .transformer import Encoder, Decoder


@torch.no_grad()
def init_transformer_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class OCRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = CNN(cfg['backbone'])
        self.encoder = Encoder(cfg['encoder'])
        self.decoder = Decoder(cfg['decoder'])

        self.encoder.apply(init_transformer_weights)
        self.decoder.apply(init_transformer_weights)

    def forward(self, x_im, x_tok, target=None):
        vis_fts = self.backbone(x_im)
        x_enc = self.encoder(vis_fts)
        x = self.decoder(x_tok, x_enc)

        if target is None:
            return x, None

        loss = F.cross_entropy(
            rearrange(x, 'b t c -> (b t) c'),
            rearrange(target, 'b t -> (b t)'),
        )
        return x, loss

    def configure_optimizers(self, device=None, **kwargs):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        weight_decay = kwargs.get('weight_decay', 0.01)
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0}
        ]

        if device == 'cuda':
            kwargs['fused'] = True

        opt = torch.optim.AdamW(optim_groups, **kwargs)
        return opt


if __name__ == '__main__':
    import yaml
    with open('simpleconfig.yaml') as f:
        cfg = yaml.load(f, yaml.Loader)['config']

    net = OCRNet(cfg)
    print(sum(p.numel() for p in net.parameters()), 'params')

    ch_in = cfg['backbone']['ch_in']
    n_vocab = cfg['decoder']['n_vocab']

    x_im = torch.randn(1, ch_in, 32, 64)
    x = torch.randint(n_vocab, size=(1, 5))
    x_ctx = x[:, :-1]
    x_target = x[:, 1:]

    logits, loss = net(x_im, x_ctx, x_target)
    print(logits.shape)
    print(loss)
    

