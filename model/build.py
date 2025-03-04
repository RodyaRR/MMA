from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn 
import torch.nn.functional as F
from collections import OrderedDict

from .pos_embed import get_2d_sincos_pos_embed

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class MMA(nn.Module):
    def __init__(self, args, num_classes=11003, norm_pix_loss=False):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.patch_size= base_cfg['vision_patch_size']
        self.grid_size = (args.img_size[0] // self.patch_size, args.img_size[1] // self.patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
 
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.embed_dim // 64, batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim //64)
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        if 'mae' in args.loss_names:
            self.norm_pix_loss=norm_pix_loss
            self.decoder_embed = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding
            self.cross_attn_Text = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer_mae = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            self.norm_1=LayerNorm(self.embed_dim)
            self.norm_2=LayerNorm(self.embed_dim)
            self.norm_3=LayerNorm(self.embed_dim)
            self.norm_4=LayerNorm(self.embed_dim)
            self.norm_5=LayerNorm(self.embed_dim)
            self.decoder_pred = nn.Linear(self.embed_dim, self.patch_size**2 * 3, bias=True)

            scale = self.cross_modal_transformer_mae.width**-0.5
            proj_std = scale * ((2 * self.cross_modal_transformer_mae.layers)**-0.5)
            attn_std = scale
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size, cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

            nn.init.normal_(self.cross_attn_Text.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn_Text.out_proj.weight, std=proj_std)
            fc_std = (2 * self.cross_modal_transformer_mae.width)**-0.5
            for block in self.cross_modal_transformer_mae.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer([x])
        x = x[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x
    
    def cross_former_mae(self, image_feats, text_feats, ids_restore):
        image_feats=self.norm_1(image_feats)
        text_feats=self.norm_2(text_feats)

        x = self.decoder_embed(image_feats)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        x = x + self.decoder_pos_embed
        x = x.to(torch.float16)
        x = self.cross_attn_Text(
                self.norm_3(x),
                text_feats,
                text_feats,
                need_weights=False)[0]
        x = self.norm_4(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer_mae([x])
        x = x[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.norm_5(x)
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]

        return x

    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return i_tse_f.float()
 
    def encode_text_tse(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, batch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
            
        if 'ndf' in self.current_task:
            loss1 = objectives.ndf_loss(i_feats, t_feats, batch['pids'], logit_scale=self.logit_scale)
            loss2 = objectives.ndf_loss(i_tse_f, t_tse_f, batch['pids'], logit_scale=self.logit_scale)

            ret.update({'Gndf_loss': loss1})
            ret.update({'Lndf_loss': loss2})

        if 'sdm' in self.current_task:
            loss1 = objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale=self.logit_scale)
            loss2 = objectives.compute_sdm(i_tse_f, t_tse_f, batch['pids'], logit_scale=self.logit_scale)

            ret.update({'GSDM_loss': loss1})
            ret.update({'LSDM_loss': loss2})

        if 'ITC' in self.current_task:
            loss1 = objectives.compute_InfoNCE_per(i_feats, t_feats, logit_scale=self.logit_scale)
            loss2 = objectives.compute_InfoNCE_per(i_tse_f, t_tse_f, logit_scale=self.logit_scale)

            ret.update({'GITC_loss': loss1})
            ret.update({'LITC_loss': loss2})

        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']
            mlm_feats, _ = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)
            x = self.mlm_head(x)

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(self, scores, mlm_labels) * self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})
        
        if 'mae' in self.current_task:
            image_feats, atten_i, text_feats, atten_t, mask, ids_restore = self.base_model(images, caption_ids,need_MAE=self.args.need_MAE,mask_ratio=self.args.mask_ratio)
            pred=self.cross_former_mae(image_feats, text_feats, ids_restore)
            loss=self.forward_loss(images,pred,mask)
            ret.update({'mae_loss': self.args.mae_loss_weight * loss})

        return ret


def build_model(args, num_classes=11003):
    model = MMA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
