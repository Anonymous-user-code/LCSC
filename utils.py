import numpy as np
import random
import torch
import umap
import struct
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
import torch.nn.functional as F
from scipy import optimize

def sample_point_in_simplex(k):
    return np.random.dirichlet(2*np.ones(k))

def get_plot_images(images, labels, class_images_idx=None, dataset = None):
    if dataset != None:
        images = []
        labels = []
        for x,y in dataset:
            images.append(x)
            labels.append(y)
        images = torch.stack(images,dim=0)

    class_images = [[] for i in range(len(set(labels)))]
    for i in range(len(labels)):
        class_images[labels[i]].append(images[i])
    
    if class_images_idx == None:
        class_images_idx = random.sample(range(len(set(labels))),k = 5)
    select_class_images = [class_images[i] for i in class_images_idx]
    plot_images = [random.choice(images) for images in select_class_images]
    if dataset == None:
        return plot_images, class_images_idx
    else:
        return plot_images, images, labels, class_images_idx

def Umap_transform(source_images,source_labels,target_images,target_labels):
    reducer = umap.UMAP(n_neighbors=8, min_dist=0.05, spread=2.5, n_components=2, metric='manhattan', random_state=42, verbose=True)  
    
    idx = np.random.choice(min(source_images.shape[0], target_images.shape[0]), size=1000, replace=False)

    source_images = source_images.reshape(source_images.shape[0],-1).numpy()
    source_labels = np.eye(len(set(source_labels)))[source_labels]
    source_data = np.hstack([source_images,source_labels])
    source_data = source_data[idx]

    target_images = target_images.reshape(target_images.shape[0],-1).numpy()
    target_labels = np.eye(len(set(target_labels)))[target_labels]
    target_data = np.hstack([target_images,target_labels])
    target_data = target_data[idx]

    all_data = np.vstack([source_data, target_data])
    domain_label = np.array([0]*len(source_data) + [1]*len(target_data))

    embedding = reducer.fit_transform(all_data)

    return embedding, domain_label

def get_label_rate(labels):
    class_labels = [[] for i in range(len(set(labels)))]
    for i in range(len(labels)):
        class_labels[labels[i]].append(labels[i])

    class_rate = [len(class_label)/len(labels) for class_label in class_labels]
    return class_rate

def save_IDX_images(images, filename):
    num_images, num_rows, num_cols = images.shape
    magic = 2051

    with open(filename, 'wb') as f:
        header = struct.pack('>IIII', magic, num_images, num_rows, num_cols)
        f.write(header)
        
        data = images.astype(np.uint8)
        f.write(data.tobytes())

def save_IDX_labels(labels, filename):
    num_labels = labels.shape[0]
    magic = 2049

    with open(filename, 'wb') as f:
        header = struct.pack('>II', magic, num_labels)
        f.write(header)

        labels = labels.astype(np.uint8)
        f.write(labels.tobytes())

def save_FJS_data(Generate_path, D_t_images, D_t_labels):
    D_t_images = (D_t_images - D_t_images.min()) * (255 / (D_t_images.max() - D_t_images.min()))
    D_t_images = D_t_images.squeeze(dim=1)
    D_t_images = D_t_images.cpu().numpy().astype(np.uint8)
    D_t_labels = np.array(D_t_labels).astype(np.uint8)
    save_IDX_images(D_t_images, Generate_path+'/Mnist-target-images')
    save_IDX_labels(D_t_labels,Generate_path+"/Mnist-target-labels")

def data_from_mnist(images_path, labels_path):
    with open(labels_path, 'rb') as lbpath:
        magic, num = struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        data = np.frombuffer(imgpath.read(), dtype=np.uint8)
        images = data.reshape(num, rows, cols)

    return images,labels

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),    # 28→24
            nn.Tanh(),
            nn.AvgPool2d(2),                   # 24→12
            nn.Conv2d(6, 16, kernel_size=5),   # 12→8
            nn.Tanh(),
            nn.AvgPool2d(2),                   # 8→4
            nn.Conv2d(16, 120, kernel_size=4), # 4→1
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class TempScaling(nn.Module):
    def __init__(self):
        super(TempScaling, self).__init__()

    def find_best_T(self, logits, labels):
        nll_criterion = nn.CrossEntropyLoss(reduction='none')
        def eval(x):
            "x ==> temperature T"
            x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            loss = torch.mean(nll_criterion(scaled_logits, labels))
            return loss
        optimal_parameter = optimize.fmin(eval, 2.0, disp=False)
        self.temperature = optimal_parameter[0]
        return self.temperature.item()

def pseudocal(Target_loader, network, DEVICE, class_num = 10):

    ## pseudo-target synthesis
    def mixup(select_loader):
        start_gather = True
        same_cnt = 0
        diff_cnt = 0
        total = 0
        all_diff_idx = None

        with torch.no_grad():
            for ep in range(1):
                for inputs, _ in select_loader:
                    batch_size = inputs.size(0)
                    sample_num = batch_size
                    inputs_a = inputs.to(DEVICE)
                    clb_lam = 0.65
                    rand_idx = torch.randperm(batch_size)
                    inputs_b = inputs_a[rand_idx]
                    outputs_a = network(inputs_a)
                    if type(outputs_a) is tuple:
                        soft_a = outputs_a[1]
                    else:
                        soft_a = outputs_a

                    soft_b = soft_a[rand_idx]
                    same_cnt += (soft_a.max(dim=-1)[1]==soft_b.max(dim=-1)[1]).nonzero(as_tuple=True)[0].shape[0]
                    diff_cnt += (soft_a.max(dim=-1)[1]!=soft_b.max(dim=-1)[1]).nonzero(as_tuple=True)[0].shape[0]
                    
                    ## consider cross-cluster mixup to cover both correct and wrong predictions
                    diff_idx = (soft_a.max(dim=-1)[1]!=soft_b.max(dim=-1)[1]).nonzero(as_tuple=True)[0] + total

                    hard_a = F.one_hot(soft_a.max(dim=-1)[1], num_classes=class_num).float()
                    hard_b = hard_a[rand_idx]

                    mix_inputs = clb_lam * inputs_a + (1 - clb_lam) * inputs_b
                    mix_soft = clb_lam * soft_a.softmax(dim=-1) + (1 - clb_lam) * soft_b.softmax(dim=-1)
                    mix_hard = clb_lam * hard_a + (1 - clb_lam) * hard_b
                    mix_outputs = network(mix_inputs)
                    if type(mix_outputs) is tuple:
                        mix_out = mix_outputs[1]
                    else:
                        mix_out = mix_outputs

                    if start_gather:
                        all_mix_out = mix_out.detach().cpu()
                        all_mix_soft = mix_soft.detach().cpu()
                        all_mix_hard = mix_hard.detach().cpu()
                        all_diff_idx = diff_idx

                        start_gather = False
                    else:
                        all_mix_out = torch.cat((all_mix_out, mix_out.detach().cpu()), 0)
                        all_mix_soft = torch.cat((all_mix_soft, mix_soft.detach().cpu()), 0)
                        all_mix_hard = torch.cat((all_mix_hard, mix_hard.detach().cpu()), 0)
                        all_diff_idx = torch.cat((all_diff_idx, diff_idx), 0)


        all_diff_idx = all_diff_idx.cpu()
        mix_logits = all_mix_out[all_diff_idx]
        mix_soft_labels = all_mix_soft.max(dim=-1)[1][all_diff_idx]
        mix_hard_labels = all_mix_hard.max(dim=-1)[1][all_diff_idx]

        return mix_logits, mix_soft_labels, mix_hard_labels

    def ts(select_loader):
        mix_logits, mix_soft_labels, mix_hard_labels = mixup(select_loader)
        cal_model = TempScaling()
        soft_temp = cal_model.find_best_T(mix_logits, mix_soft_labels)
        hard_temp = cal_model.find_best_T(mix_logits, mix_hard_labels)
        return soft_temp, hard_temp

    soft_t, hard_t = ts(Target_loader)

    return soft_t, hard_t

def test_classifier(model,test_data_loader,target_data_loader):
    model = model.eval()
    test_correct = 0
    test_total = 0.
    with torch.no_grad():
        for images,labels in test_data_loader:
            images = images.to(next(model.parameters()).device)
            labels = labels.to(next(model.parameters()).device)
            output = model(images)
            preds = output.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    test_acc = test_correct / test_total

    target_correct = 0
    target_total = 0.
    target_probs = []
    target_hits = []
    with torch.no_grad():
        soft_t, hard_t = pseudocal(target_data_loader,model,next(model.parameters()).device,model.num_classes)
        for images,labels in target_data_loader:
            images = images.to(next(model.parameters()).device)
            labels = labels.to(next(model.parameters()).device)
            output = model(images)
            output = output / soft_t
            preds = output.argmax(dim=1)
            target_correct += (preds == labels).sum().item()
            target_total += labels.size(0)
            target_probs.append(torch.softmax(output, dim=1).max(dim=1).values.cpu().detach())
            target_hits.append((preds == labels).cpu().detach())

    target_acc = target_correct / target_total
    target_probs = torch.cat(target_probs,dim=0)
    target_hits = torch.cat(target_hits, dim=0)
    return test_acc, target_acc, target_probs, target_hits

class CosineLRSchedule(torch.nn.Module):
    counter: torch.Tensor

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float):
        super().__init__()
        self.register_buffer('counter', torch.zeros(()))
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.set_lr(min_lr)

    def set_lr(self, lr: float) -> float:
        if self.min_lr <= lr <= self.max_lr:
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        return max(self.min_lr, min(self.max_lr, lr))

    def step(self) -> float:
        with torch.no_grad():
            counter = self.counter.add_(1).item()
        if self.counter <= self.warmup_steps:
            new_lr = self.min_lr + counter / self.warmup_steps * (self.max_lr - self.min_lr)
            return self.set_lr(new_lr)

        t = (counter - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        new_lr = self.min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (self.max_lr - self.min_lr)
        return self.set_lr(new_lr)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(AutoEncoder, self).__init__()
        
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        
        decoder_layers = []
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, 1))
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#-----------------------------------------------------------------Normalizing flow model
class Permutation(torch.nn.Module):

    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError('Overload me')


class PermutationIdentity(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])


class Attention(torch.nn.Module):
    USE_SPDA: bool = True

    def __init__(self, in_channels: int, head_channels: int):
        assert in_channels % head_channels == 0
        super().__init__()
        self.norm = torch.nn.LayerNorm(in_channels)
        self.qkv = torch.nn.Linear(in_channels, in_channels * 3)
        self.proj = torch.nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        self.sqrt_scale = head_channels ** (-0.25)
        self.sample = False
        self.k_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}
        self.v_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}

    def forward_spda(
        self, x: torch.Tensor, mask = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, t, d)

        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)  # note that sequence dimension is now 2
            v = torch.cat(self.v_cache[which_cache], dim=2)

        scale = self.sqrt_scale**2 / temp
        if mask is not None:
            mask = mask.bool()
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward_base(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).chunk(3, dim=2)
        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=1)
            v = torch.cat(self.v_cache[which_cache], dim=1)

        attn = torch.einsum('bmhd,bnhd->bmnh', q * self.sqrt_scale, k * self.sqrt_scale) / temp
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        attn = attn.float().softmax(dim=-2).type(attn.dtype)
        x = torch.einsum('bmnh,bnhd->bmhd', attn, v)
        x = x.reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        if self.USE_SPDA:
            return self.forward_spda(x, mask, temp, which_cache)
        return self.forward_base(x, mask, temp, which_cache)


class MLP(torch.nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)
        self.main = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * expansion),
            torch.nn.GELU(),
            torch.nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))


class AttentionBlock(torch.nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: int = 4):
        super().__init__()
        self.attention = Attention(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, attn_temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        return x


class MetaBlock(torch.nn.Module):
    attn_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_patches: int,
        permutation: Permutation,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        num_classes: int = 0,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(in_channels, channels)
        self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        if num_classes:
            self.class_embed = torch.nn.Parameter(torch.randn(num_classes, 1, channels) * 1e-2)
        else:
            self.class_embed = None
        self.attn_blocks = torch.nn.ModuleList(
            [AttentionBlock(channels, head_dim, expansion) for _ in range(num_layers)]
        )
        self.nvp = nvp
        output_dim = in_channels * 2 if nvp else in_channels
        self.proj_out = torch.nn.Linear(channels, output_dim)
        self.proj_out.weight.data.fill_(0.0)
        self.permutation = permutation
        self.register_buffer('attn_mask', torch.tril(torch.ones(num_patches, num_patches)))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        x_in = x
        x = self.proj_in(x) + pos_embed
        if self.class_embed is not None:
            if y is not None:
                if (y < 0).any():
                    m = (y < 0).float().view(-1, 1, 1)
                    class_embed = (1 - m) * self.class_embed[y] + m * self.class_embed.mean(dim=0)
                else:
                    class_embed = self.class_embed[y]
                x = x + class_embed
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, self.attn_mask)
        x = self.proj_out(x)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        scale = (-xa.float()).exp().type(xa.dtype)
        return self.permutation((x_in - xb) * scale, inverse=True), -xa.mean(dim=[1, 2])

    def reverse_step(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        i: int,
        y: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = 'cond',
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x[:, i : i + 1]  # get i-th patch but keep the sequence dimension
        x = self.proj_in(x_in) + pos_embed[i : i + 1]
        if self.class_embed is not None:
            if y is not None:
                x = x + self.class_embed[y]
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, attn_temp=attn_temp, which_cache=which_cache)  # here we use kv caching, so no attn_mask
        x = self.proj_out(x)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)
        return xa, xb

    def set_sample_mode(self, flag: bool = True):
        for m in self.modules():
            if isinstance(m, Attention):
                m.sample = flag
                m.k_cache = {'cond': [], 'uncond': []}
                m.v_cache = {'cond': [], 'uncond': []}

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False
    ) -> torch.Tensor:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        self.set_sample_mode(True)
        T = x.size(1)
        for i in range(x.size(1) - 1):
            za, zb = self.reverse_step(x, pos_embed, i, y, which_cache='cond')
            if guidance > 0 and guide_what:
                za_u, zb_u = self.reverse_step(x, pos_embed, i, None, attn_temp=attn_temp, which_cache='uncond')
                if annealed_guidance:
                    g = (i + 1) / (T - 1) * guidance
                else:
                    g = guidance
                if 'a' in guide_what:
                    za = za + g * (za - za_u)
                if 'b' in guide_what:
                    zb = zb + g * (zb - zb_u)

            scale = za[:, 0].float().exp().type(za.dtype)  # get rid of the sequence dimension
            x[:, i + 1] = x[:, i + 1] * scale + zb[:, 0]
        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)


class Model(torch.nn.Module):
    VAR_LR: float = 0.1
    var: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int,
        nvp: bool = True,
        num_classes: int = 0,
        expansion = 4,
        use_checkpoint = False
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        permutations = [PermutationIdentity(self.num_patches), PermutationFlip(self.num_patches)]

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    in_channels * patch_size**2,
                    channels,
                    self.num_patches,
                    permutations[i % 2],
                    layers_per_block,
                    expansion = expansion,
                    nvp=nvp,
                    num_classes=num_classes,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)
        # prior for nvp mode should be all ones, but needs to be learnd for the vp mode
        self.register_buffer('var', torch.ones(self.num_patches, in_channels * patch_size**2))
        self.use_checkpoint = use_checkpoint

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert an image (N,C',H,W) to a sequence of patches (N,T,C')"""
        u = torch.nn.functional.unfold(x, self.patch_size, stride=self.patch_size)
        return u.transpose(1, 2)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a sequence of patches (N,T,C) to an image (N,C',H,W)"""
        u = x.transpose(1, 2)
        return torch.nn.functional.fold(u, (self.img_size, self.img_size), self.patch_size, stride=self.patch_size)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None, reverse=False
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        if reverse:
            return self.reverse(x,y)
        else:
            x = self.patchify(x)
            outputs = []
            logdets = torch.zeros((), device=x.device)
            for block in self.blocks:
                if self.use_checkpoint:
                    x, logdet = checkpoint(block(x, y))
                else:
                    x, logdet = block(x, y)
                logdets = logdets + logdet
                outputs.append(x)
            return x, outputs, logdets

    def update_prior(self, z: torch.Tensor):
        z2 = (z**2).mean(dim=0)
        self.var.lerp_(z2.detach(), weight=self.VAR_LR)

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor):
        return 0.5 * z.pow(2).mean() - logdets.mean()

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        return_sequence: bool = False
    ) -> torch.Tensor | list[torch.Tensor]:
        seq = [self.unpatchify(x)]
        x = x * self.var.sqrt()
        for block in reversed(self.blocks):
            x = block.reverse(x, y, guidance, guide_what, attn_temp, annealed_guidance)
            seq.append(self.unpatchify(x))
        x = self.unpatchify(x)

        if not return_sequence:
            return x
        else:
            return seq