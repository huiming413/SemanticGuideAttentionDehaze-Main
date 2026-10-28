import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torchvision.transforms as transforms
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='nh', help='its or nh')
parser.add_argument('--output_path', type=str, default='./results/vis_results/nh', help='Output result folder')
parser.add_argument('--pretrained_model', type=str, default='./weights/NH_train_21.60_0.7624.pth', help='Pretrained model weights')
parser.add_argument('--dpi', type=int, default=100, help='dpi values')
opt = parser.parse_args()

DPI = opt.dpi
SAVE_PATH = opt.output_path
os.makedirs(SAVE_PATH, exist_ok=True)

# ============================================================
# 第一部分：钩子注册与特征提取
# ============================================================

class SemanticVisualizer:
    """用于提取并可视化语义引导相关特征的钩子管理器"""

    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """注册前向钩子，捕获语义特征和SGA中间特征"""

        # 1. 捕获语义抽取器输出（无语义引导时输出为 None，跳过）
        def hook_semantic(module, input, output):
            if output is not None:
                self.features['semantic'] = output.detach()

        self.hooks.append(
            self.model.semantic_extractor.register_forward_hook(hook_semantic)
        )

        # 2. 捕获每个 SemanticRefineBlock 中 SGA 的中间特征
        for i, block in enumerate([
            self.model.block.block1, self.model.block.block2,
            self.model.block.block3, self.model.block.block4,
            self.model.block.block5, self.model.block.block6,
        ]):
            def make_sga_hook(idx):
                def hook(module, input, output):
                    x = input[0]
                    sem_feat = input[1] if len(input) > 1 else None
                    attn = self._compute_sga_attention(module, x, sem_feat)
                    self.features[f'sga_attn_{idx}'] = attn.detach()
                return hook

            self.hooks.append(
                block.sga.register_forward_hook(make_sga_hook(i))
            )

    @staticmethod
    def _compute_sga_attention(sga_module, x, sem_feat):
        """复现 SGA 的注意力图计算（与 forward 逻辑一致）"""
        c1_ = sga_module.conv1(x)
        c1 = sga_module.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = sga_module.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = sga_module.conv_f(c1_)

        if sem_feat is not None:
            sem = F.interpolate(sem_feat, (x.size(2), x.size(3)),
                                mode='bilinear', align_corners=False)
            sem = sga_module.sem_proj(sem)
            spatial_feat = c3 + cf
            fused_feat = sga_module.fuse(torch.cat([spatial_feat, sem], dim=1))
        else:
            fused_feat = c3 + cf

        c4 = sga_module.conv4(fused_feat)
        m = sga_module.sigmoid(c4)
        return m

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear(self):
        self.features = {}


# ============================================================
# 第二部分：图像预处理与后处理工具
# ============================================================

def load_image(path, size=(256, 256)):
    """加载图像并归一化到 [-1, 1]"""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0)


def tensor_to_numpy(tensor):
    """将 [-1,1] 的 tensor 转为 [0,1] 的 numpy 图像"""
    img = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)
    return img


def normalize_map(feat_map):
    """将特征图归一化到 [0,1] 用于可视化"""
    feat_map = feat_map - feat_map.min()
    if feat_map.max() > 1e-8:
        feat_map = feat_map / feat_map.max()
    return feat_map


def normalize_signed(diff_map):
    """将带符号的差异图归一化到 [-1,1]，保留正负信息"""
    vmax = max(abs(diff_map.min()), abs(diff_map.max()))
    if vmax < 1e-8:
        return diff_map, 1e-8
    return diff_map / vmax, vmax


def feature_to_heatmap(feat, colormap='jet'):
    """将多通道特征图转为热力图（通道取平均后上色）"""
    fmap = feat.squeeze(0).mean(dim=0).cpu().numpy()
    fmap = normalize_map(fmap)
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(fmap)[:, :, :3]
    return heatmap


# ============================================================
# 第三部分：核心可视化函数
# ============================================================

def visualize_sga_attention(model, img_tensor, block_idx=0, use_eca=True):
    """
    可视化第 block_idx 个 SemanticRefineBlock 中 SGA 的注意力图
    对比有语义引导 vs 无语义引导的注意力分布差异
    """
    viz = SemanticVisualizer(model)

    # 有语义引导
    with torch.no_grad():
        _ = model(img_tensor, use_eca=use_eca)
    attn_with = viz.features[f'sga_attn_{block_idx}'].clone()
    viz.clear()

    # 无语义引导（临时替换 semantic_extractor 输出为 None）
    original_forward = model.semantic_extractor.forward
    model.semantic_extractor.forward = lambda x: None
    with torch.no_grad():
        _ = model(img_tensor, use_eca=use_eca)
    attn_without = viz.features[f'sga_attn_{block_idx}'].clone()
    model.semantic_extractor.forward = original_forward

    # 可视化
    img_np = tensor_to_numpy(img_tensor)
    attn_w = attn_with.squeeze(0).mean(dim=0).cpu().numpy()
    attn_wo = attn_without.squeeze(0).mean(dim=0).cpu().numpy()

    attn_w = normalize_map(attn_w)
    attn_wo = normalize_map(attn_wo)

    # 差异图保留符号，用 RdBu_r 呈现
    diff = attn_w - attn_wo
    diff_norm, vmax = normalize_signed(diff)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img_np)
    axes[0].set_title('Input Hazy Image', fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(img_np)
    axes[1].imshow(attn_wo, cmap='jet', alpha=0.5)
    axes[1].set_title('Attention w/o Semantic', fontsize=13)
    axes[1].axis('off')

    axes[2].imshow(img_np)
    axes[2].imshow(attn_w, cmap='jet', alpha=0.5)
    axes[2].set_title('Attention w/ Semantic (Ours)', fontsize=13)
    axes[2].axis('off')

    im = axes[3].imshow(diff_norm, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[3].set_title('Difference (w/ - w/o)', fontsize=13)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    plt.suptitle(f'Semantic-Guided Attention Comparison (Block {block_idx + 1})',
                 fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f'sga_attention_block{block_idx + 1}.png'),
                dpi=DPI, bbox_inches='tight')
    plt.close()

    viz.remove_hooks()

    # 打印定量指标
    print(f"[Block {block_idx + 1}] Attention entropy:")
    print(f"  w/o semantic: {_attention_entropy(attn_without):.4f}")
    print(f"  w/  semantic: {_attention_entropy(attn_with):.4f}")
    print(f"  mean diff (w/ - w/o): {diff.mean():.4f}, "
          f"positive ratio: {(diff > 0).mean():.4f}")

    return attn_with, attn_without

def visualize_semantic_consistency(model, hazy_tensor, clean_tensor,
                                   use_eca=True):
    """
    可视化雾图与清晰图的语义特征差异
    保留符号信息，用 RdBu_r 呈现正负差异
    """
    viz = SemanticVisualizer(model)

    with torch.no_grad():
        _ = model(hazy_tensor, use_eca=use_eca)
    sem_hazy = viz.features['semantic'].clone()
    viz.clear()

    with torch.no_grad():
        _ = model(clean_tensor, use_eca=use_eca)
    sem_clean = viz.features['semantic'].clone()

    # 差异图保留符号：正的表示雾图特征响应更强，负的表示清晰图更强
    diff = (sem_hazy - sem_clean).squeeze(0).mean(dim=0).cpu().numpy()
    diff_norm, vmax = normalize_signed(diff)

    # 计算定量指标
    cos_sim = _semantic_cosine_similarity(sem_hazy, sem_clean)
    rel_diff = _semantic_relative_difference(sem_hazy, sem_clean)
    print(f"Semantic consistency metrics:")
    print(f"  Cosine similarity (mean): {cos_sim:.4f}")
    print(f"  Relative L2 difference:   {rel_diff:.4f}")
    print(f"  Diff range: [{diff.min():.4f}, {diff.max():.4f}], "
          f"positive ratio: {(diff > 0).mean():.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(tensor_to_numpy(hazy_tensor))
    axes[0].set_title('Hazy Input', fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(tensor_to_numpy(clean_tensor))
    axes[1].set_title('Clean Reference', fontsize=13)
    axes[1].axis('off')

    axes[2].imshow(feature_to_heatmap(sem_hazy))
    axes[2].set_title('Semantic Feature (Hazy)', fontsize=13)
    axes[2].axis('off')

    im = axes[3].imshow(diff_norm, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[3].set_title('Feature Difference (Hazy - Clean)', fontsize=13)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    plt.suptitle('Semantic Feature Consistency between Hazy and Clean Images',
                 fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, 'semantic_consistency.png'),
                dpi=DPI, bbox_inches='tight')
    plt.close()

    viz.remove_hooks()
    return sem_hazy, sem_clean


# ============================================================
# 第四部分：定量指标计算
# ============================================================

def _attention_entropy(attn_map):
    """计算注意力图的熵，熵越低说明注意力越集中"""
    attn = attn_map.squeeze(0).mean(dim=0).cpu()  # (H, W)
    attn = attn / (attn.sum() + 1e-8)
    attn = attn.flatten()
    entropy = -(attn * torch.log(attn + 1e-8)).sum()
    return entropy.item()


def _semantic_cosine_similarity(sem_hazy, sem_clean):
    """计算雾图与清晰图语义特征的平均余弦相似度"""
    B, C, H, W = sem_hazy.shape
    sem_h = sem_hazy.view(B, C, -1)
    sem_c = sem_clean.view(B, C, -1)
    cos_sim = F.cosine_similarity(sem_h, sem_c, dim=-1)  # (B, C)
    return cos_sim.mean().item()


def _semantic_relative_difference(sem_hazy, sem_clean):
    """计算语义特征的相对 L2 差异"""
    diff = torch.norm(sem_hazy - sem_clean, p=2) / (torch.norm(sem_clean, p=2) + 1e-8)
    return diff.item()


# ============================================================
# 第五部分：主调用示例
# ============================================================

if __name__ == '__main__':
    from models import Dehaze

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Dehaze().to(device)
    model.load_state_dict(
        torch.load(opt.pretrained_model, map_location=device)
    )
    model.eval()

    hazy = load_image('./test_images/nh/54_hazy.png').to(device)
    clean = load_image('./test_images/clear/54_GT.png').to(device)

    # 1. SGA 注意力对比
    for blk in [5]:
        visualize_sga_attention(model, hazy, block_idx=blk)

    # 2. 语义一致性
    if clean is not None:
        visualize_semantic_consistency(model, hazy, clean)

    print(f"Visualization results saved to {SAVE_PATH}")