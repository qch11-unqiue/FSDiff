# diffusion/ddpm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class SAR_RFI_Diffusion(nn.Module):
    def __init__(self, denoise_fn, timesteps=1000):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.timesteps = timesteps

        # --- Noise Schedule (保持不变) ---
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1. / alphas))
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def predict_x0_from_eps(self, x_t, t, eps):
        """
        根据 x_t 和预测的噪声 eps 反推 x_0。
        不依赖复杂的预计算变量，直接从 alphas_cumprod 计算。
        """
        # 1. 获取 alpha_bar_t
        # self.alphas_cumprod 应该是在 __init__ 里定义的
        alpha_bar_t = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)

        # 2. 应用公式
        # x_0 = (x_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1. - alpha_bar_t)

        return (x_t - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t
    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    # ================= 修改 1: p_losses 接收 mask =================
    def p_losses(self, x_start, condition, mask, t):
        """
        x_start: Clean Image (GT) [B, 3, H, W]
        condition: RFI Image [B, 3, H, W]
        mask: Mask Image [B, 1, H, W]
        """
        noise = torch.randn_like(x_start)

        # 生成加噪图 (Noisy Clean)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # --- 核心修改 ---
        # 拼接: [Noisy(3), Condition(3), Mask(1)] -> 总共 7 通道
        model_input = torch.cat([x_noisy, condition, mask], dim=1)

        # 预测噪声
        predicted_noise = self.denoise_fn(model_input, t)

        loss = F.mse_loss(predicted_noise, noise)
        return loss

    # ================= 修改 2: p_sample 接收 mask =================
    @torch.no_grad()
    def p_sample(self, x, condition, mask, t, t_index):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        # --- 核心修改 ---
        # 推理时也必须拼接 Mask，否则通道数对不上
        model_input = torch.cat([x, condition, mask], dim=1)

        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.denoise_fn(model_input, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # ================= 修改 3: sample_with_mask 逻辑 =================
    @torch.no_grad()
    def sample_with_mask(self, rfi_image, mask, release_ratio=0.15):
        """
        全流程 Mask 指导采样 + 末期释放策略 (Early Release)

        参数:
            rfi_image: 带有干扰的原始条件图
            mask: 掩码 (1表示干扰区域, 0表示干净背景)
            release_ratio: 末期释放的步数比例。默认0.15，意味着在最后15%的去噪步数中，
                           停止强制替换背景，让模型自然平滑边缘。
        """
        device = rfi_image.device
        b, c, h, w = rfi_image.shape

        # 1. 初始噪声
        img = torch.randn_like(rfi_image)

        # 2. 计算释放阈值
        # 如果 timesteps 是 1000，release_ratio 是 0.15，那么 release_step 就是 150
        release_step = int(self.timesteps * release_ratio)

        for i in tqdm(reversed(range(0, self.timesteps)), desc='Masked Sampling', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)

            # --- A. 模型预测 ---
            # UNet 结合 mask 进行预测
            img_pred = self.p_sample(img, rfi_image, mask, t, i)

            # --- B. 获取真实的加噪背景 ---
            if i > 0:
                noise = torch.randn_like(rfi_image)
                t_next = torch.full((b,), i, device=device, dtype=torch.long)
                known_noisy_image = self.q_sample(x_start=rfi_image, t=t_next, noise=noise)
            else:
                known_noisy_image = rfi_image

            # --- C. 融合策略 (末期释放) ---
            mask_expanded = mask.expand_as(img)

            if i > release_step:
                # 【前中期】强制保留背景：大框架严格对齐真实 SAR 背景分布
                img = img_pred * mask_expanded + known_noisy_image * (1. - mask_expanded)
            else:
                # 【末期】释放限制：把整张图的控制权交还给扩散模型，利用模型的卷积特性自然弥合边缘
                img = img_pred

        # --- D. (可选) 最终强制背景还原 ---
        # 经过末期释放，背景的极小部分像素可能发生微弱改变(肉眼通常不可见)。
        # 如果你的任务要求非 Mask 区域的像素值【必须 100% 绝对等于】原图，可以取消下面这行的注释：
        #img = img * mask_expanded + rfi_image * (1. - mask_expanded)

        return img
    # ================= 修改 4: forward 接收 mask =================
    def forward(self, x_clean, x_rfi, mask):
        b, c, h, w = x_clean.shape
        t = torch.randint(0, self.timesteps, (b,), device=x_clean.device).long()
        return self.p_losses(x_clean, x_rfi, mask, t)