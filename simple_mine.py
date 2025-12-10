import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import pydicom
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

class SimpleResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        # 第一个卷积层:3x3卷积,输出通道数为out_channels
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)
        # 第二个卷积层:3x3卷积,输出通道数为out_channels
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)

        #残差连接：如果输入通道数不等于输出通道数，使用1x1卷积层改变通道数；否则直接连接
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self,x):
        # 第一个卷积层
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.silu(h)
        # 第二个卷积层
        h = self.conv2(h)
        h = self.norm2(h)
        # 残差连接
        h += self.residual(x)
        return h

class SelfAttention(nn.Module):
    """
    自注意力模块
    """
    def __init__(self,channels,size):
        super().__init__()
        self.channels = channels
        self.size = size

        self.mha = nn.MultiheadAttention(channels,4,batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels,channels),
            nn.SiLU(inplace=True),
            nn.Linear(channels,channels)
        )

    def forward(self,x):
        """
        x:输入特征图[B,C,H,W]
        返回:自注意力输出[B,C,H,W]
        """
        # 转换为[B,HW,C]
        x = x.view(-1,self.channels,self.size*self.size).swapaxes(1,2)
        # 自注意力层
        x_ln = self.ln(x)
        attention_value,_ = self.mha(x_ln,x_ln,x_ln)
        attention_value += x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2,1).view(-1,self.channels,self.size,self.size)

class TimeEmbedding(nn.Module):
    def __init__(self,embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim # 时间嵌入维度
        # 学习时间步的非线性映射
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim,embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embedding_dim,embedding_dim)
        )
    
    def forward(self,t):
        """
        t:时间步[B]
        返回:时间嵌入[B,embedding_dim]
        """
        # 生成正弦余弦位置编码
        half_dim = self.embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim,device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0) #[B,half_dim]
        emb = torch.cat([torch.sin(emb),torch.cos(emb)],dim=1) # [B,embedding_dim]
        return self.layers(emb)

class SimpleUNet(nn.Module):
    def __init__(self,in_channels=2,base_channels=32,time_embedding_dim=64):
        super().__init__()
        self.time_embedding = time_embedding_dim

        # 时间嵌入模块
        self.time_embedding = TimeEmbedding(time_embedding_dim)

        # 1.下采样路径：提取高层特征(需要池化层吗？)
        # down1：仅残差块，无尺寸变化 [B,2,512,512] → [B,32,512,512]
        self.down1 = SimpleResidualBlock(in_channels,base_channels)

        # down2：步长2卷积（下采样）+残差块 → [B,32,512,512] → [B,64,256,256]
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels,base_channels*2,kernel_size=2,stride=2,padding=0),
            SimpleResidualBlock(base_channels*2,base_channels*2)
        )
        # down3：步长2卷积（下采样）+残差块 → [B,64,256,256] → [B,128,128,128]
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels*2,base_channels*4,kernel_size=2,stride=2,padding=0),
            SimpleResidualBlock(base_channels*4,base_channels*4)
        )
        # down4：步长2卷积（下采样）+残差块 → [B,128,128,128] → [B,256,64,64]
        self.down4 = nn.Sequential(
            nn.Conv2d(base_channels*4,base_channels*8,kernel_size=2,stride=2,padding=0),
            SimpleResidualBlock(base_channels*8,base_channels*8)
        )

        # 2.瓶颈层：连接上下采样路径，添加注意力
        # 瓶颈层：无尺寸变化 [B,256,64,64] → [B,256,64,64]
        self.bottleneck1 = SimpleResidualBlock(base_channels*8,base_channels*8)
        self.attn1 = SelfAttention(base_channels*8,64)
        self.bottleneck2 = SimpleResidualBlock(base_channels*8,base_channels*8)

        # 3.上采样路径：恢复特征图大小
        # up1：转置卷积（上采样）+残差块 → [B,256,64,64] → [B,128,128,128]
        # 关键：添加output_padding=0，确保尺寸严格匹配下采样的256x256
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8,base_channels*4,kernel_size=2,stride=2,padding=0, output_padding=0),
            SimpleResidualBlock(base_channels*4,base_channels*4)
        )
        # up2：转置卷积（上采样）+残差块 → [B,128,256,256] → [B,64,512,512]
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4,base_channels*2,kernel_size=2,stride=2,padding=0, output_padding=0),
            SimpleResidualBlock(base_channels*2,base_channels*2)
        )
        
        # up3：转置卷积（上采样）+残差块 → [B,64,512,512] → [B,32,512,512]
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2,base_channels,kernel_size=2,stride=2,padding=0, output_padding=0),
            SimpleResidualBlock(base_channels,base_channels)
        )

        # up4：转置卷积（上采样）+残差块 → [B,32,512,512] → [B,32,512,512]
        self.up4 = SimpleResidualBlock(base_channels,base_channels)

        # 4.输出层：预测噪声
        # [B,32,512,512] → [B,1,512,512]
        self.output = nn.Conv2d(base_channels,1,kernel_size=3,padding=1)
    
    def forward(self,x,t,cond):
        """
        x:输入图像[B,1,H,W]
        t:时间步[B]
        cond:条件图像LDCT[B,1,H,W]
        返回:预测噪声[B,1,H,W]
        """
        # 拼接输入图像和条件图像
        x = torch.cat([x,cond],dim=1) # [B, 2, H, W]
        #print(f"输入拼接后尺寸：{x.shape}")  # 调试：确认输入尺寸

        # 时间嵌入
        t_emb = self.time_embedding(t) #[B,embedding_dim]

        # 下采样路径+保存跳跃连接
        h1 = self.down1(x) # [B, base, H, W]
        #print(f"h1（down1后）尺寸：{h1.shape}")  # [B,32,512,512]
        h2 = self.down2(h1) # [B, base*2, H/2, W/2]
        #print(f"h2（down2后）尺寸：{h2.shape}")  # [B,64,256,256]
        h3 = self.down3(h2) # [B, base*4, H/4, W/4]
        #print(f"h3（down3后）尺寸：{h3.shape}")  # [B,128,128,128]
        h4 = self.down4(h3) # [B, base*8, H/8, W/8]
        #print(f"h4（down4后）尺寸：{h4.shape}")  # [B,256,64,64]

        # 瓶颈层
        b = self.bottleneck1(h4)
        b = self.attn1(b) + b
        b = self.bottleneck2(b)
        #print(f"瓶颈层后尺寸：{b.shape}")  # [B,128,128,128]

        # 上采样路径
        u1 = self.up1(b)+h3 # [B, base*2, H/2, W/2]
        #print(f"u1（up1后）尺寸：{u1.shape}")    # [B,64,256,256]
        u2 = self.up2(u1)+h2 # [B, base, H, W]
        #print(f"u2（up2后）尺寸：{u2.shape}")    # [B,32,512,512]
        u3 = self.up3(u2)+h1 # [B, base, H, W]
        #print(f"u3（up3后）尺寸：{u3.shape}")    # [B,32,512,512]
        u4 = self.up4(u3) # [B, base, H, W]
        #print(f"u4（up4后）尺寸：{u4.shape}")    # [B,32,512,512]

        return self.output(u4)

class SimpleDiffusion(nn.Module):
    def __init__(self,model,T=1000,device='cuda'):
        super().__init__()
        self.model = model
        self.T = T
        self.device = device

        # 定义噪声(beta)调度（向前加噪强度）
        self.betas = torch.linspace(0.0001,0.02,T).to(device) #线性
        self.alphas = 1. - self.betas #alpha = 1 - beta
        self.alphas_cumprod = torch.cumprod(self.alphas,dim=0)

    def q_sample(self,x_0,t,noise=None):
        """
        x_0:原始图像[B,1,H,W]
        t:时间步[B]
        noise:噪声[B,1,H,W]
        返回:加噪后的图像[B,1,H,W]
        正向过程:向x0中添加t步的噪声得到xt
        """
        # 标准正态噪声
        if noise is None:
            noise = torch.randn_like(x_0)

        # 提取t时刻的累积alpha
        sqrt_alphas_cumprod_t = self._extract(self.alphas_cumprod.sqrt(), t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract((1 - self.alphas_cumprod).sqrt(), t, x_0.shape)

        # 加噪公式：xt = sqrt(alpha_cumprod_t)*x0 + sqrt(1-alpha_cumprod_t)*noise
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self,x_t,t,cond):
        """
        x_t:加噪后的图像[B,1,H,W]
        t:时间步[B]
        cond:条件图像LDCT[B,1,H,W]
        返回:预测噪声[B,1,H,W]
        反向过程:从xt中预测噪声
        """
        # 预测噪声
        noise_pred = self.model(x_t,t,cond)

        # 提取参数
        alpha_t = self._extract(self.alphas,t,x_t.shape)
        alpha_cumprod_t = self._extract(self.alphas_cumprod,t,x_t.shape)
        beta_t = self._extract(self.betas,t,x_t.shape)

        # 计算均值：去噪的主要部分 
        mean = (1 / alpha_t.sqrt()) * (x_t - (beta_t / (1 - alpha_cumprod_t).sqrt()) * noise_pred)

        # 计算方差：噪声的方差,固定值
        if t[0] == 0:
            return mean
        else:
            var = beta_t
            noise = torch.randn_like(x_t)
            return mean + (var.sqrt()) * noise

    def _extract(self,arr,t,x_shape):
        """
        arr:参数数组[1,T]
        t:时间步[B]
        x_shape:输入图像形状[B,1,H,W]
        返回:提取的参数[B,1,1,1]
        从数组中提取t时刻的参数,并调整形状用于广播
        """
        batch_size = t.shape[0]
        # 提取每个batch对应的时间步值 [B]
        out = arr.gather(0, t.long())  # 强制转long，避免索引类型错误
        # 重塑为[B, 1, 1, 1]，匹配图像张量的维度（len(x_shape)=4 → 4-1=3个1）
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(arr.device)

    def training_loss(self,x0,cond):
        """
        x0:原始图像[B,1,H,W]
        cond:条件图像LDCT[B,1,H,W]
        返回:损失值[1]
        计算训练损失:MSE(预测噪声,真实噪声)
        """
        batch_size = x0.shape[0]
        # 随机采样时间步
        t = torch.randint(0,self.T,(batch_size,)).to(self.device)

        # 生成真实噪声
        noise = torch.randn_like(x0)

        # 正向扩散得到xt
        x_t = self.q_sample(x0,t,noise)

        # 预测噪声
        noise_pred = self.model(x_t,t,cond)

        # 计算损失
        loss = F.mse_loss(noise_pred,noise)
        return loss

    @torch.no_grad()
    def sample(self,cond,img_size):
        """
        cond:条件图像LDCT[B,1,H,W]
        img_size:输出图像形状[B,1,H,W]
        返回:生成图像[B,1,H,W]
        从条件图像cond中生成图像
        """
        B,C,H,W = cond.shape
        # 从标准正态分布采样初始噪声
        x_t = torch.randn(B,1,H,W).to(self.device)

        #从T-1逐步去噪到0
        for t in reversed(range(self.T)):
            # 预测噪声
            t_tensor = torch.full((B,),t,dtype=torch.long).to(self.device)
            # 去噪
            x_t = self.p_sample(x_t,t_tensor,cond)
        return x_t

class SimpleDataset(Dataset):
    def __init__(self, data_root, split="train"):
        """
        初始化数据集（按子文件夹+文件顺序匹配LDCT-NDCT）
        data_root: 数据根目录（即data文件夹路径）
        split: 数据集划分（train/test/eval）
        """
        self.split = split
        # 定义LDCT/NDCT的根目录（如data/train/ldct、data/train/ndct）
        self.ldct_root = os.path.join(data_root, split, "ldct")
        self.ndct_root = os.path.join(data_root, split, "ndct")

        # 步骤1：收集LDCT的文件 → 按子文件夹分组，存储「排序后的文件路径列表」
        # 格式：{"L067": [path1, path2, ...], "L068": [path1, path2, ...]}
        self.ldct_subdir_files = self._collect_files_by_subdir(self.ldct_root)
        # 步骤2：收集NDCT的文件 → 同LDCT的分组逻辑
        self.ndct_subdir_files = self._collect_files_by_subdir(self.ndct_root)

        # 步骤3：按子文件夹+顺序配对LDCT和NDCT文件
        self.paired_files = []
        # 遍历所有LDCT的子文件夹（如L067）
        for subdir_name in self.ldct_subdir_files.keys():
            # 跳过NDCT中不存在的子文件夹
            if subdir_name not in self.ndct_subdir_files:
                print(f"警告：NDCT中缺失子文件夹 {subdir_name}，已跳过该文件夹的所有文件")
                continue

            # 获取当前子文件夹下LDCT/NDCT的文件列表（已排序）
            ldct_files = self.ldct_subdir_files[subdir_name]
            ndct_files = self.ndct_subdir_files[subdir_name]

            # 校验当前子文件夹内文件数量一致（顺序匹配的前提）
            if len(ldct_files) != len(ndct_files):
                raise ValueError(
                    f"子文件夹 {subdir_name} 内LDCT文件数({len(ldct_files)})与NDCT文件数({len(ndct_files)})不一致，无法按顺序匹配！"
                )

            # 按顺序配对（第i个LDCT文件 ↔ 第i个NDCT文件）
            for ldct_path, ndct_path in zip(ldct_files, ndct_files):
                self.paired_files.append({
                    "ldct_path": ldct_path,
                    "ndct_path": ndct_path,
                    "subdir": subdir_name  # 可选：记录所属子文件夹，方便调试
                })

        # 最终校验：确保有至少一个匹配的文件对
        assert len(self.paired_files) > 0, f"{split}集未找到任何可匹配的LDCT-NDCT文件对！"
        print(f"{split}集共加载 {len(self.paired_files)} 个按顺序匹配的文件对")

    def _collect_files_by_subdir(self, root_dir):
        """
        按子文件夹收集IMA文件，返回：{子文件夹名: 排序后的IMA文件路径列表}
        :param root_dir: 根目录（如data/train/ldct）
        :return: dict，例：{"L067": ["data/train/ldct/L067/001.IMA", "data/train/ldct/L067/002.IMA"]}
        """
        subdir_files = {}
        # 遍历根目录下的所有子文件夹（如L067）
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(subdir_path):  # 跳过非文件夹文件
                continue

            # 收集当前子文件夹内的所有IMA文件
            ima_files = []
            for filename in os.listdir(subdir_path):
                if filename.endswith(".IMA"):
                    ima_files.append(os.path.join(subdir_path, filename))

            # 对IMA文件按文件名排序（保证顺序固定，避免随机）
            ima_files.sort()  # 核心：排序后保证顺序一致

            # 存储当前子文件夹的文件列表（非空才存储）
            if ima_files:
                subdir_files[subdir] = ima_files
            else:
                print(f"警告：子文件夹 {subdir} 内无IMA文件，已跳过")

        return subdir_files

    def _read_dicom(self, file_path):
        """
        读取IMA(DICOM)文件并预处理为512x512张量
        :param file_path: IMA文件完整路径
        :return: 归一化后的张量 [1, 512, 512]
        """
        try:
            # 读取DICOM文件
            ds = pydicom.dcmread(file_path)
            # 提取像素数据（512x512），转换为float32
            img = ds.pixel_array.astype(np.float32)

            # 强制校验图像尺寸（确保是512x512）
            assert img.shape == (512, 512), \
                f"文件 {file_path} 尺寸异常（需512x512），实际为 {img.shape}"

            # 像素值归一化（映射到0~1区间，避免数值过大）
            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 1e-6:  # 避免除以0
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img)

            # 转换为PyTorch张量，添加通道维度 [H, W] → [1, H, W]
            img_tensor = torch.from_numpy(img).unsqueeze(0)
            return img_tensor

        except Exception as e:
            raise RuntimeError(f"读取文件 {file_path} 失败：{str(e)}")

    def __len__(self):
        """返回数据集总长度（按顺序匹配的文件对数量）"""
        return len(self.paired_files)

    def __getitem__(self, idx):
        """
        获取单个样本（按顺序匹配）
        :param idx: 样本索引
        :return: 字典 {ldct: [1,512,512], ndct: [1,512,512]}
        """
        # 获取按顺序配对的文件路径
        pair = self.paired_files[idx]
        ldct_path = pair["ldct_path"]
        ndct_path = pair["ndct_path"]

        # 读取并预处理LDCT和NDCT
        ldct = self._read_dicom(ldct_path)
        ndct = self._read_dicom(ndct_path)

        return {"ldct": ldct, "ndct": ndct}

def train(model,dataloader,epoch=10,lr=1e-4,device='cuda'):
    """
    训练模型
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=len(dataloader),eta_min=1e-6)

    for epoch_idx in range(epoch):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader,desc=f"Epoch {epoch_idx+1}/{epoch}", unit="batch")

        for batch_idx,batch in enumerate(pbar):
            ldct = batch["ldct"].to(device)
            ndct = batch["ndct"].to(device)
            # 计算损失
            loss = model.training_loss(ndct,ldct)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 累加损失
            total_loss += loss.item()

            # 计算平均损失
            avg_loss = total_loss / (batch_idx+1)
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            # 更新进度条信息（显示当前batch损失和平均损失）
            pbar.set_postfix({
                "batch_loss": f"{loss.item():.6f}",
                "avg_loss": f"{avg_loss:.6f}",
                "lr": f"{current_lr:.8f}"
            })

        # 每个epoch结束后打印总平均损失
        epoch_avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch_idx+1} 结束，平均损失: {epoch_avg_loss:.6f}\n")

def main():
    """
    主函数,用于训练模型
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1.初始化模型
    unet = SimpleUNet(in_channels=2,base_channels=32,time_embedding_dim=64)
    diffusion = SimpleDiffusion(unet,T=50,device=device)

    # 2.加载数据集
    dataset = SimpleDataset(data_root='./data',split="train")
    dataloader = DataLoader(dataset,batch_size=2,shuffle=True)

    # 3.训练模型
    train(diffusion,dataloader,epoch=10,lr=1e-4,device=device)
    # 4.保存模型
    torch.save(unet.state_dict(),'simple_diffusion_model.pth')

    # 5.采样测试
    test_dataset = SimpleDataset(data_root='./data',split="test")
    test_ldct = test_dataset[0]["ldct"].unsqueeze(0).to(device)
    test_ndct = test_dataset[0]["ndct"]
    diffusion.eval()
    with torch.no_grad():
        generated = diffusion.sample(test_ldct,img_size=(1,512,512))

    # 6.可视化结果
    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.imshow(test_ldct[0, 0].cpu().numpy(), cmap='gray')
    plt.title('Input LDCT')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(generated[0, 0].cpu().numpy(), cmap='gray')
    plt.title('Generated NDCT')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(test_ndct[0].cpu().numpy(), cmap='gray')
    plt.title('Ground Truth NDCT')
    plt.axis('off')

if __name__ == '__main__':
    main()