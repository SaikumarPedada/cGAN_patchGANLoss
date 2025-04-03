import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vgg16
from tqdm.notebook import tqdm
import numpy as np
import cv2
import math

class KITTIStereoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, target_size=(256, 512)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

        left_dir = os.path.join(root_dir, 'image_2')
        right_dir = os.path.join(root_dir, 'image_3')
        disp_dir = os.path.join(root_dir, 'disp_occ_0')

        self.left_images = []
        self.right_images = []
        self.disp_maps = []

        for img_name in sorted(os.listdir(left_dir)):
            right_img_path = os.path.join(right_dir, img_name)
            disp_path = os.path.join(disp_dir, img_name.replace('.jpg', '.png'))
            if os.path.exists(right_img_path) and (os.path.exists(disp_path) or os.path.exists(disp_path.replace('.png', '.npy'))):
                self.left_images.append(img_name)
                self.right_images.append(img_name)
                self.disp_maps.append(img_name.replace('.jpg', '.png'))

        print(f"Valid pairs found: {len(self.left_images)}")

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left_img_path = os.path.join(self.root_dir, 'image_2', self.left_images[idx])
        right_img_path = os.path.join(self.root_dir, 'image_3', self.right_images[idx])
        disp_path = os.path.join(self.root_dir, 'disp_occ_0', self.disp_maps[idx])

        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        left_img = cv2.resize(left_img, self.target_size)
        right_img = cv2.resize(right_img, self.target_size)

        if disp_path.endswith('.npy'):
            disp_map = np.load(disp_path)
        else:
            disp_map = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        disp_map = cv2.resize(disp_map, self.target_size, interpolation=cv2.INTER_LINEAR)

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        left_img = torch.tensor(left_img.transpose(2, 0, 1) / 255.0, dtype=torch.float32)
        right_img = torch.tensor(right_img.transpose(2, 0, 1) / 255.0, dtype=torch.float32)
        disp_map = torch.tensor(disp_map[np.newaxis, :, :], dtype=torch.float32) / 256.0

        return left_img, right_img, disp_map

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width*height)
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return self.gamma*out + x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
\
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EnhancedResidualBlock(nn.Module):
    def __init__(self, channels):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm2d(channels)
        self.se = SEBlock(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.relu(out)

class HighResolutionGenerator(nn.Module):
    def __init__(self):
        super(HighResolutionGenerator, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.encoder1 = self.make_encoder_block(64, 128)
        self.encoder2 = self.make_encoder_block(128, 256)
        self.encoder3 = self.make_encoder_block(256, 512)

        self.attention = SelfAttention(512)

        self.residual_blocks = nn.Sequential(
            *[EnhancedResidualBlock(512) for _ in range(9)]
        )

        self.decoder3 = self.make_decoder_block(512, 256)
        self.decoder2 = self.make_decoder_block(512, 128)
        self.decoder1 = self.make_decoder_block(256, 64)

        self.detail_enhance = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            EnhancedResidualBlock(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            EnhancedResidualBlock(out_channels)
        )

    def make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            EnhancedResidualBlock(out_channels)
        )

    def forward(self, disp_map, left_img):
        x = torch.cat([left_img, disp_map], dim=1)

        x = self.init_conv(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        x = self.attention(e3)
        x = self.residual_blocks(x)

        x = self.decoder3(x)
        x = self.decoder2(torch.cat([x, e2], dim=1))
        x = self.decoder1(torch.cat([x, e1], dim=1))

        x = self.detail_enhance(torch.cat([x, x], dim=1))

        return x

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6, ndf=64, n_layers=3):
        super(PatchGANDiscriminator, self).__init__()

        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, left_img, right_img):
        x = torch.cat([left_img, right_img], dim=1)
        return self.model(x)

class EnhancedPerceptualLoss(nn.Module):
    def __init__(self):
        super(EnhancedPerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, generated, target):
        loss = 0.0
        gen_features1 = self.slice1(generated)
        target_features1 = self.slice1(target)
        loss += nn.functional.l1_loss(gen_features1, target_features1)

        gen_features2 = self.slice2(gen_features1)
        target_features2 = self.slice2(target_features1)
        loss += nn.functional.l1_loss(gen_features2, target_features2)

        gen_features3 = self.slice3(gen_features2)
        target_features3 = self.slice3(target_features2)
        loss += nn.functional.l1_loss(gen_features3, target_features3)

        return loss

class DetailLoss(nn.Module):
    def __init__(self):
        super(DetailLoss, self).__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, x, y):
        self.sobel_x = self.sobel_x.to(x.device)
        self.sobel_y = self.sobel_y.to(x.device)

        gx_x = F.conv2d(x, self.sobel_x.expand(x.size(1), -1, -1, -1), groups=x.size(1), padding=1)
        gy_x = F.conv2d(x, self.sobel_y.expand(x.size(1), -1, -1, -1), groups=x.size(1), padding=1)
        gx_y = F.conv2d(y, self.sobel_x.expand(y.size(1), -1, -1, -1), groups=y.size(1), padding=1)
        gy_y = F.conv2d(y, self.sobel_y.expand(y.size(1), -1, -1, -1), groups=y.size(1), padding=1)

        return F.l1_loss(gx_x, gx_y) + F.l1_loss(gy_x, gy_y)

def train(model_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize dataset and dataloader
    dataset = KITTIStereoDataset(
        model_config['data_dir'],
        target_size=(256, 512)
    )
    dataloader = DataLoader(dataset,
                          batch_size=model_config['batch_size'],
                          shuffle=True)

    # Initialize models
    generator = HighResolutionGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    # Initialize loss functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = EnhancedPerceptualLoss().to(device)
    criterion_detail = DetailLoss().to(device)

    # Initialize optimizers
    g_optimizer = optim.AdamW([
        {'params': generator.parameters(), 'lr': 0.0002},
    ], betas=(0.5, 0.999))

    d_optimizer = optim.AdamW(discriminator.parameters(),
                             lr=0.0001,
                             betas=(0.5, 0.999))

    # Learning rate scheduler with warmup
    def warmup_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) /
                                 (model_config['epochs'] - warmup_epochs)))

    g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer, warmup_lambda)
    d_scheduler = optim.lr_scheduler.LambdaLR(d_optimizer, warmup_lambda)

    # Training
    for epoch in range(model_config['epochs']):
        g_loss_epoch, d_loss_epoch = 0, 0
        #progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{model_config["epochs"]}')

        for batch_idx, (left_img, right_img, disp_map) in enumerate(dataloader):
            # Move data to device
            left_img = left_img.to(device)
            right_img = right_img.to(device)
            disp_map = disp_map.to(device)

            batch_size = left_img.size(0)

            # Get discriminator patch size
            patch_h, patch_w = discriminator(left_img, right_img).size()[2:]
            patch_ones = torch.ones(batch_size, 1, patch_h, patch_w, device=device)
            patch_zeros = torch.zeros(batch_size, 1, patch_h, patch_w, device=device)

            # Normalize disparity map
            disp_map = (disp_map - disp_map.min()) / (disp_map.max() - disp_map.min())

            #  Train Discriminator

            d_optimizer.zero_grad()

            # Generate fake right image
            with torch.no_grad():
                fake_right = generator(disp_map, left_img)

            # Real and fake discriminator outputs
            real_validity = discriminator(left_img, right_img)
            fake_validity = discriminator(left_img, fake_right)

            # Calculate discriminator losses
            d_loss_real = criterion_gan(real_validity, patch_ones)
            d_loss_fake = criterion_gan(fake_validity, patch_zeros)
            d_loss = (d_loss_real + d_loss_fake) * 0.5

            d_loss.backward()
            d_optimizer.step()
            g_optimizer.zero_grad()

            # Generate fake right image again for generator training
            fake_right = generator(disp_map, left_img)
            fake_validity = discriminator(left_img, fake_right)

            # Calculate generator losses
            g_loss_gan = criterion_gan(fake_validity, patch_ones)
            g_loss_l1 = criterion_l1(fake_right, right_img)
            g_loss_perceptual = criterion_perceptual(fake_right, right_img)
            g_loss_detail = criterion_detail(fake_right, right_img)

            # Combine losses with weights
            g_loss = (g_loss_gan * 1.0 + 
                     g_loss_l1 * 100.0 + 
                     g_loss_perceptual * 10.0 + 
                     g_loss_detail * 5.0)  

            g_loss.backward()
            g_optimizer.step()

            # Update running losses
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()


        # Step schedulers
        g_scheduler.step()
        d_scheduler.step()

        # Calculate epoch losses
        g_loss_epoch /= len(dataloader)
        d_loss_epoch /= len(dataloader)

        print(f'Epoch {epoch+1}/{model_config["epochs"]}:')
        print(f'Generator Loss: {g_loss_epoch:.4f}')
        print(f'Discriminator Loss: {d_loss_epoch:.4f}')

        '''
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            save_path = f'checkpoints/epoch_{epoch+1}'
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_scheduler_state_dict': g_scheduler.state_dict(),
                'd_scheduler_state_dict': d_scheduler.state_dict(),
                'g_loss': g_loss_epoch,
                'd_loss': d_loss_epoch
            }, os.path.join(save_path, 'checkpoint.pth'))'''
    os.makedirs(model_config['save_dir'], exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(model_config['save_dir'], 'final_generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(model_config['save_dir'], 'final_discriminator.pth'))

#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)

def main():

    # Testing dataset loading
    dataset = KITTIStereoDataset('training_stereo_2015')
    print(f"Number of valid pairs: {len(dataset)}")

    config = {
        'data_dir': 'training_stereo_2015',
        'save_dir': 'stereo_gan_Adv_models',
        'batch_size': 4,
        'epochs': 50
    }
    train(config)

if __name__ == "__main__":
    main()