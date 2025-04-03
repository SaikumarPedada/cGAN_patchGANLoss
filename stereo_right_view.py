import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from google.colab import drive
import os

# Mount Google Drive (if using Colab)
#drive.mount('/content/drive')

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

class ImagePostProcessor:
    @staticmethod
    def match_exposure(generated_img, reference_img, method='histogram_matching'):
        """ Adjust the exposure of generated image to match reference image. """
        if isinstance(generated_img, np.ndarray):
            generated_np = generated_img
            reference_np = reference_img
        else:
            is_tensor = True
            device = generated_img.device
            generated_np = generated_img.detach().cpu().numpy()
            reference_np = reference_img.detach().cpu().numpy()
            
            if generated_np.ndim == 4:
                generated_np = generated_np[0].transpose(1, 2, 0)
                reference_np = reference_np[0].transpose(1, 2, 0)

        # Ensure values are in [0, 1]
        generated_np = generated_np.astype(np.float32) / 255.0 if generated_np.max() > 1 else generated_np
        reference_np = reference_np.astype(np.float32) / 255.0 if reference_np.max() > 1 else reference_np
        
        if method == 'histogram_matching':
            matched = np.zeros_like(generated_np)
            for i in range(3):
                matched[..., i] = ImagePostProcessor._match_histograms(
                    generated_np[..., i],
                    reference_np[..., i]
                )
        else:
            matched = ImagePostProcessor._match_statistics(generated_np, reference_np)
            
        # Scale back to [0, 255] range
        matched = np.clip(matched * 255.0, 0, 255).astype(np.uint8)
        return matched

    @staticmethod
    def _match_histograms(source, reference):
        """Match the histogram of source to reference."""
        src_hist, src_bins = np.histogram(source.flatten(), bins=256, range=(0, 1), density=True)
        ref_hist, ref_bins = np.histogram(reference.flatten(), bins=256, range=(0, 1), density=True)
        
        src_cdf = src_hist.cumsum()
        src_cdf = src_cdf / src_cdf[-1]
        ref_cdf = ref_hist.cumsum()
        ref_cdf = ref_cdf / ref_cdf[-1]
        
        lookup_table = np.zeros(256)
        src_values = (src_bins[:-1] + src_bins[1:]) / 2
        ref_values = (ref_bins[:-1] + ref_bins[1:]) / 2
        
        for i in range(256):
            lookup_table[i] = np.interp(src_cdf[i], ref_cdf, ref_values)
            
        matched = np.interp(source.flatten(), src_values, lookup_table)
        return matched.reshape(source.shape)

    @staticmethod
    def _match_statistics(source, reference):
        """Match mean and standard deviation of source to reference."""
        matched = np.zeros_like(source)
        
        for i in range(3):
            src_mean = np.mean(source[..., i])
            src_std = np.std(source[..., i])
            ref_mean = np.mean(reference[..., i])
            ref_std = np.std(reference[..., i])
            
            matched[..., i] = ((source[..., i] - src_mean) * (ref_std / src_std)) + ref_mean
            
        return matched

def generate_right_view(generator, left_image_path, disparity_map_path, output_path, device):
    """
    Generate right view using the trained HighResolutionGenerator with exposure correction.
    """
    left_img = cv2.imread(left_image_path)
    if left_img is None:
        raise FileNotFoundError(f"Left image not found at {left_image_path}")

    original_height, original_width = left_img.shape[:2]
    original_left = left_img.copy() 

    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

    model_size = (512, 256)
    left_img_resized = cv2.resize(left_img, model_size)
    left_img_tensor = torch.tensor(left_img_resized.transpose(2, 0, 1) / 255.0, dtype=torch.float32)
    left_img_tensor = left_img_tensor.unsqueeze(0).to(device)

    if disparity_map_path.endswith('.npy'):
        disp_map = np.load(disparity_map_path)
    else:
        disp_map = cv2.imread(disparity_map_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    if disp_map is None:
        raise FileNotFoundError(f"Disparity map not found at {disparity_map_path}")

    disp_map_resized = cv2.resize(disp_map, model_size, interpolation=cv2.INTER_LINEAR)
    disp_map_tensor = torch.tensor(disp_map_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    disp_map_tensor = (disp_map_tensor - disp_map_tensor.min()) / (disp_map_tensor.max() - disp_map_tensor.min())

    with torch.no_grad():
        right_img = generator(disp_map_tensor, left_img_tensor)
        right_img = right_img.cpu().numpy()[0]

    right_img = np.transpose(right_img, (1, 2, 0))
    right_img = (right_img + 1) / 2.0 * 255.0
    right_img = np.clip(right_img, 0, 255).astype(np.uint8)

    right_img = cv2.resize(right_img, (original_width, original_height))
    
    right_img = ImagePostProcessor.match_exposure(
        right_img,
        cv2.cvtColor(original_left, cv2.COLOR_BGR2RGB),
        method='histogram_matching'
    )
    
    right_img = cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, right_img)
    print(f"Generated right view saved to {output_path}")
    return right_img

def test_model():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = HighResolutionGenerator().to(device)

    generator_path = 'stereo_gan_Adv_models/stereo_gan_Adv_models/final_generator.pth'

    if not os.path.exists(generator_path):
        raise FileNotFoundError(f"Generator model not found at {generator_path}")

    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    print("Generator model loaded successfully.")

    left_image_path = 'inputs_output_cGAN/inputs_output_cGAN/test_image_3.jpg'
    disparity_map_path = 'inputs_output_cGAN/inputs_output_cGAN/disparity_map.npy'
    output_path = 'inputs_output_cGAN/inputs_output_cGAN/generated_right_view.jpg'

    # Generate and display the right view
    generated_img = generate_right_view(generator, left_image_path, disparity_map_path, output_path, device)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    left_img = cv2.imread(left_image_path)
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    plt.imshow(left_img)
    plt.title('Left Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    generated_img_rgb = cv2.cvtColor(generated_img, cv2.COLOR_BGR2RGB)
    plt.imshow(generated_img_rgb)
    plt.title('Generated Right Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_model()