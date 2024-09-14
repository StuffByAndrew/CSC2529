import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.fft import fft2, ifft2, fftshift, ifftshift

hw_dir = Path(__file__).parent

# Load images
img1 = io.imread(hw_dir/'image1.png').astype(np.float64)/255
img2 = io.imread(hw_dir/'image2.png').astype(np.float64)/255

print(img1.shape)

# Part (a)
W = img1.shape[0]       # = 1001 dots
d = np.array([0.4, 2])  # distances (m)
dpi = 300               # dots per inch

#### YOUR CODE HERE ####
print("part 1\n------------\n")
CMPI = 2.54 # cm per inch
dpcm = dpi/CMPI
physical_size = W/dpi
print(f"{physical_size=}")

def phys_dist(D, ang):
    return 2 * D * np.tan(np.deg2rad(ang/2))

print(f"40cm: pixels / deg {dpcm * phys_dist(40, 1)}")
print(f"2m: pixels / deg {dpcm * phys_dist(200, 1)}")

# Part (b)
cpd = 5   # Peak contrast sensitivity location (cycles per degree)

#### YOUR CODE HERE ####
print("\npart 2\n------------\n")
img_freq40 = cpd / (dpcm * phys_dist(40, 1))
print(f"40cm: pixels / deg {img_freq40}")
img_freq200 = cpd / (dpcm * phys_dist(200, 1))
print(f"2m: pixels / deg {img_freq200}")
img_freq_mid = np.mean([img_freq200, img_freq40])

# Part (c)
# Hint: fft2, ifft2, fftshift, and ifftshift functions all take an |axes|
# argument to specify the axes for the 2D DFT. e.g. fft2(arr, axes=(1, 2))
# Hint: Check out np.meshgrid.

#### YOUR CODE HERE ####
print("\npart 3\n------------\n")
freq_per_px = 0.5/((W-1)/2)
filter_rad = img_freq_mid / freq_per_px
print(f"{filter_rad=}")
x, y = np.meshgrid(np.arange(-500,500+1), np.arange(-500,500+1))
#### Change these to the correct values for the high- and low-pass filters
hpf = np.sqrt(x**2 + y**2) >= filter_rad
lpf = np.sqrt(x**2 + y**2) <= filter_rad
hpf, lpf = hpf[..., np.newaxis], lpf[..., np.newaxis]

#### Apply the filters to create the hybrid image
hybrid_img_norm = np.real(ifft2(ifftshift(
    lpf * fftshift(fft2(img1, axes=(0,1)))
    + hpf * fftshift(fft2(img2, axes=(0,1)))
), axes=(0,1)))
print(np.sum(hybrid_img_norm > 1))
print(hybrid_img_norm.dtype)
hybrid_img = hybrid_img_norm * 255
print(hybrid_img.shape)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
axs[0,0].imshow(img2)
axs[0,0].axis('off')
axs[0,1].imshow(hpf, cmap='gray')
axs[0,1].set_title("High-pass filter")
axs[1,0].imshow(img1)
axs[1,0].axis('off')
axs[1,1].imshow(lpf, cmap='gray')
axs[1,1].set_title("Low-pass filter")
plt.savefig("hpf_lpf.png", bbox_inches='tight')
io.imsave("hybrid_image.png", np.clip(hybrid_img, a_min=0, a_max=255.).astype(np.uint8))
