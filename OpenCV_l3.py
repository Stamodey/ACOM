import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    """Создание и нормализация Гауссовой матрицы"""
    k = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            x, y = i - k, j - k
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def apply_gaussian_filter(image, kernel):
    """Применение фильтра вручную"""
    h, w = image.shape
    k = kernel.shape[0] // 2
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(k, h - k):
        for j in range(k, w - k):
            region = image[i - k:i + k + 1, j - k:j + k + 1]
            output[i, j] = np.sum(region * kernel)
    return np.clip(output, 0, 255).astype(np.uint8)

def show_image(title, img):
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

img = cv2.imread('sample_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for size in [3, 5, 7]:
    kernel = gaussian_kernel(size, sigma=1.0)
    print(f"\nГауссова матрица {size}x{size}:\n", kernel)

params = [
    (3, 0.8),
    (5, 1.5)
]

for size, sigma in params:
    kernel = gaussian_kernel(size, sigma)
    filtered = apply_gaussian_filter(gray, kernel)
    show_image(f'Ручной фильтр: size={size}, sigma={sigma}', filtered)

for size, sigma in params:
    opencv_blur = cv2.GaussianBlur(gray, (size, size), sigma)
    show_image(f'OpenCV фильтр: size={size}, sigma={sigma}', opencv_blur)

show_image('Оригинал', gray)
plt.show()
