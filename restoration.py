#Import library opencv
import cv2 as cv
import numpy as np

def padding (img, filter_size):
    pad = filter_size // 2

    if len(img.shape) == 3:
        return np.pad(img,((pad,pad),(pad,pad),(0,0)),mode="edge")
    return np.pad(img,((pad,pad),(pad,pad)),mode="edge")

def manual_convolution(image):
    kernel = np.array([[0,0,0],
                       [0,1,0],
                       [0,0,0]])
    kh, kw = kernel.shape

    output = np.zeros_like(image,dtype=np.float32)

    padded_img = padding(image,kernel.shape[0])

    for i in range (img_h):
        for j in range (img_w):
            roi = padded_img[i:i+kh,j:j+kw, :]
            output[i,j] = np.sum(roi * kernel[:,:,np.newaxis],axis=(0,1))

    return np.clip(output,0,255).astype(np.uint8)

def gaussian_filter(image,size =3):
    kernel_5 = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4], 
    [1,  4,  6,  4, 1]
], dtype=np.float32) / 256.0
    
    kernel_3 = np.array([
        [1,2,1],
        [2,4,2],
        [1,2,1]
    ], dtype=np.float32) / 16.0

    kernel_7 = np.array([
    [1,  3,   7,   9,   7,   3,  1],
    [3,  12,  26,  33,  26,  12, 3],
    [7,  26,  55,  71,  55,  26, 7],
    [9,  33,  71,  92,  71,  33, 9],
    [7,  26,  55,  71,  55,  26, 7],
    [3,  12,  26,  33,  26,  12, 3],
    [1,  3,   7,   9,   7,   3,  1]
], dtype=np.float32) / 1111.0

    if size == 5:
        kernel = kernel_5
    elif size == 3:
        kernel = kernel_3
    elif size == 7:
        kernel = kernel_7
    else:
        return "kernel tidak tersedia"
    
    kh, kw = kernel.shape

    output = np.zeros_like(image,dtype=np.float32)

    padded_img = padding(image,kernel.shape[0])

    for i in range (img_h):
        for j in range (img_w):
            roi = padded_img[i:i+kh,j:j+kw, :]
            output[i,j] = np.sum(roi * kernel[:,:,np.newaxis],axis=(0,1))

    return np.clip(output,0,255).astype(np.uint8)

def median_filter(image,size=3):
    padded_img = padding(image,size)
    output = np.zeros_like(image,dtype=np.float32)

    for i in range (img_h):
        for j in range(img_w):
            roi= padded_img[i:i+size,j:j+size]
            output[i,j] = np.median(roi,axis=(0,1))
    
    return np.clip(output,0,255).astype(np.uint8)


def histogram_equalization(image):
    # 1. Convert RGB to Y (Luminance) manually
    # image is (H, W, 3) 
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    y = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    
    # 2. Equalize only the Y channel
    hist, _ = np.histogram(y.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    y_equalized = cdf[y]
    y_final = (0.2 * y_equalized + 0.8 * y).astype(np.uint8)

    scaling_factor = (y_final.astype(float) + 1e-5) / (y.astype(float) + 1e-5)
    
    output = np.zeros_like(image)
    output[:,:,0] = np.clip(r * scaling_factor, 0, 255)
    output[:,:,1] = np.clip(g * scaling_factor, 0, 255)
    output[:,:,2] = np.clip(b * scaling_factor, 0, 255)
    
    return output.astype(np.uint8)

def sharpening(image,size=3):
    kernel = np.array([[0,-1,0],
                       [-1.,5,-1],
                       [0,-1,0]], dtype=np.float32)

    padded_img = padding(image,size)
    output = np.zeros_like(image,dtype=np.float32)
    kh, kw = kernel.shape

    for i in range (img_h):
        for j in range (img_w):
            roi = padded_img[i:i+kh,j:j+kw, :]
            output[i,j] = np.sum(roi * kernel[:,:,np.newaxis],axis=(0,1))

    return np.clip(output,0,255).astype(np.uint8)

def unsharp_mask(img, ksize=5, alpha=0.8, threshold=15): 
    blurred = gaussian_filter(img, ksize)
    
    mask = img.astype(np.float32) - blurred.astype(np.float32)
    
    mask[np.abs(mask) < threshold] = 0
    
    sharpened = img.astype(np.float32) + alpha * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)

#input
image = cv.imread("bahan/test_image_lena_noisy.png")
img_h, img_w = image.shape[:2]

#process
step1 = median_filter(image,3)
step2 = gaussian_filter(step1,3)
step3 = histogram_equalization(step2)
step4 = median_filter(step3,5)
step5 = gaussian_filter(step4)
step6 = unsharp_mask(step5)

#result show
cv.imshow("origin", image)
cv.imshow("step1", step1)
cv.imshow("step2", step2)
cv.imshow("step3", step3)
cv.imshow("step4", step4)
cv.imshow("step5", step5)
cv.imshow("step6", step6)
# Menunggu sampai tombol di tekan
cv.waitKey(0)
 
# Menutup semua window
cv.destroyAllWindows()