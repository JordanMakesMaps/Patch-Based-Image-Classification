import skimage
import numpy as np 

# Custom callback for reducing learning rate
def reduceLR(epoch, lr):

    new_lr = (lr * 1 / (1 + 0.1 * epoch))
    print("New Learning Rate:", round(new_lr, 8))
    return new_lr


# Zooms in on an image by some ratio (< 0 zooms in)
def crop_center_by(image, zoom):
    w, h = image.shape[:2]
    offset = int(zoom * w)
    x, y = int(w / 2), int(h / 2)
    either_side = int(offset / 2)
    return image[x - either_side : x + either_side, y - either_side : y + either_side]


# For each iteration, the image is center-cropped so each
# successive image is *zoomed* in on, and resized to 112, 112.
def panel(img):

    panel_img = np.zeros([224, 224, 3])

    # Values between .5 and .95 are worth trying
    zoom = .72

    scale_1 = img                                                                   # 224 by 224 patch
    scale_2 = scale_1                                                               # 224 by 224 patch
    scale_1 = skimage.transform.resize(scale_1, (112, 112), anti_aliasing = True)   # Taking ^ that ^ image and downsizing to 112 
    
    scale_2 = crop_center_by(scale_2, zoom)                                         # From the copy, cropping off excess % (w, h)
    scale_3 = scale_2                                                               # Making a copy of ^ this ^
    scale_2 = skimage.transform.resize(scale_2, (112, 112), anti_aliasing = True)   # Downsizing the cropped zoomed copy to 112 by 112
    
    scale_3 = crop_center_by(scale_3, zoom)                                         # From the copy of the zoomed copy, cropping off excess % (w, h)
    scale_4 = scale_3                                                               # Making a copy of ^ this ^
    scale_3 = skimage.transform.resize(scale_3, (112, 112), anti_aliasing = True)   # Downsizing the cropped zoomed copy of the zoomed copy to 112 by 112

    scale_4 = crop_center_by(scale_4, zoom)                                         # From the zoomed copy of the zoomed copy, cropping of excess % (w, h)
    scale_4 = skimage.transform.resize(scale_4, (112, 112), anti_aliasing = True)   # Downsizing the cropped zoomed copy of the zoomed copy of the zoomed copy 112 by 112 :)
    
    panel_img[0:112, 0:112, :] = scale_1
    panel_img[0:112, 112:224, :] = scale_2
    panel_img[112:224, 0:112, :] = scale_3
    panel_img[112:224, 112:224, :] = scale_4

    del img, scale_1, scale_2, scale_3, scale_4
    
    return panel_img                                                                 # Image returned is a 4-panel image, (224, 224), [0, 1]
