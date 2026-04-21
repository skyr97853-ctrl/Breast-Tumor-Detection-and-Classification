from torchvision import transforms


def segment(img,x1, y1, x2, y2):
    return img.crop((x1, y1, x2, y2))


def resize_pad(img):
    w, h = img.size
    pad_len = int(abs(w - h)/2)
    if w > h:
        img = transforms.Pad((0, pad_len, 0, pad_len), fill=0)(img)
    else:
        img = transforms.Pad((pad_len, 0, pad_len, 0), fill=0)(img)
    img = transforms.Resize((224, 224))(img)

    return img


def main_process(img, x1, y1, x2, y2):
    img = segment(img, x1, y1, x2, y2)
    img = resize_pad(img)
    return img

