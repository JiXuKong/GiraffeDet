import numpy as np

def mosaic(img1, img2, img3, img4, label1, label2, label3, label4, img_size):
    labels5 = []
    s = img_size
    yc, xc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    for i in range(1, 5):
        if i == 1:
            # place img in img5
            #1, # top left
            h, w = img1.shape[:2]
            img5 = np.full((s * 2, s * 2, img1.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            x = label1
            
        if i == 2:
            h, w = img2.shape[:2]
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            x = label2
            
        if i == 3:
            h, w = img3.shape[:2]
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            x = label3
            
        if i == 4:
            h, w = img4.shape[:2]
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            x = label4
            
            
        img5[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b
        
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 0] = x[:, 0] + padw
            labels[:, 1] = x[:, 1] + padh
            labels[:, 2] = x[:, 2] + padw
            labels[:, 3] = x[:, 3] + padh
        labels5.append(labels)
    return img54, labels5
    
