from ultralytics import SAM
from ultralytics import FastSAM
import cv2
import numpy as np
import matplotlib.pyplot as plt


class RoadBinaryMasker():
    def __init__(self, model_path='FastSAM-s.pt', device='cpu'):
        self.model = FastSAM(model_path).to(device)

    def get_mask(self, img_path):
        results = self.model.predict(img_path)  
        masks = results[0].masks.data
        masks = masks.numpy()   
        h = masks.shape[1]
        w = masks.shape[2]

        road_points = [h-h//5 , w//2]
        road_point_y , road_point_x = road_points
        road_mask_index = None
        for i in range(len(masks)):
            if masks[i][road_point_y][road_point_x] == 1:
                road_mask_index = i
                break

        if road_mask_index is None:
            return masks
        
        road_mask = masks[road_mask_index]

        return road_mask
    
if __name__ == '__main__':
    masker = RoadBinaryMasker()
    img_path = 'road4.jpeg'
    mask = masker.get_mask(img_path)
    print(mask)
    # visualize the mask
    plt.imshow(mask , cmap='gray')
    plt.show()