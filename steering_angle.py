import numpy as np
import matplotlib.pyplot as plt
import cv2
from road_mask import RoadBinaryMasker

class SteeringAngleCalculator:
    def __init__(self, img_width, img_height, clip_range=(20, 60)):
        """
        Initializes the Steering Angle Calculator.
        
        Args:
            img_width: Width of the image.
            img_height: Height of the image.
            clip_range: Tuple specifying the percentage range (start%, end%) of the height to retain.
        """
        self.img_width = img_width
        self.img_height = img_height
        self.clip_range = clip_range  # (start%, end%)

    def calculate_steering_angle(self, road_mask, debug=False):
        # Calculate clipping boundaries
        clip_start = int(self.img_height * (self.clip_range[0] / 100))
        clip_end = int(self.img_height * (self.clip_range[1] / 100))
        clipped_mask = road_mask[clip_start:clip_end, :]  # Retain the specified range of the mask

        # Detect road boundaries using edges
        edges = cv2.Canny((clipped_mask * 255).astype(np.uint8), 50, 150)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

        visualization_img = cv2.cvtColor((road_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        if lines is None:
            # CASE 3: No boundaries detected
            if debug:
                print("No boundaries detected")
            return 0.0, visualization_img  # Keep steering angle straight

        left_lines = []
        right_lines = []

        # Separate left and right boundaries
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Adjust y-coordinates to the original mask
                y1 += clip_start
                y2 += clip_start
                slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
                if slope < 0 and x1 < self.img_width // 2:  # Left boundary
                    left_lines.append((x1, y1, x2, y2))
                    cv2.line(visualization_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for left lines
                elif slope > 0 and x2 > self.img_width // 2:  # Right boundary
                    right_lines.append((x1, y1, x2, y2))
                    cv2.line(visualization_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for right lines

        if not left_lines and not right_lines:
            # CASE 3: No boundaries detected
            return 0.0, visualization_img

        # Define utility to calculate slope and intercept
        def average_slope_intercept(lines):
            x_coords, y_coords = [], []
            for x1, y1, x2, y2 in lines:
                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])
            poly = np.polyfit(x_coords, y_coords, 1)  # Linear fit
            slope = poly[0]
            intercept = poly[1]
            return slope, intercept

        left_slope, left_intercept = average_slope_intercept(left_lines) if left_lines else (None, None)
        right_slope, right_intercept = average_slope_intercept(right_lines) if right_lines else (None, None)

        if left_slope is not None and right_slope is not None:
            # CASE 1: Both boundaries visible
            intersection_x = (right_intercept - left_intercept) / (left_slope - right_slope)
            intersection_y = left_slope * intersection_x + left_intercept
        elif left_slope is not None:
            # CASE 2: Only left boundary visible
            intersection_x = self.img_width
            intersection_y = left_slope * intersection_x + left_intercept
        elif right_slope is not None:
            # CASE 2: Only right boundary visible
            intersection_x = 0
            intersection_y = right_slope * intersection_x + right_intercept
        else:
            # CASE 3: No boundaries detected
            return 0.0, visualization_img

        # Draw intersection point
        intersection_point = (int(intersection_x), int(intersection_y))
        cv2.circle(visualization_img, intersection_point, 5, (255, 0, 0), -1)  # Blue for intersection point

        # Calculate steering angle
        vehicle_mid_x = self.img_width // 2
        vehicle_mid_y = self.img_height
        delta_x = intersection_x - vehicle_mid_x
        delta_y = vehicle_mid_y - intersection_y

        if delta_y == 0:
            return 0.0, visualization_img  # No steering required if perfectly aligned

        steering_angle = np.arctan(delta_x / delta_y) * 180 / np.pi  # Convert to degrees

        # Draw midline
        cv2.line(visualization_img, (vehicle_mid_x, vehicle_mid_y), intersection_point, (255, 255, 0), 2)  # Yellow line
        return steering_angle, visualization_img

if __name__ == "__main__":
    # Example usage
    masker = RoadBinaryMasker(model_path="FastSAM-s.pt")
    img_path = "road4.jpeg"
    mask = masker.get_mask(img_path)

    if mask is not None:
        height, width = mask.shape
        calculator = SteeringAngleCalculator(width, height, clip_range=(20, 80))
        angle, visualization = calculator.calculate_steering_angle(mask, debug=True)

        print(f"Calculated Steering Angle: {angle:.2f} degrees")

        # Plot the visualization
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
        plt.title(f"Steering Angle Visualization (Angle: {angle:.2f}°)")
        plt.axis("off")
        plt.show()
    else:
        print("No road detected in the image.")
