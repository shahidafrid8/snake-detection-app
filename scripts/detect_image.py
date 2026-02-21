import cv2
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from detect import detect

def draw_boxes_from_api(img, predictions):
    """
    Draw bounding boxes on the image from Roboflow API predictions.
    
    Args:
        img: The image array.
        predictions: The JSON predictions dict from Roboflow API.
    
    Returns:
        The image with boxes drawn.
    """
    if 'predictions' not in predictions:
        return img
    
    for pred in predictions['predictions']:
        x_center = pred['x']
        y_center = pred['y']
        width = pred['width']
        height = pred['height']
        confidence = pred['confidence']
        class_name = pred['class']
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img

def main():
    image_path = input("Please enter the path to the image file: ").strip()

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    # Run detection with API fallback to local
    predictions, method = detect(image_path)

    if method == "FAILED":
        print("Detection failed with both API and local model.")
        return

    print(f"Detection completed using {method}.")

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from '{image_path}'.")
        return

    # Annotate image based on method
    if method == "API":
        annotated_img = draw_boxes_from_api(img, predictions)
    elif method == "LOCAL":
        annotated_img = predictions[0].plot()

    # Show results
    output_path = image_path.replace('.png', '_detected.png')
    cv2.imwrite(output_path, annotated_img)
    print(f"Annotated image saved to {output_path}")

if __name__ == "__main__":
    main()