import argparse
import cv2
import os
import torch
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def draw_boxes(image, boxes, labels, confidences, color=(255, 0, 0)):
    for box, label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {conf:.2f}"
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_image(image_path, person_model, ppe_model, output_dir, confidence_threshold=0.5):
    try:
        logger.info(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return

        img_height, img_width = image.shape[:2]

        # Person detection
        person_results = person_model(image)[0]
        person_boxes = person_results.boxes.xyxy.cpu().numpy()
        person_confs = person_results.boxes.conf.cpu().numpy()

        # Filter person detections by confidence threshold
        person_boxes = [box for box, conf in zip(person_boxes, person_confs) if conf >= confidence_threshold]
        person_confs = [conf for conf in person_confs if conf >= confidence_threshold]

        # Iterate over detected persons
        for person_box in person_boxes:
            x1, y1, x2, y2 = map(int, person_box)

            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)

            # Crop the person image
            cropped_image = image[y1:y2, x1:x2]

            if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                # PPE detection on the cropped image
                ppe_results = ppe_model(cropped_image)[0]
                ppe_boxes = ppe_results.boxes.xyxy.cpu().numpy()
                ppe_labels = [ppe_model.names[int(cls)] for cls in ppe_results.boxes.cls.cpu().numpy()]
                ppe_confs = ppe_results.boxes.conf.cpu().numpy()

                # Filter PPE detections by confidence threshold
                ppe_boxes = [box for box, conf in zip(ppe_boxes, ppe_confs) if conf >= confidence_threshold]
                ppe_labels = [label for label, conf in zip(ppe_labels, ppe_confs) if conf >= confidence_threshold]
                ppe_confs = [conf for conf in ppe_confs if conf >= confidence_threshold]

                # Adjust PPE box coordinates relative to the original image
                ppe_boxes = [[box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1] for box in ppe_boxes]

                # Draw PPE boxes on the original image
                draw_boxes(image, ppe_boxes, ppe_labels, ppe_confs, color=(0, 255, 0))

        # Save the output image
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)
        logger.info(f"Saved processed image: {output_path}")
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Perform inference using person and PPE detection models')
    parser.add_argument('input_dir', help='Path to the input directory containing images')
    parser.add_argument('output_dir', help='Path to the output directory for annotated images')
    parser.add_argument('person_det_model', help='Path to the person detection model weights')
    parser.add_argument('ppe_detection_model', help='Path to the PPE detection model weights')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold for detections')
    args = parser.parse_args()

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load models
    person_model = YOLO(args.person_det_model).to(device)
    ppe_model = YOLO(args.ppe_detection_model).to(device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)

    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(args.input_dir, image_file)
        logger.info(f"Processing image {i}/{total_images}: {image_file}")
        process_image(image_path, person_model, ppe_model, args.output_dir, args.confidence_threshold)

    logger.info(f"Inference completed. Annotated images saved in {args.output_dir}")

if __name__ == '__main__':
    main()