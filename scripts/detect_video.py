# scripts/detect_video.py
# Video detection script for snake detection
# Processes video frame by frame and creates annotated output

import cv2
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from detect import detect
import tempfile

def detect_video(video_path, progress_callback=None):
    """
    Detect snakes in a video file.
    
    Args:
        video_path (str): Path to input video file
        progress_callback (callable): Optional callback for progress updates
        
    Returns:
        tuple: (output_path, method, error_msg, stats)
    """
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, "Failed to open video file", None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video file
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "detected_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_count = 0
        method_used = None
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Save frame temporarily
            temp_frame_path = os.path.join(temp_dir, "temp_frame.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            # Detect on frame (every 5th frame to speed up processing)
            if frame_count % 5 == 0 or frame_count == 1:
                predictions, method, error_msg = detect(temp_frame_path)
                
                if error_msg:
                    cap.release()
                    out.release()
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
                    return None, method, error_msg, None
                
                method_used = method
                
                # Draw detections
                if method == "API":
                    if 'predictions' in predictions and predictions['predictions']:
                        detection_count += len(predictions['predictions'])
                        for pred in predictions['predictions']:
                            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                            x1 = int(x - w/2)
                            y1 = int(y - h/2)
                            x2 = int(x + w/2)
                            y2 = int(y + h/2)
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{pred['class']} {pred['confidence']:.2f}"
                            cv2.putText(frame, label, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                elif method == "LOCAL":
                    if len(predictions[0].boxes) > 0:
                        detection_count += len(predictions[0].boxes)
                        frame = predictions[0].plot()
            
            # Write frame to output
            out.write(frame)
            
            # Progress callback
            if progress_callback:
                progress_callback(frame_count, total_frames)
            
            # Clean up temp frame
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
        
        # Release resources
        cap.release()
        out.release()
        
        stats = {
            'total_frames': total_frames,
            'processed_frames': frame_count,
            'detections': detection_count,
            'fps': fps
        }
        
        return output_path, method_used, None, stats
        
    except Exception as e:
        return None, None, f"Video processing error: {str(e)}", None


def main():
    """Command line interface for video detection"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python detect_video.py <video_path>")
        return
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    print(f"Starting snake detection on video: {video_path}")
    
    def progress(current, total):
        percent = (current / total) * 100
        print(f"\rProgress: {current}/{total} frames ({percent:.1f}%)", end='')
    
    output_path, method, error_msg, stats = detect_video(video_path, progress)
    
    if error_msg:
        print(f"\n\nError: {error_msg}")
        return
    
    print(f"\n\nDetection completed using {method}")
    print(f"Total detections: {stats['detections']}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
