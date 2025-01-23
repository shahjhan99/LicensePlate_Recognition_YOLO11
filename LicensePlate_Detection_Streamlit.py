import cv2
import streamlit as st
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import tempfile
import os
import easyocr  # Import EasyOCR

# Initialize YOLO model
model_path = "best.pt"
model = YOLO(model_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # You can add other languages if needed

# Function to process video
def process_video(video_file):
    cap = cv2.VideoCapture(video_file)
    mini_screen_scale = 0.4
    results_data = []

    excel_file = "Licence_Plate_detected_results.xlsx"
    df = pd.DataFrame(columns=["Frame", "BoundingBox", "Detected_Text", "TimeStamp"])
    df.to_excel(excel_file, index=False)

    frame_count = 0
    # Streamlit container for video frames
    video_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for detection in results[0].boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())

            # Zoom in on the bounding box area by cropping and resizing it
            bounding_box_region = frame[y1:y2, x1:x2]
            zoomed_bounding_box = cv2.resize(bounding_box_region, None, fx=2, fy=2)  # Increase the size by a factor of 2

            # Run OCR on the zoomed region
            ocr_results = reader.readtext(zoomed_bounding_box)
            detected_text = " ".join([text[1] for text in ocr_results])  # Extract and join detected text

            # Get the coordinates to place the zoomed region above the bounding box
            zoomed_height, zoomed_width = zoomed_bounding_box.shape[:2]
            upper_y_position = max(0, y1 - zoomed_height - 20)  # Place it above the bounding box with a gap

            # Add padding for the zoomed area to create a border
            border_thickness = 10
            zoomed_bounding_box_with_border = cv2.copyMakeBorder(
                zoomed_bounding_box,
                top=border_thickness,
                bottom=border_thickness,
                left=border_thickness,
                right=border_thickness,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 255]  # Red border
            )

            # Ensure the zoomed bounding box with border fits within the frame
            zoomed_height_with_border, zoomed_width_with_border = zoomed_bounding_box_with_border.shape[:2]
            if x1 + zoomed_width_with_border > frame.shape[1]:
                zoomed_bounding_box_with_border = cv2.resize(
                    zoomed_bounding_box_with_border,
                    (frame.shape[1] - x1, zoomed_height_with_border)
                )

            # Check if the zoomed bounding box with border fits vertically
            if upper_y_position >= 0:
                try:
                    frame[upper_y_position:upper_y_position + zoomed_height_with_border, x1:x1 + zoomed_width_with_border] = zoomed_bounding_box_with_border
                except ValueError:
                    pass

            # Draw the bounding box
            font_scale = 1
            font_thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the text above the zoomed region with black background and white text
            text_position = (x1, max(10, upper_y_position - 30))  # Position for the text
            (text_width, text_height), _ = cv2.getTextSize(detected_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Background rectangle for text
            text_background_start = (text_position[0], text_position[1] - text_height - 5)
            text_background_end = (text_position[0] + text_width + 10, text_position[1] + 5)
            cv2.rectangle(frame, text_background_start, text_background_end, (0, 0, 0), -1)  # Black rectangle

            # Overlay the detected text
            cv2.putText(frame, detected_text, (text_position[0] + 5, text_position[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

            # Save the detection data for Excel
            results_data.append({"Frame": frame_count,
                                 "BoundingBox": (x1, y1, x2, y2),
                                 "Detected_Text": detected_text,
                                 "TimeStamp": current_time
                                 })

        # Save results to Excel after every frame (or after a batch of frames)
        if results_data:
            temp_df = pd.DataFrame(results_data)
            with pd.ExcelWriter(excel_file, mode='a', if_sheet_exists='overlay') as writer:
                temp_df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)

        # Resize frame for better visualization
        mini_frame = cv2.resize(frame, None, fx=mini_screen_scale, fy=mini_screen_scale)

        # Display the processed frame in Streamlit (just the detection video)
        video_placeholder.image(mini_frame, channels="BGR", caption="License Plates Detection (Zoomed Bounding Box Above)")

        frame_count += 1

    cap.release()
    print("Results Saved to Excel File")
    return excel_file


# Streamlit UI
def main():
    st.title("License Plate Detection with YOLO and OCR")

    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])
     # Add the credit in the sidebar
    st.sidebar.markdown("### Project developed by Shahjhan99 \n Mail:  shahjhangondal99@gamil.com")

    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_video_path = tmp_file.name
        
        # Button to start detection
        if st.button("Start Detection"):
            st.write("Starting detection...")

            # Process the video for license plate detection and OCR
            excel_file = process_video(tmp_video_path)
            st.write(f"Results saved to {excel_file}")

            # Provide an option to download the results
            with open(excel_file, "rb") as f:
                st.download_button("Download Detection Results (Excel)", f, file_name="Licence_Plate_detected_results.xlsx")

            # Clean up temporary video file
            os.remove(tmp_video_path)


if __name__ == "__main__":
    main()
