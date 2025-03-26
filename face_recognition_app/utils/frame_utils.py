import time

import cv2
import face_recognition
import numpy as np
from django.http import JsonResponse
from rest_framework import status


def gen_frames(known_encoding):
    """Generate frames with face recognition"""
    video_capture = cv2.VideoCapture(0)
    known_name = "Matching"

    if not video_capture.isOpened():
        yield JsonResponse(
            {'error': 'Could not open webcam'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        return

    process_this_frame = True
    match_detected = False
    match_start_time = None
    match_confirmed = False

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Only process every other frame to save time
        if process_this_frame and not match_confirmed:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            current_frame_match = False

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces([known_encoding], face_encoding,0.4)
                name = "Unknown"

                face_distances = face_recognition.face_distance([known_encoding], face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_name
                    current_frame_match = True

                face_names.append(name)

            # Handle match detection timing
            if current_frame_match:
                if not match_detected:
                    match_detected = True
                    match_start_time = time.time()
                else:
                    # Check if we've had a match for 3 seconds
                    if time.time() - match_start_time >= 3:
                        match_confirmed = True
            else:
                match_detected = False
                match_start_time = None

        process_this_frame = not process_this_frame

        # Draw results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Choose color based on recognition
            if name == known_name:
                box_color = (0, 255, 0)  # Green
                text_color = (0, 0, 255)  # Red text
            else:
                box_color = (0, 0, 255)  # Red
                text_color = (255, 255, 255)  # White text

            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, text_color, 1)

        # Add match confirmation message
        if match_confirmed:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "MATCH CONFIRMED!", (50, 50), font, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "Redirecting...", (50, 100), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # After sending the confirmation frame, we can break the loop
            video_capture.release()
            return

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()

