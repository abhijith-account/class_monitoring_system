# -*- coding: utf-8 -*-
"""
Classroom Monitor System - Library Version
- Automatic YouTube Streaming enabled.
- Includes ThreadedCamera, GmailManager, DriveUploadManager, ClassroomMonitor.
- Updated to send reports to logged-in Faculty.
"""

import os
import sys
import subprocess
import signal

# --- RASPBERRY PI OPTIMIZATIONS ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
cv2.setNumThreads(4)

import time
import numpy as np
import pickle
import traceback
import threading
from queue import Queue, Empty, Full
from datetime import datetime
import pandas as pd
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# --- GOOGLE SERVICES IMPORTS ---
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload

# --- EMAIL IMPORTS ---
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# ================= CONFIGURATION =================
DEBUG_MODE = True

# --- YOUTUBE STREAM SETTINGS ---
HARDCODED_RTMP_URL = "rtmp://a.rtmp.youtube.com/live2/yhej-1eyu-2d2v-utgc-e87f"

# --- PATHS (Update these!) ---
SUSPECT_MODEL_PATH = r"/home/dsplab/Downloads/suspect.pt"
BEHAVIOR_MODEL_PATH = r"/home/dsplab/Downloads/best.pt"
DROWSINESS_MODEL_PATH = r"/home/dsplab/Downloads/drowsy.pt"

# --- SETTINGS ---
EVIDENCE_COOLDOWN_SECONDS = 15  
MOBILE_SUSPECT_DELAY_SECONDS = 3.0 
SLEEP_DELAY_SECONDS = 3.0            
OCCLUSION_TIMEOUT_SECONDS = 2.0 
CREDENTIALS_FILE = "gdrive5.json" 
TOKEN_PICKLE = "token.pickle"
GOOGLE_SCOPES = ['https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/gmail.send', 'https://www.googleapis.com/auth/youtube.force-ssl']
DRIVE_FOLDER_NAME = "Classroom_Evidence_Logs" 
REPORT_RECIPIENT_EMAIL = "cb.en.u4ece22163@cb.students.amrita.edu" # Default fallback

# --- THRESHOLDS ---
CLASS_NAMES = ['listening', 'hand', 'read', 'sleep', 'write']
BEHAVIOR_THRESHOLDS = {'hand': 0.75, 'read': 0.4, 'sleep': 0.6, 'write': 0.5, 'listening': 0.2}
SUSPECT_CLASS_THRESHOLDS = { 0: 0.35, 1: 0.35, 2: 0.5 }
SUSPECT_MIN_CONF = 0.3 
CELL_PHONE_CLASS_ID = 67
PERSON_CLASS_ID = 0
INSIGHTFACE_RECOGNITION_THRESHOLD = 0.35
FACE_REID_THRESHOLD = 0.5 
PROCESS_FRAME_INTERVAL = 2
MAX_ID_INACTIVITY_FRAMES = 150
ACTIVITY_OVERLAP_THRESHOLD = 0.2
MIN_TRACKING_IOU = 0.3

# Colors
COLOR_MAP = {
    'hand': (255, 255, 0), 'read': (0, 255, 0), 'sleep': (0, 0, 255),
    'write': (255, 0, 255), 'listening': (200, 200, 0), 
    'phone': (255, 165, 0), 'suspect': (0, 165, 255),
    'person_default': (0, 255, 0), 'person_unclaimed': (192, 192, 192),
    'drowsy': (0, 0, 255)
}

# ================= THREADED CAMERA CLASS =================
class ThreadedCamera:
    def __init__(self, src=0, use_pi_camera=False):
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.stream_queue = None
        self.status = False
        self.frame = None
        self.capture = None
        self.width = 640
        self.height = 480

        if use_pi_camera:
            print("[CAMERA] Attempting GStreamer pipeline for Pi Camera (libcamerasrc)...")
            gst_pipeline = "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink"
            self.capture = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        else:
            print(f"[CAMERA] Attempting standard V4L2 for Index {src}...")
            self.capture = cv2.VideoCapture(src, cv2.CAP_V4L2)
            if not self.capture.isOpened():
                self.capture = cv2.VideoCapture(src)

        if self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[CAMERA] Actual Resolution detected: {self.width}x{self.height}")

            self.status, self.frame = self.capture.read()
            if self.status: print("[CAMERA] Success! Camera opened.")
            else: print("[CAMERA] Camera opened but failed to read first frame.")
        else:
            print("[CAMERA] Failed to open camera.")

    def start(self, stream_queue):
        self.stream_queue = stream_queue
        if self.status:
            threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stop_event.is_set():
            if self.capture.isOpened():
                status, frame = self.capture.read()
                if status:
                    with self.lock:
                        self.status = status
                        self.frame = frame
                    if self.stream_queue is not None:
                        if not self.stream_queue.full():
                            self.stream_queue.put(frame)
                else: self.stop_event.set()
            else: time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.status, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.stop_event.set()
        if self.capture: self.capture.release()

# ================= SHARED AUTHENTICATION =================
def get_google_creds():
    creds = None
    if os.path.exists(TOKEN_PICKLE):
        with open(TOKEN_PICKLE, "rb") as token: creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try: creds.refresh(Request())
            except Exception: creds = None
        if not creds:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"[ERROR] '{CREDENTIALS_FILE}' not found. Google services disabled.")
                return None
            print("[AUTH] Initiating authentication flow...")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, GOOGLE_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PICKLE, "wb") as token: pickle.dump(creds, token)
    return creds

# ================= YOUTUBE STREAMING HELPERS =================
def setup_youtube_stream(creds):
    if HARDCODED_RTMP_URL: return HARDCODED_RTMP_URL
    try:
        print("[YOUTUBE] Searching for active broadcasts...")
        youtube = build("youtube", "v3", credentials=creds)
        active_broadcasts = youtube.liveBroadcasts().list(part="id,snippet,contentDetails", broadcastStatus="active", maxResults=1).execute()
        if active_broadcasts.get("items"):
            broadcast = active_broadcasts["items"][0]
            print(f"[YOUTUBE] Found Active Broadcast: {broadcast['snippet']['title']}")
            bound_stream_id = broadcast["contentDetails"].get("boundStreamId")
            if bound_stream_id:
                stream_details = youtube.liveStreams().list(part="cdn", id=bound_stream_id).execute()
                if stream_details.get("items"):
                    ingestion = stream_details["items"][0]["cdn"]["ingestionInfo"]
                    return f"{ingestion['ingestionAddress']}/{ingestion['streamName']}"
        print("[YOUTUBE] No active broadcast found. Go to YouTube Studio -> Go Live first.")
        return None
    except Exception as e:
        print(f"[YOUTUBE ERROR] Setup failed: {e}")
        return None

def ffmpeg_writer(process, frame_queue):
    while True:
        try:
            frame = frame_queue.get()
            if frame is None: break
            if process.poll() is not None:
                print("[FFMPEG WRITER] FFmpeg process has DIED. Stopping stream writer.")
                break
            process.stdin.write(frame.tobytes())
            frame_queue.task_done()
        except BrokenPipeError:
            print("[FFMPEG WRITER] Broken Pipe! FFmpeg closed the connection.")
            break
        except Exception as e:
            print(f"[FFMPEG WRITER] Error: {e}")
            break

def ffmpeg_stderr_reader(process):
    while True:
        line = process.stderr.readline()
        if not line: break
        try:
            # print(f"[FFMPEG LOG] {line.decode('utf-8').strip()}") # Uncomment for debugging
            pass
        except: pass

# ================= GMAIL & DRIVE MANAGERS =================
class GmailManager:
    def __init__(self, creds):
        self.service = None
        if creds:
            try: self.service = build('gmail', 'v1', credentials=creds)
            except Exception: pass
    def send_email_with_attachments(self, to_email, subject, body, file_paths):
        if not self.service: return
        try:
            message = MIMEMultipart()
            message['to'] = to_email; message['subject'] = subject
            message.attach(MIMEText(body))
            for file_path in file_paths:
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as fp:
                        msg = MIMEBase('application', 'octet-stream')
                        msg.set_payload(fp.read())
                    encoders.encode_base64(msg)
                    msg.add_header('Content-Disposition', 'attachment', filename=os.path.basename(file_path))
                    message.attach(msg)
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            self.service.users().messages().send(userId="me", body={'raw': raw}).execute()
        except Exception: pass

class DriveUploadManager:
    def __init__(self, creds):
        self.service = None
        self.upload_queue = Queue()
        self.stop_event = threading.Event()
        self.root_folder_id = None
        if creds:
            try:
                self.service = build('drive', 'v3', credentials=creds)
                self.root_folder_id = self._create_folder_if_not_exists(DRIVE_FOLDER_NAME)
                self.worker_thread = threading.Thread(target=self._upload_worker, daemon=True)
                self.worker_thread.start()
            except Exception: pass
    def _create_folder_if_not_exists(self, folder_name, parent_id=None):
        if not self.service: return None
        try:
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            query += f" and '{parent_id}' in parents" if parent_id else " and 'root' in parents"
            response = self.service.files().list(q=query, fields="files(id, name)").execute()
            if response.get('files'): return response.get('files')[0]['id']
            meta = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
            if parent_id: meta['parents'] = [parent_id]
            return self.service.files().create(body=meta, fields='id').execute().get('id')
        except Exception: return None
    def add_upload_task(self, local_path, student_name, violation_type):
        if self.service: self.upload_queue.put((local_path, student_name, violation_type))
    def _upload_worker(self):
        while not self.stop_event.is_set():
            try:
                task = self.upload_queue.get(timeout=1) 
                local_path, student_name, violation_type = task
                if not os.path.exists(local_path):
                    self.upload_queue.task_done(); continue
                student_id = self._create_folder_if_not_exists(student_name, self.root_folder_id)
                violation_map = {'Using Mobile': 'Phone_Evidence', 'Mobile Suspect': 'MS_Evidence', 'Sleeping': 'Sleep_Evidence'}
                final_id = self._create_folder_if_not_exists(violation_map.get(violation_type, 'Other'), student_id)
                media = MediaFileUpload(local_path, mimetype='image/jpeg', resumable=True)
                self.service.files().create(body={'name': os.path.basename(local_path), 'parents': [final_id]}, media_body=media).execute()
                print(f"[DRIVE] ✅ Uploaded: {os.path.basename(local_path)}")
                try: os.remove(local_path)
                except: pass
                self.upload_queue.task_done()
            except Empty: continue
            except Exception: self.upload_queue.task_done()
    def stop(self):
        self.stop_event.set()
        if hasattr(self, 'worker_thread'): self.worker_thread.join()

# ================= MAIN MONITORING CLASS =================
class ClassroomMonitor:
    def __init__(self, dataset_path='face_dataset'):
        print("Initializing Classroom Monitoring System...")
        self.evidence_root_dir = f"Evidence_Temp_{datetime.now().strftime('%Y%m%d')}"
        os.makedirs(self.evidence_root_dir, exist_ok=True)
        self.load_configs()
        self.creds = get_google_creds()
        self.drive_manager = DriveUploadManager(self.creds)
        self.gmail_manager = GmailManager(self.creds)

        print("Loading AI models...")
        self.behavior_model = YOLO(BEHAVIOR_MODEL_PATH, verbose=False)
        self.object_model = YOLO('yolov8n.pt', verbose=False) 
        self.drowsiness_model = YOLO(DROWSINESS_MODEL_PATH, verbose=False)
        self.suspect_model = YOLO(SUSPECT_MODEL_PATH)
        self.awake_classes = ["Awake", "awake_glass", "Normal", "Alert", "awaker"]
        self.drowsy_classes = ["Drowsiness", "Drowsiness -Glasses-", "Drowsiness -SunGlasses-", "Eyes closed", "drowsy"]
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self.face_app.models['detection'].det_thresh = 0.3
        
        self.known_face_encodings = np.empty((0, 512)); self.known_face_names = []
        self.load_or_create_known_faces()
        self.person_trackers = {}; self.suspect_timers = {}; self.sleep_timers = {}; self.evidence_log = {}    
        self.next_person_id = 0; self.frame_count = 0; self.prev_time = time.time()
        self.is_video_mode = True; self.session_start_time = None; self.session_end_time = None
        self.attendance_log = set(); self.distraction_log = {}
        self.headless_mode = False; self.last_calculated_fps = 0.0

    def load_configs(self):
        self.CLASS_NAMES = CLASS_NAMES; self.BEHAVIOR_THRESHOLDS = BEHAVIOR_THRESHOLDS
        self.SUSPECT_CLASS_THRESHOLDS = SUSPECT_CLASS_THRESHOLDS; self.SUSPECT_MIN_CONF = SUSPECT_MIN_CONF
        self.CELL_PHONE_CLASS_ID = CELL_PHONE_CLASS_ID; self.PERSON_CLASS_ID = PERSON_CLASS_ID
        self.INSIGHTFACE_RECOGNITION_THRESHOLD = INSIGHTFACE_RECOGNITION_THRESHOLD; self.FACE_REID_THRESHOLD = FACE_REID_THRESHOLD
        self.PROCESS_FRAME_INTERVAL = PROCESS_FRAME_INTERVAL; self.MAX_ID_INACTIVITY_FRAMES = MAX_ID_INACTIVITY_FRAMES
        self.ACTIVITY_OVERLAP_THRESHOLD = ACTIVITY_OVERLAP_THRESHOLD; self.MIN_TRACKING_IOU = MIN_TRACKING_IOU
        self.COLOR_MAP = COLOR_MAP; self.OCCLUSION_TIMEOUT_SECONDS = OCCLUSION_TIMEOUT_SECONDS 
        self.MOBILE_SUSPECT_DELAY_SECONDS = MOBILE_SUSPECT_DELAY_SECONDS; self.SLEEP_DELAY_SECONDS = SLEEP_DELAY_SECONDS

    def capture_evidence(self, frame, name, violation, box):
        current_time = time.time()
        key = (name, violation)
        if key in self.evidence_log:
            if current_time - self.evidence_log[key] < EVIDENCE_COOLDOWN_SECONDS: return 
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{name}_{violation}_{timestamp_str}.jpg"
        save_path = os.path.join(self.evidence_root_dir, filename)
        evidence_img = frame.copy()
        cv2.rectangle(evidence_img, (box[0], box[1]), (box[2], box[3]), self.COLOR_MAP.get('phone', (255,255,255)), 2)
        cv2.putText(evidence_img, f"{violation}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        try:
            cv2.imwrite(save_path, evidence_img)
            self.evidence_log[key] = current_time 
            print(f"📸 Captured: {filename}")
            self.drive_manager.add_upload_task(save_path, name, violation)
        except Exception: pass

    def load_or_create_known_faces(self, db_file='known_faces_db.pkl'):
        if os.path.exists(db_file):
            try:
                with open(db_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']; self.known_face_names = data['names']
                return
            except Exception: pass
        self.known_face_encodings = np.empty((0, 512)); self.known_face_names = []

    def add_face_from_image(self, image_path, name, db_file='known_faces_db.pkl'):
        if not os.path.exists(image_path): print("Image not found"); return
        img = cv2.imread(image_path)
        if img is None: return
        faces = self.face_app.get(img)
        if len(faces) != 1: print("Found 0 or >1 faces."); return
        self.known_face_encodings = np.vstack([self.known_face_encodings, faces[0].normed_embedding])
        self.known_face_names.append(name)
        with open(db_file, 'wb') as f: pickle.dump({'encodings': self.known_face_encodings, 'names': self.known_face_names}, f)
        print(f"Added {name}.")

    def preprocess_frame(self, frame):
        frame = np.ascontiguousarray(frame)
        if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4: frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def calculate_intersection_over_area(self, box_large, box_small):
        xA = max(box_large[0], box_small[0]); yA = max(box_large[1], box_small[1])
        xB = min(box_large[2], box_small[2]); yB = min(box_large[3], box_small[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        box_small_area = (box_small[2] - box_small[0]) * (box_small[3] - box_small[1])
        return interArea / float(box_small_area) if box_small_area > 0 else 0

    def non_max_suppression(self, boxes, scores, iou_threshold):
        if len(boxes) == 0: return []
        indices = np.argsort(scores)[::-1]; keep_indices = []
        while len(indices) > 0:
            current_index = indices[0]; keep_indices.append(current_index)
            remaining_indices = indices[1:]
            ious = np.array([self.calculate_iou(boxes[current_index], boxes[i]) for i in remaining_indices])
            indices = remaining_indices[ious < iou_threshold]
        return keep_indices
     
    def update_person_trackers(self, current_detections):
        updated_trackers = {}
        unmatched_det_indices = list(range(len(current_detections)))
        unmatched_tracker_ids = list(self.person_trackers.keys())
        if self.person_trackers and current_detections:
            matches = []
            for tracker_id, tracker in self.person_trackers.items():
                for det_idx, detection in enumerate(current_detections):
                    iou = self.calculate_iou(tracker['box'], detection['box'])
                    if iou >= self.MIN_TRACKING_IOU: matches.append((iou, tracker_id, det_idx))
            matches.sort(key=lambda x: x[0], reverse=True)
            for iou, tracker_id, det_idx in matches:
                if tracker_id in unmatched_tracker_ids and det_idx in unmatched_det_indices:
                    detection = current_detections[det_idx]; tracker = self.person_trackers[tracker_id]
                    detection['id'] = tracker_id; detection['name'] = tracker['name'] if detection.get('name', 'Unknown') == "Unknown" else detection['name']
                    detection['last_seen_frame'] = self.frame_count; detection['last_seen_time'] = time.time(); detection['is_occluded'] = False
                    if 'face_embedding' in tracker: detection['face_embedding'] = tracker['face_embedding']
                    updated_trackers[tracker_id] = detection
                    unmatched_tracker_ids.remove(tracker_id); unmatched_det_indices.remove(det_idx)
        lost_trackers = {tid: self.person_trackers[tid] for tid in unmatched_tracker_ids if self.person_trackers[tid].get('is_occluded')}
        reid_det_indices_to_remove = []
        for det_idx in unmatched_det_indices:
            detection = current_detections[det_idx]
            if 'face_embedding' not in detection: continue
            best_match_score = -1; best_match_id = -1
            for tracker_id, tracker in lost_trackers.items():
                if 'face_embedding' in tracker:
                    similarity = np.dot(detection['face_embedding'], tracker['face_embedding'])
                    if similarity > self.FACE_REID_THRESHOLD and similarity > best_match_score:
                        best_match_score = similarity; best_match_id = tracker_id
            if best_match_id != -1:
                tracker = self.person_trackers[best_match_id]
                detection['id'] = best_match_id; detection['name'] = tracker['name']
                detection['last_seen_frame'] = self.frame_count; detection['last_seen_time'] = time.time(); detection['is_occluded'] = False
                detection['face_embedding'] = tracker['face_embedding']
                updated_trackers[best_match_id] = detection
                unmatched_tracker_ids.remove(best_match_id); reid_det_indices_to_remove.append(det_idx); del lost_trackers[best_match_id]
        unmatched_det_indices = [i for i in unmatched_det_indices if i not in reid_det_indices_to_remove]
        for tracker_id in unmatched_tracker_ids:
            tracker = self.person_trackers[tracker_id]
            if time.time() - tracker.get('last_seen_time', time.time()) < self.OCCLUSION_TIMEOUT_SECONDS:
                tracker['is_occluded'] = True; tracker['activities'] = set(); updated_trackers[tracker_id] = tracker
        for det_idx in unmatched_det_indices:
            detection = current_detections[det_idx]
            new_id = self.next_person_id; self.next_person_id += 1
            detection['id'] = new_id; detection['last_seen_frame'] = self.frame_count; detection['last_seen_time'] = time.time(); detection['is_occluded'] = False
            updated_trackers[new_id] = detection
        self.person_trackers = updated_trackers

    def _determine_label_and_color(self, tracker_data):
        label_parts = [f"ID: {tracker_data['id']}", tracker_data.get('name', "Unknown")]
        if tracker_data.get('is_occluded', False) and self.frame_count - tracker_data.get('last_seen_frame', 0) > self.PROCESS_FRAME_INTERVAL: label_parts.append("Occluded")
        activities = tracker_data.get('activities', set())
        if activities and not tracker_data.get('is_occluded', False): label_parts.append(", ".join(sorted(activities)))
        final_label = " | ".join(label_parts)
        if "Using Mobile" in activities: color = self.COLOR_MAP['phone']
        elif "Mobile Suspect" in activities: color = self.COLOR_MAP['suspect']
        elif "Sleeping" in activities: color = self.COLOR_MAP['drowsy']
        elif "Writing" in activities: color = self.COLOR_MAP['write']
        elif "Reading" in activities: color = self.COLOR_MAP['read']
        elif "Hand Raising" in activities: color = self.COLOR_MAP['hand']
        elif "Listening" in activities: color = self.COLOR_MAP['listening']
        elif tracker_data.get('name') != "Unknown": color = self.COLOR_MAP['person_default']
        else: color = self.COLOR_MAP['person_unclaimed']
        return final_label, color

    def start_session(self):
        self.session_start_time = datetime.now(); self.attendance_log = set(); self.distraction_log = {}
        print(f"Monitoring session started at {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def end_session(self):
        self.session_end_time = datetime.now()
        print(f"Monitoring session ended at {self.session_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.drive_manager.stop() 

    def update_logs(self, delta_time):
        for _, tracker in self.person_trackers.items():
            name = tracker.get('name')
            if name and name != "Unknown":
                self.attendance_log.add(name); activities = tracker.get('activities', set()); tracker_id = tracker['id']
                distractions = {'Sleeping', 'Using Mobile', 'Mobile Suspect'}; found_distractions = distractions.intersection(activities)
                if name not in self.distraction_log: self.distraction_log[name] = {'Sleeping': 0, 'Using Mobile': 0, 'Mobile Suspect': 0}
                for distraction in found_distractions:
                    if distraction == 'Mobile Suspect':
                        if tracker_id in self.suspect_timers:
                            if time.time() - self.suspect_timers[tracker_id] < self.MOBILE_SUSPECT_DELAY_SECONDS: continue 
                    if distraction == 'Sleeping':
                        if tracker_id in self.sleep_timers:
                            if time.time() - self.sleep_timers[tracker_id] < self.SLEEP_DELAY_SECONDS: continue
                    if distraction in self.distraction_log[name]: self.distraction_log[name][distraction] += delta_time

    def _format_time(self, seconds): return f"{int(seconds // 60)} min {int(seconds % 60)} sec"

    # --- MODIFIED: ACCEPT RECIPIENT_EMAIL ARGUMENT ---
    def generate_reports(self, recipient_email=None):
        if not self.session_start_time or not self.session_end_time: return
        print("\nGenerating reports... Please wait.")
        
        # Use the passed email if available, otherwise use default
        target_email = recipient_email if recipient_email else REPORT_RECIPIENT_EMAIL
        
        timestamp = self.session_start_time.strftime("%Y-%m-%d_%H-%M-%S")
        duration = (self.session_end_time - self.session_start_time).total_seconds()
        info_df = pd.DataFrame({'Parameter': ['Start', 'End', 'Duration (s)'], 'Value': [self.session_start_time, self.session_end_time, round(duration, 2)]})
        files_to_email = []; all_students = sorted(self.known_face_names)
        if all_students:
            att_data = [{'Student': n, 'Status': 'Present' if n in self.attendance_log else 'Absent'} for n in all_students]
            att_file = f"Attendance_{timestamp}.xlsx"
            with pd.ExcelWriter(att_file, engine='openpyxl') as writer:
                info_df.to_excel(writer, sheet_name='Report', index=False)
                pd.DataFrame(att_data).to_excel(writer, sheet_name='Report', index=False, startrow=5)
            files_to_email.append(att_file)
        dist_data = []
        for name, d in self.distraction_log.items():
            if any(v > 0 for v in d.values()):
                dist_data.append({'Student': name, 'Mobile': self._format_time(d['Using Mobile']), 'Suspect': self._format_time(d['Mobile Suspect']), 'Sleep': self._format_time(d['Sleeping'])})
        if dist_data:
            dist_file = f"Distraction_{timestamp}.xlsx"
            with pd.ExcelWriter(dist_file, engine='openpyxl') as writer:
                info_df.to_excel(writer, sheet_name='Report', index=False)
                pd.DataFrame(dist_data).to_excel(writer, sheet_name='Report', index=False, startrow=5)
            files_to_email.append(dist_file)
        
        # Send to the dynamic target email
        if files_to_email:
            self.gmail_manager.send_email_with_attachments(target_email, f"Classroom Reports - {timestamp}", "Attached are the session reports.", files_to_email)
            print(f"Reports sent successfully to {target_email}.")

    def detect_behaviors(self, frame):
        self.frame_count += 1
        annotated_frame = frame.copy()
        if self.is_video_mode and self.frame_count % self.PROCESS_FRAME_INTERVAL != 0:
            if not self.headless_mode:
                for _, tracker in self.person_trackers.items():
                    label, color = self._determine_label_and_color(tracker)
                    x1, y1, x2, y2 = tracker['box']
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                fps_text = f"FPS: {self.last_calculated_fps:.1f} (Skipped)"
                cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
            return annotated_frame

        try:
            frame_prep = self.preprocess_frame(frame)
            person_res = self.object_model(frame_prep, classes=[self.PERSON_CLASS_ID], conf=0.6, verbose=False, iou=0.5, agnostic_nms=True, device='cpu')
            phone_res = self.object_model(frame_prep, classes=[self.CELL_PHONE_CLASS_ID], conf=0.25, verbose=False, iou=0.5, agnostic_nms=True, device='cpu')
            behav_res = self.behavior_model(frame_prep, conf=0.1, verbose=False, iou=0.5, agnostic_nms=True, device='cpu')
            drowsy_res = self.drowsiness_model(frame_prep, conf=0.4, verbose=False, iou=0.5, agnostic_nms=True, device='cpu')
            suspect_res = self.suspect_model(frame_prep, conf=self.SUSPECT_MIN_CONF, verbose=False, iou=0.5, agnostic_nms=True, device='cpu') 
            insight_faces = self.face_app.get(frame_prep)
            
            raw_boxes = [list(map(int, box.xyxy[0])) for r in person_res for box in r.boxes]
            raw_scores = [float(box.conf[0]) for r in person_res for box in r.boxes]
            yolo_dets = []
            if raw_boxes:
                keep = self.non_max_suppression(np.array(raw_boxes), np.array(raw_scores), 0.5)
                for i in keep: yolo_dets.append({'class': 'person', 'box': raw_boxes[i], 'name': "Unknown", 'activities': set(), 'last_seen_time': time.time()})

            yolo_others = []
            for r in phone_res:
                for box in r.boxes: yolo_others.append({'class': 'phone', 'box': list(map(int, box.xyxy[0]))})
            for r in suspect_res:
                for box in r.boxes:
                    if float(box.conf[0]) >= self.SUSPECT_CLASS_THRESHOLDS.get(int(box.cls[0]), 0.5):
                        yolo_others.append({'class': 'mobile_suspect', 'box': list(map(int, box.xyxy[0]))})
            for r in behav_res:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id < len(self.CLASS_NAMES) and float(box.conf[0]) >= self.BEHAVIOR_THRESHOLDS.get(self.CLASS_NAMES[cls_id], 0.5):
                        yolo_others.append({'class': self.CLASS_NAMES[cls_id], 'box': list(map(int, box.xyxy[0]))})
            
            for p in yolo_dets:
                best_face, max_iou = None, 0
                for face in insight_faces:
                    iou = self.calculate_iou(p['box'], list(map(int, face.bbox)))
                    if iou > max_iou: max_iou, best_face = iou, face
                if best_face:
                    p['face_embedding'] = best_face.normed_embedding
                    if self.known_face_encodings.size > 0:
                        sim = np.dot(self.known_face_encodings, best_face.normed_embedding)
                        if np.max(sim) > self.INSIGHTFACE_RECOGNITION_THRESHOLD: p['name'] = self.known_face_names[np.argmax(sim)]
                for r in drowsy_res:
                    for box in r.boxes:
                        if self.calculate_iou(p['box'], list(map(int, box.xyxy[0]))) > self.ACTIVITY_OVERLAP_THRESHOLD:
                            cls = self.drowsiness_model.names[int(box.cls[0])]
                            if cls in self.drowsy_classes: p['activities'].add('Sleeping')
                            elif cls in self.awake_classes: p['activities'].add('Listening')
            self.update_person_trackers(yolo_dets)
            
            for _, tracker in self.person_trackers.items():
                if tracker.get('is_occluded'): tracker['activities'] = set(); continue
                activities = tracker.get('activities', set()); tid = tracker['id']
                for b in yolo_others:
                    metric = self.calculate_intersection_over_area if b['class'] in ['phone', 'mobile_suspect'] else self.calculate_iou
                    if metric(tracker['box'], b['box']) > self.ACTIVITY_OVERLAP_THRESHOLD:
                        cls = b['class']
                        if cls == 'phone': activities.add('Using Mobile')
                        elif cls == 'mobile_suspect': activities.add('Mobile Suspect')
                        elif cls == 'hand': activities.add('Hand Raising')
                        elif cls == 'read': activities.add('Reading')
                        elif cls == 'write': activities.add('Writing')
                        elif cls == 'sleep': activities.add('Sleeping')
                        elif cls == 'listening': activities.add('Listening')
                        else: activities.add(cls.capitalize())
                if 'Mobile Suspect' in activities:
                    if tid not in self.suspect_timers: self.suspect_timers[tid] = time.time()
                elif tid in self.suspect_timers: del self.suspect_timers[tid]
                if 'Sleeping' in activities:
                    if tid not in self.sleep_timers: self.sleep_timers[tid] = time.time()
                elif tid in self.sleep_timers: del self.sleep_timers[tid]

                primary = activities.intersection({'Writing', 'Reading', 'Hand Raising', 'Sleeping', 'Using Mobile', 'Mobile Suspect'})
                final = set()
                if 'Using Mobile' in primary: 
                    final = {'Using Mobile'}
                    if self.is_video_mode: self.capture_evidence(frame, tracker.get('name', 'Unknown'), 'Using Mobile', tracker['box'])
                elif 'Mobile Suspect' in primary: 
                    final = {'Mobile Suspect'} 
                    if tid in self.suspect_timers and (time.time() - self.suspect_timers[tid] >= self.MOBILE_SUSPECT_DELAY_SECONDS):
                        if self.is_video_mode: self.capture_evidence(frame, tracker.get('name', 'Unknown'), 'Mobile Suspect', tracker['box'])
                elif 'Sleeping' in primary: 
                    final = {'Sleeping'}
                    if tid in self.sleep_timers and (time.time() - self.sleep_timers[tid] >= self.SLEEP_DELAY_SECONDS):
                        if self.is_video_mode: self.capture_evidence(frame, tracker.get('name', 'Unknown'), 'Sleeping', tracker['box'])
                elif primary: final = primary
                elif 'Listening' in activities: final = {'Listening'}
                else: final = {'Listening'} if self.is_video_mode else set()
                tracker['activities'] = final
            
            if not self.headless_mode:
                for _, tracker in self.person_trackers.items():
                    label, color = self._determine_label_and_color(tracker)
                    x1, y1, x2, y2 = tracker['box']
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    for b in yolo_others:
                        if b['class'] in ['mobile_suspect', 'phone']:
                            if self.calculate_intersection_over_area(tracker['box'], b['box']) > self.ACTIVITY_OVERLAP_THRESHOLD:
                                bx1, by1, bx2, by2 = b['box']
                                b_col = self.COLOR_MAP['suspect'] if b['class'] == 'mobile_suspect' else self.COLOR_MAP['phone']
                                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), b_col, 2)
                
                curr_time = time.time(); delta = curr_time - self.prev_time
                if delta > 0: self.last_calculated_fps = 1.0 / delta
                fps_text = f"FPS: {self.last_calculated_fps:.1f} (Process)"
                cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
            
            curr_time = time.time(); delta = curr_time - self.prev_time
            if self.is_video_mode: self.update_logs(delta)
            self.prev_time = curr_time
            return annotated_frame
        except Exception: traceback.print_exc(); return frame

# ================= MAIN EXECUTION =================
running = True
def signal_handler(sig, frame): global running; print("\n[INFO] Exiting requested..."); running = False
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    try:
        monitor = ClassroomMonitor()
        while running:
            print("\n--- Classroom Monitor Menu ---\n1. Run from Camera\n2. Add New Face\n3. Exit")
            choice = input("Enter your choice: ")
            if choice == '1':
                monitor.is_video_mode = True
                
                # --- AUTOMATICALLY ENABLE YOUTUBE STREAMING ---
                use_youtube = True 
                
                cam_index = 0; use_pi_cam = False; rtmp_url = None; ffmpeg_process = None; frame_queue = None
                
                # --- STEP 1: OPEN CAMERA FIRST ---
                camera = ThreadedCamera(src=cam_index, use_pi_camera=use_pi_cam)
                if not camera.status:
                    print("[ERROR] Could not open camera. Exiting to menu..."); camera.release(); continue

                # --- STEP 2: SETUP FFmpeg ---
                if use_youtube:
                    rtmp_url = setup_youtube_stream(monitor.creds)
                    if rtmp_url:
                        print(f"[YOUTUBE] Streaming to: {rtmp_url}")
                        print(f"[YOUTUBE] Using Input Resolution: {camera.width}x{camera.height}")
                        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{camera.width}x{camera.height}', '-r', '30', '-i', '-', '-f', 'lavfi', '-i', 'anullsrc', '-c:a', 'aac', '-ar', '44100', '-b:a', '128k', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast', '-g', '60', '-b:v', '2000k', '-maxrate', '2000k', '-bufsize', '4000k', '-f', 'flv', rtmp_url]
                        ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                        frame_queue = Queue(maxsize=30)
                        threading.Thread(target=ffmpeg_writer, args=(ffmpeg_process, frame_queue), daemon=True).start()
                        threading.Thread(target=ffmpeg_stderr_reader, args=(ffmpeg_process,), daemon=True).start()
                    else:
                        print("[YOUTUBE ERROR] Could not get valid RTMP URL. Streaming DISABLED.")
                
                # --- STEP 3: START CAMERA STREAMING ---
                camera.start(frame_queue)
                monitor.start_session()
                session_running = True 
                
                try:
                    while session_running and running:
                        status, frame = camera.read()
                        if not status or frame is None: break
                        processed = monitor.detect_behaviors(frame)
                        
                        if not monitor.headless_mode:
                            try:
                                cv2.imshow("Classroom Monitor", processed)
                                if cv2.waitKey(1) & 0xFF == ord('q'): 
                                    print("\n[INFO] 'q' pressed. Stopping session...")
                                    session_running = False 
                            except cv2.error:
                                print("\n[WARNING] OpenCV Display failed. Switching to HEADLESS MODE.")
                                monitor.headless_mode = True
                except KeyboardInterrupt: 
                    print("\n[INFO] Stopped by User.")
                    session_running = False
                finally:
                    camera.release()
                    try:
                        cv2.destroyAllWindows()
                        cv2.waitKey(1) 
                    except Exception: pass
                    
                    monitor.end_session()
                    monitor.generate_reports()
                    
                    if frame_queue: frame_queue.put(None)
                    if ffmpeg_process: ffmpeg_process.stdin.close(); ffmpeg_process.terminate()
            
            elif choice == '2':
                img_path = input("Enter path to image for new face: ").strip().strip('"').strip("'")
                if os.path.exists(img_path):
                    name = input("Enter name: ")
                    if name: monitor.add_face_from_image(img_path, name)
            elif choice == '3': break
            else: print("Invalid choice.")
    except Exception as e: print(f"Fatal error: {str(e)}"); traceback.print_exc()