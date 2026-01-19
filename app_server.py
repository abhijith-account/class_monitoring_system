from flask import Flask, Response, jsonify, render_template_string, request, url_for
import threading
import time
import subprocess
import os
import json
import pandas as pd
import signal
import sys
import cv2
import numpy as np
from queue import Queue, Empty, Full

# --- IMPORTS WITH FALLBACK ---
try:
    from classroom_monitor import ClassroomMonitor, ThreadedCamera, setup_youtube_stream, ffmpeg_stderr_reader
except ImportError:
    print("Warning: classroom_monitor module not found. Ensure the file exists.")
    # Dummy classes for testing
    class ClassroomMonitor:
        headless_mode = False
        is_video_mode = False
        creds = None
        def start_session(self): pass
        def end_session(self): pass
        def detect_behaviors(self, frame): pass
        def update_logs(self, t): pass
        def generate_reports(self, recipient_email=None): 
            print(f"Generating reports for {recipient_email}")

    class ThreadedCamera:
        status = True
        def __init__(self, src=0, use_pi_camera=False): pass
        def start(self, cb): pass
        def read(self): return True, None
        def release(self): pass
    def setup_youtube_stream(creds): return None
    def ffmpeg_stderr_reader(proc): pass

app = Flask(__name__)

# --- GLOBAL VARIABLES ---
monitor = None
camera = None
is_running = False
ffmpeg_process = None
frame_queue = Queue(maxsize=5)
monitor_thread = None
latest_frame = None  
ai_lock = threading.Lock()
ai_running = False
current_faculty_email = None 

# --- CONFIGURATION ---
YOUTUBE_STREAM_KEY = "yhej-1eyu-2d2v-utgc-e87f"
CREDENTIALS_FILE = "users.xlsx"
TARGET_FPS = 25 

# --- HTML TEMPLATE (OPTIMIZED FOR MOBILE) ---
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Classroom Monitor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        body { 
            background-color: #000; 
            margin: 0; 
            padding: 0; 
            width: 100vw;
            height: 100vh;
            display: flex; 
            justify-content: center; 
            align-items: center; 
            overflow: hidden;
        }
        .feed-box { 
            width: 100%; 
            height: 100%;
            display: flex; 
            justify-content: center;
            align-items: center;
        }
        img { 
            max-width: 100%; 
            max-height: 100%;
            object-fit: contain;
            display: block;
        }
    </style>
</head>
<body>
    <div class="feed-box">
        <img id="cam_feed" alt="Waiting for stream..." />
    </div>

    <script>
        // Force reload of image source to bypass Android WebView caching
        // This is critical for the stream to appear again after stopping/starting
        window.onload = function() {
            var timestamp = new Date().getTime();
            document.getElementById("cam_feed").src = "stream_video?t=" + timestamp;
        };
    </script>
</body>
</html>
"""

# --- ROBUST CLEANUP LOGIC ---
def cleanup_resources():
    global is_running, ai_running, ffmpeg_process, camera, monitor
    print("\n[SYSTEM] Performing Cleanup...")
    
    # 1. Stop the loops first
    is_running = False
    ai_running = False
    
    # 2. Correctly terminate FFmpeg to tell YouTube "Stream Finished"
    if ffmpeg_process:
        print("[SYSTEM] Stopping FFmpeg...")
        try:
            # A. Close the input. This sends "End of File" to FFmpeg.
            # YouTube needs this to know the stream ended gracefully.
            if ffmpeg_process.stdin:
                ffmpeg_process.stdin.close()
            
            # B. Wait a moment for FFmpeg to finish flushing data
            time.sleep(1.0) 
            
            # C. Terminate nicely
            ffmpeg_process.terminate()
            ffmpeg_process.wait(timeout=2)
        except (subprocess.TimeoutExpired, Exception):
            # D. If stuck, Force Kill
            print("[SYSTEM] FFmpeg stuck. Force killing...")
            try: ffmpeg_process.kill()
            except: pass
        
        ffmpeg_process = None 

    # 3. Release Camera
    if camera:
        print("[SYSTEM] Releasing Camera...")
        try: camera.release()
        except: pass
        camera = None

    # 4. Stop AI Monitor
    if monitor:
        try: monitor.end_session()
        except: pass

def signal_handler(sig, frame):
    cleanup_resources()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --- 1. YOUTUBE PUSH THREAD ---
def push_frames_to_ffmpeg(process, queue):
    last_frame_bytes = None
    while is_running:
        if process.poll() is not None: break 
        try:
            # Non-blocking get with timeout
            try:
                frame = queue.get(timeout=0.04)
                if frame is not None:
                    last_frame_bytes = frame.tobytes()
            except Empty:
                if not last_frame_bytes: continue

            if last_frame_bytes:
                try:
                    if process.stdin.closed: break
                    process.stdin.write(last_frame_bytes)
                    process.stdin.flush() 
                except (BrokenPipeError, ValueError, OSError):
                    break
        except Exception:
            break

# --- 2. MOBILE MJPEG GENERATOR ---
def generate_frames():
    global latest_frame, ai_lock, is_running
    
    # 1. Create a "Loading" frame
    # We yield this IMMEDIATELY so the browser/app gets a valid response header.
    # If we wait for the camera, the mobile app will timeout and show a broken image.
    blank_image = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(blank_image, "STARTING STREAM...", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, blank_encoded = cv2.imencode('.jpg', blank_image)
    blank_bytes = blank_encoded.tobytes()

    # YIELD IMMEDIATELY (Fix for Mobile "Not Appearing")
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')

    while True:
        if not is_running:
            # If stopped, send one offline frame and wait (keeps connection open but idle)
            cv2.putText(blank_image, "OFFLINE", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, off_encoded = cv2.imencode('.jpg', blank_image)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + off_encoded.tobytes() + b'\r\n')
            time.sleep(1.0)
            continue

        frame_to_encode = None
        with ai_lock:
            if latest_frame is not None:
                try:
                    # Resize for mobile optimization (640x480 is standard/safe)
                    frame_to_encode = cv2.resize(latest_frame, (640, 480))
                except Exception:
                    pass
        
        output_bytes = blank_bytes
        if frame_to_encode is not None:
            # Encode Quality 50% - Best balance for Mobile Data
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            ret, buffer = cv2.imencode('.jpg', frame_to_encode, encode_param)
            if ret:
                output_bytes = buffer.tobytes()
        
        # Yield the actual frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + output_bytes + b'\r\n')
        
        time.sleep(0.04) # ~25 FPS cap

# --- 3. AI THREAD ---
def ai_worker_task():
    global latest_frame, ai_running, monitor
    prev_time = time.time()
    while ai_running and is_running:
        frame_to_process = None
        with ai_lock:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()
        
        if frame_to_process is not None:
            monitor.detect_behaviors(frame_to_process)
            monitor.update_logs(time.time() - prev_time)
            prev_time = time.time()
        else:
            time.sleep(0.1)

# --- 4. MAIN TASK ---
def background_monitor_task():
    global monitor, camera, is_running, ffmpeg_process, frame_queue, latest_frame, ai_running
    
    print("[SYSTEM] Initializing Backend...")
    monitor = ClassroomMonitor()
    monitor.headless_mode = True
    monitor.is_video_mode = True

    camera = ThreadedCamera(src=0, use_pi_camera=False)
    
    start_cam_time = time.time()
    while not camera.status: 
        if time.time() - start_cam_time > 10: 
            print("Camera Timeout")
            return
        time.sleep(0.5)
        
    capture_queue = Queue(maxsize=1) 
    camera.start(capture_queue) 
    
    # Wait for first frame before proceeding
    try:
        initial_frame = capture_queue.get(timeout=5)
    except Empty: return

    height, width = initial_frame.shape[:2]
    # Ensure dimensions are even (required by FFmpeg)
    if width % 2 != 0: width -= 1
    if height % 2 != 0: height -= 1
    
    monitor.start_session()
    ai_running = True
    threading.Thread(target=ai_worker_task, daemon=True).start()

    # YouTube Stream Setup
    rtmp_url = None
    retry_count = 0
    while is_running and rtmp_url is None:
        rtmp_url = setup_youtube_stream(monitor.creds) 
        if not rtmp_url:
            rtmp_url = "rtmp://a.rtmp.youtube.com/live2/" + YOUTUBE_STREAM_KEY
        if rtmp_url: break
        time.sleep(5)
        retry_count += 1

    if is_running and rtmp_url:
        print(f"[YOUTUBE] Streaming to: {rtmp_url}")
        cmd = [
            'ffmpeg', '-y', '-re', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(TARGET_FPS),
            '-thread_queue_size', '4096', '-i', '-', '-f', 'lavfi',
            '-thread_queue_size', '4096', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
            '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast', '-g', str(TARGET_FPS * 2), '-sc_threshold', '0',
            '-b:v', '1500k', '-maxrate', '2000k', '-bufsize', '4000k',
            '-flvflags', 'no_duration_filesize', 
            '-c:a', 'aac', '-b:a', '128k', '-ar', '44100', '-f', 'flv', rtmp_url
        ]
        ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=push_frames_to_ffmpeg, args=(ffmpeg_process, frame_queue), daemon=True).start()
        threading.Thread(target=ffmpeg_stderr_reader, args=(ffmpeg_process,), daemon=True).start()
    
    frame_duration = 1.0 / TARGET_FPS
    next_frame_time = time.time()
    
    print("[SYSTEM] Streams Active.")

    while is_running:
        try:
            frame = capture_queue.get(timeout=0.1)
            
            with ai_lock: latest_frame = frame 
            
            now = time.time()
            if now >= next_frame_time:
                try: frame_queue.put(frame, block=False)
                except Full: pass 
                next_frame_time = now + frame_duration
        except: continue
    
    print("[SYSTEM] Loop Ended.")

# --- ROUTES ---

@app.route('/')
def index():
    return "Server Running. Use App."

@app.route('/stream_video')
def stream_video():
    # MJPEG Standard
    response = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    # Headers to prevent caching and allow cross-origin
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/login', methods=['POST'])
def login():
    global current_faculty_email 
    try:
        # Robust JSON parsing for App Inventor
        try:
            data = request.get_json(force=True, silent=True)
            if not data: data = json.loads(request.data.decode('utf-8'))
        except: return "error_parsing_data"

        username = data.get("user")
        password = data.get("pass")

        if not os.path.exists(CREDENTIALS_FILE): return "error_no_database"

        try:
            df = pd.read_excel(CREDENTIALS_FILE)
            df.columns = [c.strip().lower() for c in df.columns]
            user_row = df[df['username'].astype(str) == username]
            
            if not user_row.empty:
                # Ensure password comparison is string-to-string
                if str(user_row.iloc[0]['password']) == str(password):
                    current_faculty_email = username
                    return "success"
            return "invalid_credentials"
        except: return "error_reading_database"
    except: return "error"

@app.route('/video_feed')
def video_feed():
    return render_template_string(HTML_PAGE)

@app.route('/start')
def start():
    global is_running, monitor_thread
    if is_running: return jsonify({"status": "Error", "message": "Already running!"})
    
    # Clear queues to prevent old frames from showing
    while not frame_queue.empty():
        try: frame_queue.get_nowait()
        except: pass

    is_running = True
    monitor_thread = threading.Thread(target=background_monitor_task, daemon=True)
    monitor_thread.start()
    return jsonify({"status": "Started", "message": "Streaming Started"})

@app.route('/stop')
def stop():
    global current_faculty_email
    cleanup_resources()
    if monitor:
        if current_faculty_email: monitor.generate_reports(recipient_email=current_faculty_email)
        else: monitor.generate_reports() 
    return jsonify({"status": "Stopped", "message": "Streaming Stopped"})

if __name__ == '__main__':
    # Threaded=True is ESSENTIAL for simultaneous stream + app control
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)