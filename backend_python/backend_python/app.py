import json
import uuid
import requests
from datetime import datetime
from threading import Thread
from flask import Flask, request, send_file#, url_for
from werkzeug.utils import secure_filename
#from flask_api import FlaskAPI, status, exceptions

# video 
from Models.VideoProcessing.VideoEditor import VideoEditor

# ai
from Models.AI.model_emotion import model_emotion
from Models.AI.model_suicide_detection import model_P

UPLOAD_FOLDER = './UploadDIR'
ALLOWED_EXTENSIONS = {'mp4', 'png', 'jpg', 'jpeg', 'gif'}
WEB_SERVER = "http://127.0.0.1:5000" # "http://192.168.1.2:8000"

# ----------------------------------------------

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 # default no limit
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return "hello there !!\nthis server to process videos\nyou need to see API documentation"

@app.route('/test/video_done/', methods=['GET', 'POST'])
def test_done():
    
    req = request.json
    if req is None or len(req) == 0:
        return "empty request. be sure that request sended as JSON"
        
    req_pretty = json.dumps(req, indent = 4)
    print(req_pretty)
    
    res = dict()
    res['status'] = 'success'    
    return json.dumps(res, indent = 4)

@app.route('/video/process/', methods=['GET', 'POST'])
def video_process():
    """
    process video
    """
    if request.method != 'POST':
        return "use 'POST' method"
        
    req = request.json
    if req is None or len(req) == 0:
        return "empty request. be sure that request sended as JSON"
    
    print('--- New Req (video_process)---')

    # read video input and mask and ouput
    input_video = req.get('input_video')
    # token = req.get('token')
    token = request.headers.get('Authorization') # request.headers['']
    mask_path = req.get('mask_path', None)
    output_video = req.get('output_video', None)
    
    # init
    video = VideoEditor(input_video, token, mask_path, output_video)
    
    # start process in new thread
    thr = Thread(target = start_process_and_send_response, args=[video])
    thr.start()

    # return output result
    res = dict()
    res['status'] = 'success'
    
    return json.dumps(res, indent = 4)

def start_process_and_send_response(video):
    
    # process
    video.hide_face()

    # send token and new link
    res = dict()
    res['processed'] = 1 # res['status'] = 'success'
    res['_method'] = "PUT"
    # res['video'] = video.output_video # output_video
    # res['token'] = video.token
    json_res = json.dumps(res, indent = 4) 

    files = {'video' : open(video.output_video, "rb")} # send_from_directory(app.config["CLIENT_IMAGES"], filename=image_name, as_attachment=True) # response = send_file(safe_path, as_attachment=True)
    header = {"Authorization": video.token, "accept": "application/json", "Accept-Encoding": "gzip, deflate, br"}
    
    r = requests.put(WEB_SERVER + "/api/story/1", json = json_res, header = header, files = files)

    if r.status_code != 200:
        print("Error at sending done process sign")

@app.route('/video/upload/', methods=['GET', 'POST'])
def video_upload():
    """
    upload video
    """
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file'] # .read() / .save('/tmp/foo')
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        filename = datetime.now().strftime("%Y_%m_%d_%H_%M_%S___") + str(uuid.uuid4())
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('download_file', name=filename))
    return ''

@app.route('/ai/model_emotion/', methods=['GET', 'POST'])
def ai_model_emotion():
    
    req = request.json
    if req is None or len(req) == 0:
        return "empty request. be sure that request sended as JSON"
    
    print('--- New Req (ai_model_emotion)---')

    text = req.get('text')
    res = dict()
    res['status'] = 'success'
    res['result'] = model_emotion(text)

    return json.dumps(res, indent = 4)

@app.route('/ai/model_suicide_detection/', methods=['GET', 'POST'])
def ai_model_suicide_detection():
    
    req = request.json
    if req is None or len(req) == 0:
        return "empty request. be sure that request sended as JSON"
    
    print('--- New Req (ai_model_suicide_detection)---')

    text = req.get('text')
    res = dict()
    res['status'] = 'success'
    res['result'] = str(model_P(text)[0])

    return json.dumps(res, indent = 4)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)