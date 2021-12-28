from flask import Flask
from flask import request, send_file
import numpy as np
import os, cv2
from inference import CFG, infer
from crop import crop

app = Flask(__name__)
cfg = CFG()

@app.route('/', methods = ['GET', 'POST'])
def index():
    img = request.files['image']
    input_fname = str(np.random.randint(100000000)) + '.jpg'
    input_path = os.path.join(cfg.input_dir, input_fname)
    msk_path = input_path.replace('/input/', '/out/')
    print(input_path)
    img.save(input_path)
    img, msk = infer(cfg, input_path)
    cv2.imwrite(msk_path, msk)
    cropped = crop(input_path, msk_path)
    cropped_path = input_path.replace('/input/', '/cropped/')
    cv2.imwrite(cropped_path, cropped)

    return send_file(os.path.join(cfg.cropped_path, input_fname), mimetype='image/gif')

app.run(host='0.0.0.0', port=5000, debug=False)