from flask import Flask
from flask import request, Response, render_template, send_file
import numpy as np
import os
from inference import CFG, infer, process_input, write_result

app = Flask(__name__)
cfg = CFG()

@app.route('/', methods = ['GET', 'POST'])
def index():
    img = request.files['image']
    input_fname = str(np.random.randint(100000000)) + '.jpg'
    input_path = os.path.join(cfg.testimgs_dir, input_fname)
    print(input_path)
    img.save(input_path)
    img, msk = infer(cfg, input_path)
    img, msk, output = process_input(img, msk, cfg.grass_img_path)
    write_result(cfg, input_fname, output)

    return send_file(os.path.join(cfg.output_dir, input_fname), mimetype='image/gif')

app.run(host='0.0.0.0', port=5000, debug=False)