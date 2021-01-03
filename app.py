import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import cloudinary
from cloudinary.uploader import upload

from flask import Flask, request, send_file
from gevent.pywsgi import WSGIServer
import io
import os
import time
import base64

characters = ['2','3','4','5','6','7','9','A','C','D','E','F','H','J','K','L','M','N','P','R','S','T','U','V','W','X','Y','Z']

prediction_model = load_model('turbobit.h5')

char_to_num = layers.experimental.preprocessing.StringLookup(vocabulary=list(characters), num_oov_indices=0, mask_token=None)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:,:4]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def get_code(img):
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [50, 150])
    img = tf.transpose(img, perm=[2, 1, 0])
    pred = prediction_model.predict(img)
    rez = decode_batch_predictions(pred)[0]
    return rez

cloudinary.config(cloud_name=os.environ['cloud_name'],
                  api_key=os.environ['api_key'],
                  api_secret=os.environ['api_secret'])
save_files = int(os.environ['save_files'])

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 24 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
MAX_FILE_SIZE = 24 * 1024 + 1

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/<path:filename>')
def last(filename):
    global lastlmage, lastcode
    return send_file(
                io.BytesIO(lastimage),
                attachment_filename='%s.png' % (filename),
                mimetype='image/png')

@app.route('/', methods=['GET', 'POST'])
@app.route('/turbo', methods=['GET', 'POST'])
def upload_file():
    global total, save_files, lastimage, lastcode, lastname
    if request.method == 'POST':
        rcode = ''
        if 'turbo' in request.path:
            tryes = '9'
            text_data = request.form['text']
            try:
                bytes_data = bytes(text_data, 'utf-8')
                file_bytes = base64.decodebytes(bytes_data)
            except:
                return 'TEST '
        else:
            file = request.files['file']
            if file and allowed_file(file.filename):
                file_bytes = file.read(MAX_FILE_SIZE)
            else:
                return 'TEST '
            tryes = file.filename[-5]
        total +=1
        try:
            rez = get_code(file_bytes)
            lastimage = file_bytes
            lastcode = rez
            lastname = rez + str(int(time.time()))[5:]
        except:
            rez = 'TEST  '
            lastimage = ''
        if rez.find('[UNK]') > -1:
            rcode = rez.replace('[UNK]', '8')
            rez = rez.replace('[UNK]', '')
            if len(rez):
                while len(rez) < 4:
                    rez = rez[0]+rez
            else:
                rez = 'QWER'
        if save_files:
            try:
                year, month, day, hour = map(int, time.strftime("%Y %m %d %H").split())
                dirname = 'upload/%s%s%s_%s/' % (year, month, day, hour//save_files)
                fname = str(int(time.time())) + '_' + rez + '_' + tryes + '0' + rcode
                upload_result = upload(file_bytes, folder = dirname, public_id = fname)
            except:
                pass
        return rez
    now = int(time.time())
    if 'turbo' in request.path:
        rez = '''
        <!doctype html>
        <title>Enter text</title>
        <h1>Enter text</h1>
        <form action="/turbo" method=post>
          <textarea name="text"></textarea>
             <input type="submit">
        </form>
        Working %s days %s hours %s min %s sec<br>
        Counter: %s
        ''' % ((now-start)//86400, ((now-start)%86400)//3600, ((now-start)%3600)//60, (now-start)%60, total)
        if lastimage != '':
            rez += '<br>Last file: %s' % (lastcode)
            rez += '<br><img src="/%s.png" alt="Image">' % (lastname)
        return rez
    rez = '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    Working %s days %s hours %s min %s sec<br>
    Counter: %s
    ''' % ((now-start)//86400, ((now-start)%86400)//3600, ((now-start)%3600)//60, (now-start)%60, total)
    if lastimage != '':
        rez += '<br>Last file: %s' % (lastcode)
        rez += '<br><img src="/%s.png" alt="Image">' % (lastname)
    return rez

if __name__ == '__main__':
    start = int(time.time())
    total = 0
    lastimage = ''
    lastcode = ''
    lastname = ''
    port = int(os.environ.get("PORT", 5000))
    http_server = WSGIServer(('0.0.0.0', port), app)
    http_server.serve_forever()
