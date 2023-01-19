import os
from flask import Flask, flash, request, redirect, url_for, current_app
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.getcwd() # '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

port = 5555
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part		
        print(request.files)		
        if ('mother-upload-button' not in request.files) and ('father-upload-button' not in request.files):
            flash('No file part')
            return redirect(request.url)
        file = request.files['mother-upload-button']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file', name=filename))
    return current_app.send_static_file('index.html')
    """return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''"""
	
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port)	