from flask import Flask, jsonify, request, render_template, redirect, send_from_directory, current_app, send_file
import os
import string, random
# from u2net_human_seg_test import main
from mainn import main_functionn
app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "./raw_images"
app.config["IMAGE_DOWNLOAD"] = "./final_output"
# app.config["IMAGE_UPLOADS"] = "/root/project/U-2-Net-master/test_data/test_human_images"
# app.config["IMAGE_DOWNLOAD"] = "/root/project/U-2-Net-master/test_data/test_human_images_results"

@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image2 = request.files["image2"]
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            image2.save(os.path.join(app.config["IMAGE_UPLOADS"], image2.filename))
            # agee=request.Ob('age')
            agee = int(request.form["age"])
            print(agee)
            # print(os.path.join(app.config["IMAGE_UPLOADS"], image.filename),'llllllllllll')
            inputt_path=os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
            inputt_path2=os.path.join(app.config["IMAGE_UPLOADS"], image2.filename)
            # print(image)
            # join = ''.join
            # filename_out = join(random.choices(string.ascii_letters, k=8))
            urlss=main_functionn(inputt_path,inputt_path2,app.config["IMAGE_DOWNLOAD"],agee)

        # return jsonify({'data': 'working','input_path':input,'output_path':outt})
        return jsonify({'Url': urlss})
        # return send_file(outt, as_attachment=True)

# @app.route('/root/project/U-2-Net-master/test_data/test_human_images_results/<path:filename>', methods=['GET', 'POST'])
# def download(filename):
#     print(app.root_path,'rooot')
#     full_path = os.path.join(app.root_path, app.config['IMAGE_DOWNLOAD'])
#     print(full_path,'dddd')
#     return send_from_directory(full_path, filename)


if __name__ == '__main__':
  app.run(debug=True, host = '0.0.0.0', port = 8000, threaded=False)
