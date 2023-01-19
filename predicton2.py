import os
import boto3
import cv2
import math
import pickle
import imageio
import warnings
import PIL.Image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output
from align_images import align
from encode_images import mainEncoder
import config
import dnnlib
import dnnlib.tflib as tflib
import glob

import os
import shutil
warnings.filterwarnings("ignore")




def get_watermarked(pil_image: Image) -> Image:
  try:
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    (h, w) = image.shape[:2]
    image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
    pct = 0.08
    full_watermark = cv2.imread('/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/BabyGAN-master/media/logo.png', cv2.IMREAD_UNCHANGED)
    (fwH, fwW) = full_watermark.shape[:2]
    wH = int(pct * h*2)
    wW = int((wH * fwW) / fwH*0.1)
    watermark = cv2.resize(full_watermark, (wH, wW), interpolation=cv2.INTER_AREA)
    overlay = np.zeros((h, w, 4), dtype="uint8")
    (wH, wW) = watermark.shape[:2]
    overlay[h - wH - 10 : h - 10, 10 : 10 + wW] = watermark
    output = image.copy()
    cv2.addWeighted(overlay, 0.5, output, 1.0, 0, output)
    rgb_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)
  except: return pil_image



def plot_three_images(imgB, fs = 10):
  f, axarr = plt.subplots(1,3, figsize=(fs,fs))
  axarr[0].imshow(Image.open('aligned_images/father_01.png'))
  axarr[0].title.set_text("Father's photo")
  axarr[1].imshow(imgB)
  axarr[1].title.set_text("Child's photo")
  axarr[2].imshow(Image.open('aligned_images/mother_01.png'))
  axarr[2].title.set_text("Mother's photo")
  plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
  # plt.show()


def generate_final_images(latent_vector, animation_size, direction, coeffs, i, face_img,generator):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeffs * direction)[:8]
    new_latent_vector = new_latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(new_latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    if animation_size[0] >= 512: img = get_watermarked(img)
    img_path = "/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/BabyGAN-master/for_animation/" + str(
        i) + ".png"
    img.thumbnail(animation_size, PIL.Image.ANTIALIAS)
    img.save(img_path)
    face_img.append(imageio.imread(img_path))
    clear_output()
    return img


def generate_final_image(latent_vector, size, direction, coeffs,generator,age):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeffs * direction)[:8]
    new_latent_vector = new_latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(new_latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    if size[0] >= 512: img = get_watermarked(img)
    img.thumbnail(size, PIL.Image.ANTIALIAS)
    img_save_path = "/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/BabyGAN-master/output_image/Pair_10/" + str(age) +'.png'
    img.save(img_save_path)
    return img, img_save_path



def main_image(age,hybrid_face, age_direction,generator):

    person_age = age #@param {type:"slider", min:10, max:50, step:1}
    intensity = -((person_age/5)-6)
    #@markdown ---
    #@markdown **Download the final image?**
    download_image = False #@param {type:"boolean"}
    #@markdown **Resolution of the downloaded image:**
    resolution = "512" #@param [256, 512, 1024]
    size = int(resolution), int(resolution)

    face,save_path = generate_final_image(hybrid_face,size, age_direction, intensity,generator,age)


    return face, save_path



# def face_vid(genes_influence,age_direction):
#     face_img = []
#     gender_influence = genes_influence # @param {type:"slider", min:0.01, max:0.99, step:0.01}
#     animation_resolution = "512"  # @param [256, 512, 1024]
#     animation_size = int(animation_resolution), int(animation_resolution)
#     # print(animation_size,"-----------------animation_size--------",type(animation_size))
#     # @markdown **Number of frames:**
#     frames_number = 20  # @param {type:"slider", min:9, max:50, step:1}
#     # @markdown **Option for animation:**
#     for i in range(3, frames_number+5, 1):
#         intensity = (i/5)-6
#
#         # horizontal_intensity = intensity / 25
#         # vertical_intensity = intensity / 25
#         # eyes_open_intensity = -(intensity / 12.5)
#         # gender_intensity = intensity / 50
#         # smile_intensity = intensity / 50
#         age_intensity = -(intensity)
#
#         # horizontal = horizontal_direction * horizontal_intensity
#         # vertical = vertical_direction * vertical_intensity
#         # eyes_open = eyes_open_direction * eyes_open_intensity
#         # gender = gender_direction * gender_intensity
#         # smile = smile_direction * smile_intensity
#         age = age_direction * age_intensity
#
#         dir_int = age  # @param ["horizontal", "vertical", "eyes_open", "gender", "smile", "age"] {type:"raw"}
#
#         generate_final_images(hybrid_face,animation_size, dir_int, 1, i,face_img,generator)
#         # clear_output()
#         print(str(i) + " of {} photo generated".format(str(frames_number)))
#
#     for j in reversed(face_img):
#         face_img.append(j)
#
#     # @markdown ---
#     # @markdown **Download the final animation?**
#     automatic_download = False  # @param {type:"boolean"}
#
#     if gender_influence <= 0.3:
#         animation_name = "boy.mp4"
#     elif gender_influence >= 0.7:
#         animation_name = "girl.mp4"
#     else:
#         animation_name = "animation.mp4"
#     imageio.mimsave('for_animation/' + animation_name, face_img)
#     clear_output()
#     if automatic_download == True:
#         files.download('for_animation/' + animation_name)
#     # display(mpy.ipython_display('for_animation/' + animation_name, height=400, autoplay=1, loop=1))
#     return 'for_animation/' + animation_name

def isEmpty(path):
    if os.path.exists(path) and not os.path.isfile(path):
        print('Empty...',os.path.isfile(path))
    else:
        files = glob.glob(path + "*")
        for f in files:
            os.remove(f)



def mainPrediction(mother_img,father_img,age,video=False,genes_influence = 0.2):
    Genrated_dir= 'generated_images/'
    latent_dir=  'latent_representations/'


    raw_path="raw_images/"

    if os.path.exists(raw_path):

        # Delete Folder code
        shutil.rmtree(raw_path)
        os.mkdir('raw_images')

        print("The folder has been deleted successfully!")
    else:
        os.mkdir('raw_images')

        print("Can not delete the folder as it doesn't exists")
    align_path="aligned_images/"
    isEmpty(raw_path)
    isEmpty(Genrated_dir)
    isEmpty(align_path)
    mother_img = cv2.imread(mother_img)
    cv2.imwrite(raw_path+'mother.jpeg', mother_img)
    father_img = cv2.imread(father_img)
    cv2.imwrite(raw_path+'father.jpeg', father_img)

    source_dir=align(raw_path,align_path)
    print("Face alignment Done..",source_dir)
    fgenerator,fdiscriminator_network=mainEncoder(source_dir,Genrated_dir,latent_dir)



    age_direction = np.load('/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/BabyGAN-master/ffhq_dataset/latent_directions/age.npy')
    # horizontal_direction = np.load('/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/BabyGAN-master/ffhq_dataset/latent_directions/angle_horizontal.npy')
    # vertical_direction = np.load('/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/BabyGAN-master/ffhq_dataset/latent_directions/angle_vertical.npy')
    # eyes_open_direction = np.load('/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/BabyGAN-master/ffhq_dataset/latent_directions/eyes_open.npy')
    # gender_direction = np.load('/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/BabyGAN-master/ffhq_dataset/latent_directions/gender.npy')
    # smile_direction = np.load('/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/BabyGAN-master/ffhq_dataset/latent_directions/smile.npy')
    from encoder.generator_model import Generator

    tflib.init_tf()

    # clear_cache()
    URL_FFHQ = "karras2019stylegan-ffhq-1024x1024.pkl"
    try:
        with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)
        generator = Generator(Gs_network, batch_size=1, randomize_noise=True)
    except:
        generator,discriminator_network=fgenerator,fdiscriminator_network
    model_scale = int(2*(math.log(1024,2)-1))



    #
    # # clear_output()
    # if len(os.listdir('generated_images')) == 2:
    #     first_face = np.load('latent_representations/father_01.npy')
    #     second_face = np.load('latent_representations/mother_01.npy')
    #     print("Generation of latent representation is complete! Now comes the fun part.")



    first_face = np.load('latent_representations/father_01.npy')
    second_face = np.load('latent_representations/mother_01.npy')


    genes_influence=genes_influence#@param {type:"slider", min:0.01, max:0.99, step:0.01}
    #@markdown **Styling a photo:**
    style = "Default" #@param ["Default", "Father's photo", "Mother's photo"]
    if style == "Father's photo":
      lr = ((np.arange(1,model_scale+1)/model_scale)**genes_influence).reshape((model_scale,1))
      rl = 1-lr
      hybrid_face = (lr*first_face) + (rl*second_face)
    elif style == "Mother's photo":
      lr = ((np.arange(1,model_scale+1)/model_scale)**(1-genes_influence)).reshape((model_scale,1))
      rl = 1-lr
      hybrid_face = (rl*first_face) + (lr*second_face)
    else: hybrid_face = ((1-genes_influence)*first_face)+(genes_influence*second_face)


    baby_image, image_path = main_image(age,hybrid_face, age_direction,generator)



    # image_path_lis = image_path.split("/")
    # if image_path_lis[-1] == "/" or image_path_lis[-1] == " " or image_path_lis[-1] == "":
    #     image_path_lis.pop()
    #
    # object_key = image_path_lis[-1]
    # bucket_name = 'animationmodel'  # bucket_name
    # file_name = image_path  # file path to upload
    # object_key = "babygan/" + object_key  # fine_name_in_s3
    # print(object_key, "fjhghgkgjkghjkhjkhjkhjkhjkghkghhhhhhhhhhhhhhhg")
    #
    # aws_id = 'AKIAYRFDMC3T2AXA6UXR'  # aws_access_key_id
    # aws_secret = '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'  # aws_secret_access_key
    # # client = boto3.client('s3', region_name=" ap-northeast-1", aws_access_key_id=aws_id, aws_secret_access_key=aws_secret,
    # #                       config=boto3.session.Config(signature_version='s3v4', ))
    # client = boto3.client('s3', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret)
    #
    # # s3_region = client.get_bucket_location(Bucket=bucket_name)
    # # region = s3_region['LocationConstraint']
    # # print(s3_region['LocationConstraint'])
    #
    # client.upload_file(Filename=file_name, Bucket=bucket_name, Key=object_key)
    # print("File uploaded successfully")
    #
    # response_image = client.generate_presigned_url('get_object',
    #                                                Params={'Bucket': bucket_name,
    #                                                        'Key': object_key},
    #                                                ExpiresIn=3600
    #                                                )
    #
    # response_video=''
    # # print(video,"----------------------------------------",type(video))
    # if video==True:
    #     print("-----------------------------making video")
    #     vid_path=face_vid()
    #     print(vid_path,'------------------videopath-----')
    #     vid_path=os.path.abspath(vid_path)
    #     video_path_lis = vid_path.split("/")
    #     if video_path_lis[-1] == "/" or video_path_lis[-1] == " " or video_path_lis[-1] == "":
    #         video_path_lis.pop()
    #     file_name2 = vid_path
    #     object_key2 = video_path_lis[-1]
    #
    #     object_key2 = "babygan/video/" + object_key2  # fine_name_in_s3
    #     client.upload_file(Filename=file_name2, Bucket=bucket_name, Key=object_key2)
    #     print("Video File uploaded successfully")
    #
    #     response_video = client.generate_presigned_url('get_object',
    #                                                    Params={'Bucket': bucket_name,
    #                                                            'Key': object_key2},
    #                                                    ExpiresIn=3600
    #                                                    )
    #     # print(vid_path,';;;;;;;;;;;;;;;;;;;')
    # # return {"Generated Baby image path: ":os.path.abspath(image_path),"Baby Animation Video path: ":vid_path}
    #
    #
    # # return {"Generated Baby image path: ":os.path.abspath(image_path),"Baby Animation Video path: ":vid_path}
    # return {"Generated Baby image path: ":response_image,"Baby Animation Video path: ":response_video}

# mother_img='/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/BabyGAN-master/mother.jpeg'
# father_img='/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/BabyGAN-master/father.jpeg'

# import glob
#
# path_='/home/aj/Documents/Pair_Images'
#
# folder_name=glob.glob(path_+'/*')
# print(folder_name)
# for i in folder_name:
#     print(i)
#     for images in glob.glob(i+ '/*'):
#         print(images.split('/')[-1].split('_')[0],'ffffffffffffffffffffff')
#         if images.split('/')[-1].split('_')[0]=='Man':
#
#             father_img=images
#             print(father_img, '14144444444444444444444444444444444')
#         else:
#             mother_img=images
#
#     res=mainPrediction(mother_img,father_img,age=10)
#     print(res)
#
#     quit()
MO='/home/aj/Documents/Pair_Images/Pair_10/Woman_10.png'
FA='/home/aj/Documents/Pair_Images/Pair_10/Man_10.jpg'
# for i in range(2,20,2):
    # res=mainPrediction(MO,FA,age=i)
# print(res)
# 2,4,7,10, 14
res=mainPrediction(MO,FA,age=14)
# print(res)