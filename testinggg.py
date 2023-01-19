import subprocess
import warnings
warnings.filterwarnings("ignore")
import os
import cv2
import math
import pickle
import imageio
import PIL.Image
import numpy as np
from PIL import Image
import tensorflow as tf
from random import randrange
import matplotlib.pyplot as plt
import config
import dnnlib
import dnnlib.tflib as tflib
# from encoder.generator_model import Generatoryear
from headpose.detect import PoseEstimator


# align_imagess='/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/new_babygan/aligned_images'
# test = os.listdir(align_imagess)
# # print(test)
# for images in test:
#     if images.endswith(".png"):
#         os.remove(os.path.join(align_imagess, images))

# exit()



father_image='./aligned_images/Man_6_01.png'
mother_image='./aligned_images/Woman_6_01.png'
output_folder='./output_folder/new_age_pair6'



def generate_image(latent_vector, direction, coeffs, filename='face', resolution=512,outt=output_folder):
    size = int(resolution), int(resolution)
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeffs * direction)[:8]
    new_latent_vector = new_latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(new_latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    img.thumbnail(size, PIL.Image.ANTIALIAS)

    img.save(f"{outt}/{filename}.png")
    return img

def plot_images(img_, fs=10):
  f, ax = plt.subplots(1, 3, figsize=(fs,fs))
  imgs_ = [Image.open('./aligned_images/Man_2_01.png'),
           img_, Image.open('./aligned_images/woman_2_01.png')]
  titles_ = ["Father's photo", "Child's photo", "Mother's photo"]
  for i in range(len(ax)):
        ax[i].imshow(imgs_[i])
        ax[i].title.set_text(titles_[i])
  plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
  plt.show()


def get_pose(face):
    image = np.array(face)
    est = PoseEstimator()
    est.detect_landmarks(image)
    roll, pitch, yaw = est.pose_from_image(image)
    return roll, pitch, yaw


def generate_parameters(father_face, mother_face, pil_father, pil_mother, yaw_mean, emphasis="father", gender=50,
                        age=-100, genes_influence=0.01):
    # Model dimension ffhq 1024x1024
    output_scale = int(2 * (math.log(1024, 2) - 1))
    if emphasis == "father":
        lr = ((np.arange(1, output_scale + 1) / output_scale) ** genes_influence)
        lr = lr.reshape((output_scale, 1))
        rl = 1 - lr
        hybrid_face = (lr * father_face) + (rl * mother_face)

    elif emphasis == "mother":
        lr = ((np.arange(1, output_scale + 1) / output_scale) ** (1 - genes_influence)).reshape((output_scale, 1))
        lr = lr.reshape((output_scale, 1))
        rl = 1 - lr
        hybrid_face = (rl * father_face) + (lr * mother_face)

    else:
        hybrid_face = ((1 - genes_influence) * father_face) + (genes_influence * mother_face)

    if yaw_mean <= 0:
        horizontal_intensity = max(-1.5, yaw_mean)
    else:
        horizontal_intensity = min(1.5, yaw_mean)

    vertical = 0
    vertical_intensity = vertical / 50
    gender = gender
    gender_intensity = gender / 100
    age = age
    age_intensity = -(age / 25)
    face_params = (gender_direction * gender_intensity) + (age_direction * age_intensity) + (
                horizontal_direction * horizontal_intensity) + (vertical_direction * vertical_intensity)
    return hybrid_face, face_params

directories = ["./aligned_images", "./data", "./father_image", "./mother_image", output_folder]
for directory in directories:
    if not os.path.exists(f'{directory}'):
        os.mkdir(directory)

#
age_direction = np.load('./ffhq_dataset/latent_directions/age.npy')
horizontal_direction = np.load('./ffhq_dataset/latent_directions/angle_horizontal.npy')
vertical_direction = np.load('./ffhq_dataset/latent_directions/angle_vertical.npy')
eyes_open_direction = np.load('./ffhq_dataset/latent_directions/eyes_open.npy')
gender_direction = np.load('./ffhq_dataset/latent_directions/gender.npy')
smile_direction = np.load('./ffhq_dataset/latent_directions/smile.npy')
#
# # !python align_images.py ./father_image ./aligned_images
#
# if not os.path.isfile('/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/new_babygan/aligned_images//Man_2_01.png'):
#     raise ValueError('No face was found or there is more than one in the photo.')
#
#

if os.path.isfile(father_image):
  pil_father = Image.open(father_image)
  (fat_width, fat_height) = pil_father.size
  resize_fat = max(fat_width, fat_height)/256
  pil_father.resize((int(fat_width/resize_fat), int(fat_height/resize_fat)))
else:
    raise ValueError('Failed to find appropriate number of faces')
#
# # !python align_images.py ./mother_image ./aligned_images
#
#
# if not os.path.isfile('./aligned_images/woman_2_01.png'):
#     raise ValueError('No face was found or there is more than one in the photo.')
#
#

if os.path.isfile(mother_image):
  pil_mother = Image.open(mother_image)
  (mot_width, mot_height) = pil_mother.size
  resize_mot = max(mot_width, mot_height)/256
  pil_mother=pil_mother.resize((int(mot_width/resize_mot), int(mot_height/resize_mot)))
  # img_save_path = "/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/new_babygan/hhh/mother.png"
  # pil_mother.save(img_save_path)
else:
    raise ValueError('Failed to find appropriate number of faces.')
#
#
# # !python encode_images.py --early_stopping False --lr=0.25 --batch_size=2 --iterations=100 ./aligned_images ./generated_images ./latent_representations
#
#
tflib.init_tf()
#
from encoder.generator_model import Generator
URL_FFHQ = "/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/new_babygan/karras2019stylegan-ffhq-1024x1024.pkl"
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)
#
#
generator = Generator(Gs_network, batch_size=1, randomize_noise=False)
#
#
if len(os.listdir('./generated_images')) >= 2:

    latent_father = os.path.split(father_image)[1].split('.')[0]
    latent_mother = os.path.split(mother_image)[1].split('.')[0]
    genn_images_f = './generated_images/' + str(latent_father)+'.png'
    genn_images_m = './generated_images/' + str(latent_mother)+'.png'

    f_lateeent='./latent_representations/'+str(latent_father)+'.npy'
    m_lateeent='./latent_representations/'+str(latent_mother)+'.npy'
    father_face = np.load(f_lateeent)
    # print(father_face)
    mother_face = np.load(m_lateeent)
    print("Generation of latent representation is complete!")
else:
    raise ValueError('Failed to read/locate latent representation of provided images!')
#
#
_, _, yawF = get_pose(pil_father)
_, _, yawM = get_pose(pil_mother)
yaw_mean = np.mean([yawF * 0.35, yawM * 0.35])
#
#
#
#
# # :: +ve right, -ve left
# # :: +ve down, -ve up
# # :: +ve masc, -ve fem
# # :: +ve older, -ve younger
#
params = []
age, gender = -50, 0
genes_influence = 0.5
age_offset = 10
gender_offset = 20
gene_age_offset = 0.075
gene_gender_offset = 0.2
gene_offset = 0.05
hh=[-70,-60,-50,-40,-30]
for i in range(10):
    if i < 5:
        # params.append(generate_parameters(father_face, mother_face, pil_father, pil_mother, yaw_mean,
        #                                   emphasis="father", gender=(gender + (gender_offset * i)),
        #                                   age=(age + (age_offset * i)),
        #                                   genes_influence=(genes_influence - gene_gender_offset - (gene_offset * i))))
        params.append(generate_parameters(father_face, mother_face, pil_father, pil_mother, yaw_mean,
                                          emphasis="father", gender=(gender + (gender_offset * i)),
                                          age=hh[i],
                                          genes_influence=(genes_influence - gene_gender_offset - (gene_offset * i))))
    else:
        # params.append(generate_parameters(father_face, mother_face, pil_father, pil_mother, yaw_mean,
        #                                   emphasis="mother", gender=(gender + (-gender_offset * (i % 5))),
        #                                   age=(age + (age_offset * (i % 5))),
        #                                   genes_influence=(genes_influence + gene_gender_offset + (gene_offset * (i % 5)))))
        params.append(generate_parameters(father_face, mother_face, pil_father, pil_mother, yaw_mean,
                                          emphasis="mother", gender=(gender + (-gender_offset * (i % 5))),
                                          age=hh[i-5],
                                          genes_influence=(
                                                      genes_influence + gene_gender_offset + (gene_offset * (i % 5)))))

# Combined male
# params.append(generate_parameters(father_face, mother_face, pil_father, pil_mother, yaw_mean,
#                                   emphasis="father", gender=50,
#                                   age=-60, genes_influence=0.15))
# # Combined female
# params.append(generate_parameters(father_face, mother_face, pil_father, pil_mother, yaw_mean,
#                                   emphasis="mother", gender=-50,
#                                   age=-60, genes_influence=0.85))



i = 1
for hybrid_face, face_params in params:
    face = generate_image(hybrid_face, face_params, 1, filename=i, resolution=1024,outt=output_folder)
    # plot_images(face, fs=20)
    i += 1


oocc=[father_image,mother_image,m_lateeent,f_lateeent,genn_images_f,genn_images_m]
for o in oocc:
    os.remove(o)
#
# #
# # for file in os.listdir('./output'):
# #     img_ = Image.open('./output/' + file)
#     # display(img_)


# # !backgroundremover -i "face.png" -o "face.jpg"
# for file in os.listdir('./output_folder/pair_1/'):
#     new_name = 'no_bg_' + file
#     # !backgroundremover -i {'./output/' + file} -m "u2net_human_seg" -o {'./output/' + new_name}
#     res = subprocess.call(['backgroundremover', '-i', str('./output/' + str(file)), '-m', 'u2net_human_seg', '-o', str('./output/' + str(new_name))])
# backgroundremover -i ./output/2.png -m "u2net_human_seg" -o ./output/2_new.png
#
# for file in os.listdir('./output'):
#     if 'no_bg' in file:
#         img_ = Image.open('./output/' + file)
#         (width_, height_) = img_.size
#         resize_ = max(width_, height_)/256
#         # display(img_.resize((int(width_/resize_), int(height_/resize_))))