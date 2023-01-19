import glob
import os
import shutil
from argparse import Namespace
import sys
from encode import run_single
# sys.path.append(".")
# sys.path.append("..")
# from align_images import main_alligned
import time
import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from criteria.parse_related_loss import average_lab_color_loss
from tqdm import tqdm
from mapper.datasets.latents_dataset_inference import LatentsDatasetInference
from mapper.options.test_options import TestOptions
from mapper.hairclip_mapper import HairCLIPMapper


def run(test_opts,latent_path,out_path):
    original_name = os.path.split(latent_path)[1].split('.')[0]
    device = 'cuda:0'
    # out_path_results = os.path.join(test_opts.exp_dir, test_opts.editing_type, test_opts.input_type)
    out_path_results=out_path
    os.makedirs(out_path_results, exist_ok=True)
    # update test options with options used during training
    # ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    ckpt = torch.load(test_opts.checkpoint_path)
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)
    net = HairCLIPMapper(opts)
    net.eval()
    net.cuda()
    print(os.path.join(out_path_results, original_name + ".jpg"))
    latent_codes_origin = np.reshape(np.load(latent_path), (1, 18, 512))

    mapper_input = latent_codes_origin.copy()
    test_latents = torch.from_numpy(mapper_input).cuda().float()

    # test_latents = torch.load(opts.latents_test_path)
    dataset = LatentsDatasetInference(latents=test_latents.cpu(),
                                      opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)
    opts.parsenet_weights='pretrained_models/parsenet.pth'
    average_color_loss = average_lab_color_loss.AvgLabLoss(opts).to(device).eval()
    assert (opts.start_index >= 0) and (opts.end_index <= len(dataset))
    global_i = 0
    for input_batch in tqdm(dataloader):
        if global_i not in range(opts.start_index, opts.end_index):
            if global_i >= opts.end_index:
                break
            global_i += 1
            continue
        with torch.no_grad():
            w, hairstyle_text_inputs_list, color_text_inputs_list, selected_description_tuple_list, hairstyle_tensor_list, color_tensor_list = input_batch
            for i in range(len(selected_description_tuple_list)):
                hairstyle_text_inputs = hairstyle_text_inputs_list[i]
                color_text_inputs = color_text_inputs_list[i]
                selected_description = selected_description_tuple_list[i][0]
                hairstyle_tensor = hairstyle_tensor_list[i]
                color_tensor = color_tensor_list[i]
                w = w.cuda().float()
                hairstyle_text_inputs = hairstyle_text_inputs.cuda()
                color_text_inputs = color_text_inputs.cuda()
                hairstyle_tensor = hairstyle_tensor.cuda()
                color_tensor = color_tensor.cuda()
                if hairstyle_tensor.shape[1] != 1:
                    hairstyle_tensor_hairmasked = hairstyle_tensor * average_color_loss.gen_hair_mask(hairstyle_tensor)
                else:
                    hairstyle_tensor_hairmasked = torch.Tensor([0]).unsqueeze(0).cuda()
                if color_tensor.shape[1] != 1:
                    color_tensor_hairmasked = color_tensor * average_color_loss.gen_hair_mask(color_tensor)
                else:
                    color_tensor_hairmasked = torch.Tensor([0]).unsqueeze(0).cuda()
                result_batch = run_on_batch(w, hairstyle_text_inputs, color_text_inputs, hairstyle_tensor_hairmasked,
                                            color_tensor_hairmasked, net)

                if (hairstyle_tensor.shape[1] != 1) and (color_tensor.shape[1] != 1):
                    img_tensor = torch.cat([hairstyle_tensor, color_tensor], dim=3)
                elif hairstyle_tensor.shape[1] != 1:
                    img_tensor = hairstyle_tensor
                elif color_tensor.shape[1] != 1:
                    img_tensor = color_tensor
                else:
                    img_tensor = None

                im_path = str(global_i).zfill(5)
                if img_tensor is not None:
                    if img_tensor.shape[3] == 1024:
                        couple_output = torch.cat(
                            [result_batch[2][0].unsqueeze(0), result_batch[0][0].unsqueeze(0), img_tensor])
                    elif img_tensor.shape[3] == 2048:
                        couple_output = torch.cat([result_batch[2][0].unsqueeze(0), result_batch[0][0].unsqueeze(0),
                                                   img_tensor[:, :, :, 0:1024], img_tensor[:, :, :, 1024::]])
                else:
                    couple_output = torch.cat([result_batch[2][0].unsqueeze(0), result_batch[0][0].unsqueeze(0)])

                # torchvision.utils.save_image(couple_output, os.path.join(out_path_results, f"{im_path}-{str(i).zfill(4)}-{selected_description}.jpg"), normalize=True, range=(-1, 1))
                # print(out_path_results)
                # torchvision.utils.save_image(result_batch[0][0].unsqueeze(0), os.path.join(out_path_results,"result.jpg"),
                #                              normalize=True, range=(-1, 1))
                print(os.path.join(out_path_results, original_name+".jpg"))

                torchvision.utils.save_image(result_batch[2][0].unsqueeze(0),
                                            os.path.join(out_path_results, original_name+".png"),
                                            normalize=True, range=(-1, 1))

            global_i += 1

        return couple_output


def run_on_batch(inputs, hairstyle_text_inputs, color_text_inputs, hairstyle_tensor_hairmasked, color_tensor_hairmasked,
                 net):
    w = inputs
    with torch.no_grad():
        w_hat = w + 0.1 * net.mapper(w, hairstyle_text_inputs, color_text_inputs, hairstyle_tensor_hairmasked,
                                     color_tensor_hairmasked)
        x_hat, w_hat = net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False,
                                   truncation=1)
        result_batch = (x_hat, w_hat)
        x, _ = net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1)
        result_batch = (x_hat, w_hat, x)
    return result_batch



def backkground_clean(path):

    t=time.time()
    test_opts = TestOptions().parse()
    if test_opts.test_batch_size != 1:
        raise Exception('This version only supports test batch size to be 1.')
    # main_alligned('indir', 'outdir')
    # dir_alline = main_alligned('indir', 'outdir')
    # path = run_single(dir_alline)
    # latent_path='/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/hairai/HairCLIP-main/code/atikh-bana-2c0midsQKe0-unsplash.npy'
    # src_dir='/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/new_babygan/output_folder/pair_1/'
    # dest_dir='outdir'
    # files = os.listdir(src_dir)

    # shutil.copytree(src_dir, dest_dir)
    # shutil.copy(src_dir, dest_dir)

    dir_alline=path
    latent_paths=run_single(dir_alline)
    print(latent_paths)
    for latent_path in latent_paths:
        print(latent_paths)
        # imagee='./outdir/'+os.path.split(latent_path)[1].split('.')[0]+'.png'
    # print(latent_path,'kkkkk')
        with open('readme.txt', 'w') as f:
            f.write('bob cut hairstyle')

        test_opts.exp_dir='poop'
        test_opts.color_description='light brown'
        test_opts.input_type='text'
        # test_opts.hairstyle_description='hairstyle_list.txt'
        test_opts.hairstyle_description='readme.txt'
        # try:
        run(test_opts,latent_path,dir_alline)
        # except:
        #     pass
        os.remove(latent_path)
        # os.remove(imagee)
    # # tensor__ = torch.stack(couple_output, dim=0)
    # ndarr = couple_output.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # im = Image.fromarray(ndarr)

    # im=re_image(couple_output,normalize=True, range=(-1, 1))
    # im.show()

    # plt.imshow(im)
    # plt.show()
    # print(time.time()-t)
