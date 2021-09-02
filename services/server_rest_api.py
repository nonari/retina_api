#!flask/bin/python
from flask import Flask, request, send_file, abort, make_response, jsonify
from flask_httpauth import HTTPBasicAuth

from werkzeug.utils import secure_filename
import os
import cv2 as cv
import base64
from PIL import Image
from skimage import img_as_ubyte, img_as_float
import numpy as np
import random

from utils.request_data_utils import *

import torch
from matplotlib import pyplot
from matplotlib import patches
from torchvision import transforms
import segmentation_models_pytorch as smp

from datetime import datetime

import argparse
import copy
import io

import time
import multiprocessing as mp
import cv2

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv("UPLOAD_FOLDER")
app.config['OUTPUT_FOLDER'] = os.getenv("OUTPUT_FOLDER")

auth = HTTPBasicAuth()

pool = mp.Pool()

@auth.get_password
def get_password(username):
    if username == 'varpa':
        return 'radiology'
    return None

@app.errorhandler(400)
def send_error_msg(error):
    return make_response(jsonify({'error': error.description}), 400)

def get_filepaths(upload_folder, output_folder, user, suffix):
    cmd = 'mkdir -p ' + upload_folder + user
    os.system(cmd)

    cmd = 'mkdir -p ' + output_folder + user
    os.system(cmd)

    imagename =  user + '_' + get_timestamp_str()
    img_path = upload_folder + user + '/' +  imagename +  '.jpg'
    out_path = output_folder + user + '/' + imagename +  suffix + '.jpg'
    outfile_path = output_folder + user + '/' + user +  suffix + '.txt'

    return imagename, img_path, out_path, outfile_path

def rearrange_mask(mask):
    sparse_labels = np.unique(mask)
    for idx, l in enumerate(sparse_labels):
        mask[mask == l] = idx + 1


def fill_gaps(im, layer):
    mask = im.copy()
    mask[mask != layer] = 0
    stats = cv.connectedComponentsWithStats(im, connectivity=8)
    vols = stats[2][1:, 1]
    obj_idx = np.argmin(vols) + 1
    res = stats[1]
    res[res != obj_idx] = 0
    res = res.astype(np.uint8)
    conts, hier = cv.findContours(res, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    filled_gaps = np.zeros(im.shape)
    cv.drawContours(filled_gaps, conts, -1, 1, thickness=cv.FILLED)
    im[filled_gaps == 1] = layer


def layer_thickness(im, layer):
    mask = im.copy()
    mask[mask != layer] = 0
    mask[mask == layer] = 1
    arrow = np.arange(0, mask.shape[0])
    arrow = arrow.reshape(mask.shape[0], 1)
    hmap = arrow * mask
    hmap[hmap == 0] = 999999
    upper = np.argmin(hmap, axis=0)
    hmap[hmap == 999999] = -1
    lower = np.argmax(hmap, axis=0)
    diff_layer = lower - upper + 1
    return diff_layer, upper, lower


def mask_retina(im):
    im = im.copy()
    im[im == 9] = 0
    im[im == 1] = 0
    im[im == 10] = 1
    for i in range(2, 9):
        im[im == i] = 1
    return im


def fluid_stats(im, orig):
    im = im.copy()
    im[im != 10] = 0
    im[im == 10] = 1

    fluid_data = {}
    stats = cv.connectedComponentsWithStats(im)
    areas = stats[2][1:, 4]
    total_count = len(areas)
    fluid_data['total'] = total_count

    centroids = stats[3][1:, 0]

    x, y = im.shape
    areas_zone = []
    count_zone = []
    sep = [0, 0.285, 0.428, 0.571, 0.714, 1]
    for i in range(5):
        p1, p2 = sep[i], sep[i + 1]
        ini, fin = int(p1 * x), int(p2 * x)
        if i > 0:
            cv.line(orig, (ini, 0), (ini, y), thickness=1, color=(250, 255, 61))
        area = (im[:, ini:fin] == 1).sum()
        areas_zone.append(int(area))
        count_l = ini < centroids
        count_g = fin > centroids
        count = np.logical_and(count_l, count_g).sum()
        count_zone.append(int(count))

    fluid_data["area_by_zone"] = areas_zone
    fluid_data["count_by_zone"] = count_zone
    conts, hier = cv.findContours(im, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    by_cyst = {}
    fluid_data['by_cyst'] = by_cyst
    for c in conts:
        _, _, w, h = cv.boundingRect(c)
        x, y = c[0, 0]
        idx = int(stats[1][y, x])
        by_cyst[idx] = {"measures": (w, h)}
        by_cyst[idx]["ratio"] = float(round(h / w, 3))
        by_cyst[idx]["area"] = int(stats[2][idx, 4])

    return fluid_data


layer_names = ["HV", "NFL", "GCL-IPL", "INL", "OPL", "OPL-ISM", "ISE", "OS-RPE", "C", "Fluid"]


def layers_info(im, orig=None):
    layers = {}
    by_layer = {}
    layers["by_layer"] = by_layer

    for l_idx in range(2, 9):
        thickness, upper, lower = layer_thickness(im, l_idx)
        if orig is not None:
            for idx, (u, l) in enumerate(zip(upper, lower)):
                cv.circle(orig, (idx, u), radius=0, color=(0, 255, 0), thickness=1)
                cv.circle(orig, (idx, l), radius=0, color=(0, 255, 0), thickness=1)
        std = float(round(np.std(thickness), 3))
        mean = float(round(np.mean(thickness), 3))
        maxim = np.max(thickness)
        minim = np.min(thickness)
        name = layer_names[l_idx - 1]
        by_layer[name] = {"std": float(std)}
        by_layer[name]['mean'] = float(mean)
        by_layer[name]['max'] = int(maxim)
        by_layer[name]['min'] = int(minim)

    retina_mask = mask_retina(im)
    thickness, upper, lower = layer_thickness(retina_mask, 1)
    std = np.std(thickness)
    mean = np.mean(thickness)
    maxim = np.max(thickness)
    minim = np.min(thickness)
    layers['retina_std'] = float(std)
    layers['retina_mean'] = float(mean)
    layers['retina_max'] = int(maxim)
    layers['retina_min'] = int(minim)

    return layers

def get_numpy_img(img_data):
    return np.array(img_data)

def get_timestamp_str():
    return datetime.now().isoformat()

def prediction_to_mask_x(tensor):
    arr = torch.squeeze(tensor, dim=0).numpy()
    mask = np.argmax(arr, axis=0)
    mask = mask + 1

    return mask

im_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Resize((512, 512), Image.BILINEAR)
])

def process(img, model):
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    net = smp.Unet(
        encoder_name=model,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",
        in_channels=1,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=10,  # model output channels (number of classes in your dataset)
    )
    net.load_state_dict(torch.load(f'{model}.pth', map_location=device))
    img_tensor = im_trans(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    net.eval()
    pred = net(img_tensor)
    pred = pred.clone().detach()
    mask = prediction_to_mask_x(pred)
    return mask


def color_layer(raw_mask):
    buf = io.BytesIO()
    raw_mask[0, 0] = 10
    pyplot.imsave(buf, raw_mask)
    # pyplot.savefig(buf, format='png')
    im = Image.open(buf)
    mask = np.array(im)[:, :, :3]

    uniques = np.unique(raw_mask)
    color_assoc = {}
    leg_color = []
    for idx, u in enumerate(uniques):
        matches = np.where(raw_mask == u)
        y, x = matches[0][0], matches[1][0]
        rgb = list(map(lambda i: int(i), mask[y, x, :]))
        color_assoc[layer_names[idx]] = rgb
        leg_color.append(patches.Patch(color=tuple(mask[y, x, :]/255), label=layer_names[idx]))

    pyplot.imshow(mask)
    pyplot.legend(handles=leg_color, bbox_to_anchor=(1.3, 1))
    buf = io.BytesIO()
    pyplot.savefig(buf, format='png')
    label_mask = Image.open(buf)

    return color_assoc, np.array(label_mask)[:, :, :3]

def show_fluid(mask, img):
    mask = mask.copy()
    mask[mask != 10] = 0
    mask[mask == 10] = 1
    conts, hier = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, conts, -1, (0, 0, 255), thickness=2)
    stats = cv.connectedComponentsWithStats(mask)
    centroids = stats[3][1:]
    for c in centroids:
        cv.drawMarker(img, (int(c[0]), int(c[1])), (255,0,0), markerType=cv.MARKER_CROSS, markerSize=6, thickness=2)


################################################################################################################
### Services
@app.route('/services/proofService', methods=['POST'])
#@auth.login_required
def proofService_url():
    print('\n--------------------------------------------------------------------------')
    print(request.headers)
    try:
        image = get_image_from_request(request)
        (user, institution, department, description) = get_data_from_request(request)
        if 'model' in request.json and len(request.json['model']) > 0:
            model = request.json['model']
        else:
            print('Net model not specified, defaulting to resnet34')
            model = 'resnet34'
    except:
        print ('Proof service - Invalid data in request')
        abort(400, 'Invalid data in request')

    return computeProofService(image, user, institution, department, description, '_proof', model)


################################################################################################################

def computeProofService(out_im_data, user, institution, department, description, output_suffix, model):
    try:
        img_data = Image.open(out_im_data)
        img_data = img_data.convert('RGB')  # to accept any image format
    except:
        abort(400, 'Invalid image data')

    try:
        imagename, img_path, out_path, outfile_path = get_filepaths(app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], user, output_suffix)
        img_data.save(img_path)
        
        print (time.strftime('%X %x %Z'), 'Computing Proof service for', img_path)

        # Process image
        img = get_numpy_img(img_data)
        img_orig = img.copy()
        img = np.dot(img, [0.2989, 0.5870, 0.1140])
        img = img.astype(np.uint8)
        print(img.shape)
        mask = process(img, model)
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, img_orig.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        show_fluid(mask, img_orig)
        fluid_data = fluid_stats(mask, img_orig)
        layers_data = layers_info(mask, img_orig)
        result = {'fluid_data': fluid_data, 'layers_data': layers_data}
        proportion = sum(fluid_data['area_by_zone']) / (layers_data['retina_mean'] * mask.shape[0])
        result['proportion'] = proportion
        #output_img = cv2.circle(mask, (int(mask.shape[0]/2),int(img.shape[1]/2)), 80, (0,255,0), 3)

        output_img = Image.fromarray(img_orig)
        color_assoc, color_mask = color_layer(mask)
        result['colors_map'] = color_assoc
        output_mask = Image.fromarray(color_mask)

        # Save data to file
        f = open(outfile_path, "a")
        f.write('%s;%s;%s;%s;%s;\n' % (imagename, user, institution, department, description))
        f.close()

        # Save img result
        out_im_data = io.BytesIO()
        output_img.save(out_im_data, 'JPEG')
        pool.apply_async(output_img.save, args=(out_path, 'JPEG'))
        encoded_out_image = base64.b64encode(out_im_data.getvalue()).decode('utf-8')

        # Save img result
        out_mask_data = io.BytesIO()
        output_mask.save(out_mask_data, 'JPEG')
        pool.apply_async(output_mask.save, args=('./', 'JPEG'))
        encoded_out_mask = base64.b64encode(out_mask_data.getvalue()).decode('utf-8')

        result['image'] = encoded_out_image
        result['mask'] = encoded_out_mask
        result['pathological'] = proportion > 0.05

        return jsonify(result)

    except Exception as e:
        print('Proof service - Image could not be processed. Exception', e)
        abort(400, 'Image could not be processed')


###############################################################################################################################################################################

#Usage: python3 server_rest_api.py --debug
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    app.run(debug=args.debug, host='0.0.0.0', port=5100) 


