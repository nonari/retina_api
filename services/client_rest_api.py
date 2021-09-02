import requests
import os
import argparse
import base64
import json as js


def post_image(url, filepath, savepath, model):
    response = None
    json = {'model': model}
    try:
        with open(filepath, 'rb') as file:    
            # data = file.read()
            image_data = base64.b64encode(file.read())
            json['image'] = image_data
            response = requests.post(url, json=json)
    except:
        print (url, '- Invalid data in request')
        return 
        
    process_response_json(response, savepath)

        
def process_response_json(response, savepath):
    if response is not None:
        data = None
        try:
            data = response.json()
        except:
            print('No json in response')
                    
        if data is not None:    
            print(data)
            if 'image' in data:
                img_data = base64.b64decode(data['image'])
                with open(savepath, 'wb') as file:
                    file.write(img_data)
            elif 'error' in data:
                print(data['error'])
    else:
        print('No response')


# Usage: python3 client_rest_api.py --img ../images/image1.jpg --output image1-out.jpg
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str) 
    parser.add_argument('--output', type=str, default='response.jpg')
    parser.add_argument('--model', type=str, default="")
    args = parser.parse_args()
    
    # Server en local
    proof_url ='http://172.17.0.1:5100/services/proofService'

    post_image(proof_url, args.img, args.output, args.model)

