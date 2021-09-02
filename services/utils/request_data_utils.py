#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 19:36:15 2021
@author: varpa
"""
import io
import base64
import cv2
import numpy as np
from PIL import Image
import sys

def get_image_from_request(request):
    if request.json and 'image' in request.json:
        return io.BytesIO(base64.b64decode(request.json['image']))
    elif request.data != b'':
        return io.BytesIO(request.data)
    elif 'file' in request.files:
        return request.files['file']
    else:
        return(400, 'No data')

def get_data_from_request(request):
    user='Nouser'
    description='No description'
    institution='No institution'
    department='No department'
    if request.json:
        if 'user' in request.json and len(request.json['user'])>0:
            user = request.json['user']
        if 'institution' in request.json and len(request.json['institution'])>0:
            institution = request.json['institution']
        if 'department' in request.json and len(request.json['department'])>0:
            department = request.json['department']
        if 'description' in request.json and len(request.json['description'])>0:
            description = request.json['description']
    return (user, institution, department, description)