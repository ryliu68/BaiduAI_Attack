from __future__ import absolute_import, division, print_function

import argparse
import functools
import os

import numpy as np

import models
import paddle
import paddle.fluid as fluid
from PIL import Image
from utils import *
from utils import (add_arguments, calc_mse, init_prog, print_arguments,
                   process_img, save_adv_image, tensor2img)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('class_dim',        int,   121,                  "Class number.")
add_arg('shape',            str,   "3,224,224",          "output image shape")
add_arg('input',            str,   "input_image/",
        "Input directory with images")
add_arg('output',           str,   "output_image/",
        "Output directory with images")

args = parser.parse_known_args()[0]
print_arguments(args)

image_shape = [int(m) for m in args.shape.split(",")]
class_dim = args.class_dim
input_dir = args.input
output_dir = args.output

model_name1 = "DARTS_4M"
model_name2 = "Adv_MobileNetV2_x2_0"
model_name3 = "Adv_InceptionV4"
model_name4 = "Adv_DenseNet169"
model_name5 = "Adv_ResNeXt50_32x4d"
model_name6 = "Adv_DPN98"
model_name7 = "ResNeXt50_vd_64x4d"
model_name8 = "VGG19"
pretrained_model = "models_parameters"

val_list = 'val_list.txt'
use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
main_programs = fluid.default_main_program()

input_layer = fluid.layers.data(
    name='image', shape=image_shape, dtype='float32')
input_layer.stop_gradient = False

model1 = models.__dict__[model_name1]()
out_logit1 = model1.net(input=input_layer, class_dim=class_dim)
out1 = fluid.layers.softmax(out_logit1)

model2 = models.__dict__[model_name2]()
out_logit2 = model2.net(input=input_layer, class_dim=class_dim)
out2 = fluid.layers.softmax(out_logit2)

model3 = models.__dict__[model_name3]()
out_logit3 = model3.net(input=input_layer, class_dim=class_dim)
out3 = fluid.layers.softmax(out_logit3)

model4 = models.__dict__[model_name4]()
out_logit4 = model4.net(input=input_layer, class_dim=class_dim)
out4 = fluid.layers.softmax(out_logit4)

model5 = models.__dict__[model_name5]()
out_logit5 = model5.net(input=input_layer, class_dim=class_dim)
out5 = fluid.layers.softmax(out_logit5)

model6 = models.__dict__[model_name6]()
out_logit6 = model6.net(input=input_layer, class_dim=class_dim)
out6 = fluid.layers.softmax(out_logit6)

model7 = models.__dict__[model_name7]()
out_logit7 = model7.net(input=input_layer, class_dim=class_dim)
out7 = fluid.layers.softmax(out_logit7)

model8 = models.__dict__[model_name8]()
out_logit8 = model8.net(input=input_layer, class_dim=class_dim)
out8 = fluid.layers.softmax(out_logit8)

fluid.io.load_persistables(exe, pretrained_model, main_program=main_programs)
print('ok')
init_prog(main_programs)
eval_program = main_programs.clone(for_test=True)

label = fluid.layers.data(name="label", shape=[1], dtype='int64')
y = fluid.layers.data(name="y", shape=[8], dtype='int64')
out_logits = (out_logit1[:, :121]*y[0]+out_logit2*y[1]+out_logit3*y[2]+out_logit4*y[3]+out_logit5 *
              y[4]+out_logit6*y[5]+out_logit7*y[6]+out_logit8*y[7])/(y[0]+y[1]+y[2]+y[3]+y[4]+y[5]+y[6]+y[7])
out = fluid.layers.softmax(out_logits)
loss = fluid.layers.cross_entropy(input=out, label=label)
gradients = fluid.gradients(targets=loss, inputs=[input_layer])[0]


def inference(img):
    result1, result2, result3, result4, result5, result6, result7, result8 = exe.run(eval_program,
                                                                                     fetch_list=[
                                                                                         out1, out2, out3, out4, out5, out6, out7, out8],
                                                                                     feed={'image': img})
    result1 = result1[0, :121]
    pred1 = np.argmax(result1)
    result2 = result2[0]
    pred2 = np.argmax(result2)
    result3 = result3[0]
    pred3 = np.argmax(result3)
    result4 = result4[0]
    pred4 = np.argmax(result4)
    result5 = result5[0]
    pred5 = np.argmax(result5)
    result6 = result6[0]
    pred6 = np.argmax(result6)
    result7 = result7[0]
    pred7 = np.argmax(result7)
    result8 = result8[0]
    pred8 = np.argmax(result8)

    result = (result1+result2+result3+result4 +
              result5+result6+result7+result8)/8
    pred_label = np.argmax(result)
    pred_score = result[pred_label].copy()
    return pred_label, pred_score, pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8


def SI_NI_FGSM(o, y, mlabel, step_size=16.0/256, epsilon=16.0/256, isTarget=False, target_label=0, use_gpu=False, T=10, u=1.0):

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    target_label = mlabel

    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)
    g = np.zeros(shape=[1, 3, 224, 224], dtype='float32')
    adv = o
    for i in range(T):

        # 计算梯度
        grad = exe.run(main_programs, fetch_list=[gradients.name],
                       feed={'image': adv, 'label': target_label, 'y': y})
        grad = grad[0]
        grad = grad / np.mean(np.abs(grad), (1, 2, 3), keepdims=True)
        g = u * g + grad
        g = g / np.mean(np.abs(g), (1, 2, 3), keepdims=True)

        adv = adv+np.clip(np.round(g), -20, 20)*step_size/T

    return adv


def attack_nontarget_by_SINIFGSM(img, src_label):

    pred_label, pred_score, pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8 = inference(
        img)
    step = 8.0/256.0
    eps = 32.0/256.0
    adv = img
    label_lists = [pred_label, pred1, pred2,
                   pred3, pred4, pred5, pred6, pred7, pred8]
    m = 0
    y = np.array([2, 1, 1, 1, 1, 1, 1, 1])
    while src_label in label_lists[:7]:
        m = m+1
        for i in range(len(label_lists)-1):
            if label_lists[i+1] == src_label:
                y[i] = 3
            y[7] = 1
        adv = SI_NI_FGSM(o=adv, y=y, mlabel=pred_label, step_size=step, epsilon=eps,
                         isTarget=False, target_label=0, use_gpu=use_gpu, T=5, u=0.8)

        label_lists[0], pred_score, label_lists[1], label_lists[2], label_lists[3], label_lists[
            4], label_lists[5], label_lists[6], label_lists[7], label_lists[8] = inference(adv)
        print('labels:{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}'.format(
            src_label, label_lists[0], label_lists[1], label_lists[2], label_lists[3], label_lists[4], label_lists[5], label_lists[6], label_lists[7], label_lists[8]))
        if m > 10:
            break
    print("Test-score: {0}, class {1}".format(pred_score, pred_label))
    adv_img = tensor2img(adv)
    return adv_img


def get_original_file(filepath):
    with open(filepath, 'r') as cfile:
        full_lines = [line.strip() for line in cfile]
    cfile.close()
    original_files = []
    for line in full_lines:
        label, file_name = line.split()
        original_files.append([file_name, int(label)])
    return original_files


def gen_adv():
    mse = 0
    original_files = get_original_file(
        'input_image/' + val_list)

    for filename, label in original_files:
        img_path = input_dir + filename.split('.')[0]+'.png'
        print("Image: {0} ".format(img_path))
        img = process_img(img_path)
        adv_img = attack_nontarget_by_SINIFGSM(img, label)
        image_name, image_ext = filename.split('.')
        # Save adversarial image(.png)
        save_adv_image(adv_img, output_dir+image_name+'.png')

        org_img = tensor2img(img)
        score = calc_mse(org_img, adv_img)
        mse += score
    print("ADV {} files, AVG MSE: {} ".format(
        len(original_files), mse / len(original_files)))
