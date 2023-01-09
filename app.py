import gradio as gr
import cv2
import os
import subprocess

def DetImage(file):
    subprocess.run(['python', 'deploy/pipeline/pipeline.py', '--config', 'deploy/pipeline/config/infer_cfg_pphuman.yml', '--image_file='+file, '--device=cpu'], stdin = subprocess.PIPE, stdout=subprocess.PIPE)
    #subprocess.run("python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml --image_file=test_image.jpg --device=cpu")
    
    im = cv2.imread('output/'+os.path.basename(file), cv2.IMREAD_COLOR) 
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im

human_det_image =  gr.Interface(DetImage, 
    [gr.inputs.Image(type="filepath", label="Input")], 
    gr.outputs.Image(type="numpy", label="Output"),
    description="行人检测-图片",
    article="行人检测",
    enable_queue=True
    )

def AttrImage(file):
    subprocess.run(['python', 'deploy/pipeline/pipeline.py', '--config', 'deploy/pipeline/config/examples/infer_cfg_human_attr.yml', '--image_file='+file, '--device=cpu'], stdin = subprocess.PIPE, stdout=subprocess.PIPE)
    #subprocess.run("python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml --image_file=test_image.jpg --device=cpu")
    
    im = cv2.imread('output/'+os.path.basename(file), cv2.IMREAD_COLOR) 
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im

human_attr_image =  gr.Interface(AttrImage, 
    [gr.inputs.Image(type="filepath", label="Input")], 
    gr.outputs.Image(type="numpy", label="Output"),
    description="行人属性-图片",
    article="行人属性",
    enable_queue=True
    )


def DetVideo(file):
    #os.system("python deploy/pipeline/pipeline.py " + "--config " + "deploy/pipeline/config/infer_cfg_pphuman.yml " + "--image_file=" + file  " --device=cpu")
    subprocess.run(['python', 'deploy/pipeline/pipeline.py', '--config', 'deploy/pipeline/config/infer_cfg_pphuman.yml', '--video_file='+file, '--device=cpu'], stdin = subprocess.PIPE, stdout=subprocess.PIPE)
    return 'output/'+os.path.basename(file)
    #return

human_det_video =  gr.Interface(DetVideo, 
    [gr.inputs.Video(type="mp4", label="Input")], 
    gr.outputs.Video(type="mp4", label="Output"),
    description="行人检测-视频",
    article="行人检测",
    enable_queue=True
    )

def TrackingVideo(file):
    #os.system("python deploy/pipeline/pipeline.py " + "--config " + "deploy/pipeline/config/infer_cfg_pphuman.yml " + "--image_file=" + file  " --device=cpu")
    subprocess.run(['python', 'deploy/pptracking/python/mot_jde_infer.py', '--model_dir=output_inference/fairmot_hrnetv2_w18_dlafpn_30e_576x320', '--video_file='+file, '--device=cpu'], stdin = subprocess.PIPE, stdout=subprocess.PIPE)
    return 'output/'+os.path.basename(file)
    #return

tracking_video =  gr.Interface(TrackingVideo, 
    [gr.inputs.Video(type="mp4", label="Input")], 
    gr.outputs.Video(type="mp4", label="Output"),
    description="行人跟踪-视频",
    article="行人跟踪",
    enable_queue=True
    )


def snap(image):
    return [image]

tracking_webcam = gr.Interface(
    snap,
    [gr.Image(source="webcam", tool=None)],
    ["image"],
    description="行人跟踪-摄像头",
    article="行人跟踪",
    enable_queue=True
)

def DetImage_Car(file):
    subprocess.run(['python', 'deploy/pipeline/pipeline.py', '--config', 'deploy/pipeline/config/infer_cfg_pphuman.yml', '--image_file='+file, '--device=cpu'], stdin = subprocess.PIPE, stdout=subprocess.PIPE)
    #subprocess.run("python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml --image_file=test_image.jpg --device=cpu")
    
    im = cv2.imread('output/'+os.path.basename(file), cv2.IMREAD_COLOR) 
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im

car_det_image =  gr.Interface(DetImage_Car, 
    [gr.inputs.Image(type="filepath", label="Input")], 
    gr.outputs.Image(type="numpy", label="Output"),
    description="车辆计数-图片",
    article="车辆计数",
    enable_queue=True
    )


demo_person = gr.TabbedInterface([human_det_image, human_attr_image, tracking_video, tracking_webcam], ["行人检测-图片", "行人属性-图片","行人跟踪-视频","行人跟踪-摄像头"])
demo_car = gr.TabbedInterface([car_det_image], ["车辆计数-图片"])
demo = gr.TabbedInterface([demo_person, demo_car], ["行人", "车辆"])
if __name__ == "__main__":
    demo.launch(share=True)