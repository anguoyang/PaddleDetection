import gradio as gr
import cv2
import os
import subprocess

def TestImage(file):
    subprocess.Popen(['python', 'deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml --image_file=test_image.jpg --device=cpu'], stdin = subprocess.PIPE, stdout=subprocess.PIPE)
    #subprocess.run("python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml --image_file=test_image.jpg --device=cpu")
    #im = cv2.imread(file, cv2.IMREAD_COLOR) 

    return file

human_image =  gr.Interface(TestImage, 
    [gr.inputs.Image(type="filepath", label="Input")], 
    gr.outputs.Image(type="numpy", label="Output"),
    description="行人检测-图片",
    article="行人检测",
    enable_queue=True
    )

def TestVideo(file):
    #os.system("python deploy/pipeline/pipeline.py " + "--config " + "deploy/pipeline/config/infer_cfg_pphuman.yml " + "--image_file=" + file  " --device=cpu")

    return file

human_video =  gr.Interface(TestVideo, 
    [gr.inputs.Video(type="filepath", label="Input")], 
    gr.outputs.Video(type="numpy", label="Output"),
    description="行人检测-视频",
    article="行人检测",
    enable_queue=True
    )

demo = gr.TabbedInterface([human_image, human_video], ["行人检测-图片", "行人检测-视频"])

if __name__ == "__main__":
    demo.launch(share=True)