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

    return file

human_det_video =  gr.Interface(DetVideo, 
    [gr.inputs.Video(type="filepath", label="Input")], 
    gr.outputs.Video(type="numpy", label="Output"),
    description="行人检测-视频",
    article="行人检测",
    enable_queue=True
    )

demo = gr.TabbedInterface([human_det_image, human_attr_image, human_det_video], ["行人检测-图片", "行人属性-图片","行人检测-视频"])

if __name__ == "__main__":
    demo.launch(share=True)