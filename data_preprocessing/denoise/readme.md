Clone here the repository for MPRNet: https://github.com/swz30/MPRNet.git

Place the denoise.sh utility bash script inside of the cloned MPRNet folder, to run denoising on several input folders.

To use pretrained models, download and place them in pretrained_models folder of each task (Denoising, Deraining):

To test the pre-trained models of [Deblurring](https://drive.google.com/file/d/1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb/view?usp=sharing), [Deraining](https://drive.google.com/file/d/1O3WEJbcat7eTY6doXWeorAbQ1l_WmMnM/view?usp=sharing), [Denoising](https://drive.google.com/file/d/1LODPt9kYmxwU98g96UrRA0_Eh5HYcsRw/view?usp=sharing) on your own images, run 
```
python demo.py --task Task_Name --input_dir path_to_images --result_dir save_images_here
```
Here is an example to perform Deblurring:
```
python demo.py --task Deblurring --input_dir ./samples/input/ --result_dir ./samples/output/
```
