# Cartoonizing Portrait Image

# 1. Requirements
```angular2html
pytorch >= 1.5
matplotlib
torchvision >= 0.5
cv2
PIL
tqdm
```

# 2. Set up
#### 2.1. Clone this project
```angular2html
git clone https://github.com/HoSyTuyen/APGAN
cd APGAN
```

#### 2.2. Create environment (with python 3.5 or above) and install requirements as in section 1

#### 2.3. Dataset 
```angular2html
mkdir data
```
Download load dataset [selfie2anime](https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view) or [CartoonFace10K](https://drive.google.com/file/d/19-vzVBGNorcF3zxWZPeRio-S1JmTYncf/view?usp=sharing). Put and unzip in `data` folder \
The dataset folder should follow:
```angular2html
data
|_____name_of_dataset
        |______trainA
        |        |_______image1.jpg
        |        |_______imgaek.jpg
        |______trainB
        |        |_______image1.jpg
        |        |_______imgaev.jpg
        |______testA
        |        |_______image1.jpg
        |        |_______imgaen.jpg
        |______testB
                 |_______image1.jpg
                 |_______imgaem.jpg


```
You can train on your own data with the same structure



# 2. Training
```angular2html
mkdir checkpoint
```
Download the [pre-trained for HED](https://drive.google.com/file/d/1GBytgs63qzCUyPrvj8gQuR2EYdMXcpWP/view?usp=sharing) and put in `checkpoint` folder

- Run this command for training
```angular2html
python CFGAN.py --name "name_of_experiment"
                --batch_size 8
                --con_lambda 30
                --adv_lambda 1
                --hed_lambda 5
                --data_path "data/name_of_dataset"
```
For other arguments, please find in `CFGAN.py`

- After training you will attain a checkpoint in .pkl format

# 2. Test
- Run this command for testing
```angular2html
python test.py --pre_trained_model "path_to_checkpoint.pkl" 
               --image_dir "path_to_input_folder" 
               --output_image_dir "path_to_output_folder"
```
For other arguments, please find in `test.py`
