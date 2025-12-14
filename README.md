# oldNepali-synthetic-data-generation

This repository is for creating images (printed text) for nepali. We use raw texts, and convert them to images using the PIL library. The purpose of doing is because we wanted to teach contexual and texual information to a ML model. 

The texts we used here were extracted and combined from various nepali textbooks (history, political, religious, literary) form [https://archive.org/](https://archive.org/). We extracted a total of 105,000 text lines. We also chose the specific line lengths, and image sizes to resemble our main dataset line level text lines, which were wide. Similary, some features like adding a `dot' at 20 percent of samples were also done since our text lines have dots as word boundaries in many samples. 

Once these lines were generated, we also add noisy augmentations to the generated images. This was done with the goal to help the decoder also breakdown and learn the text in noisier data samples like blurs, ink blurs, rotating some characters etc. With this, we tried to make the samples look as close as possible to the handwritten texts we had in our original dataset.

# Code
We have two scripts to run for the data generation here:

## Step 1: Generating text to images
This part is where we take the corpus_105.txt and convert the lines here to texts using different font styles, font sizes, angles etc. These variations are also added to add variations. 
```
python src/1_generate_images.py
```

## Step 2: Noisy image augmentations
With the generated images in step 1 (at data/oldNepaliSynthetic_105k), in this step we add noise. To do this:
```
python src/2_noisy_augmentations.py
```

Once these are completed, you ca view the images at data/oldNepaliSynthetic_105k_vnoisy. The labels.json is also saved with image paths and text labels, which can be further used for model training. 

# Dataset
The generated dataset can also be downloaded at: [dataset-link](https://drive.google.com/drive/folders/1Iq40Aejr5B1t022q8r7RwiFbwxAdxlHL?usp=sharing)