# Unsupervised approach - Convolutional Autoencoder 

## Load packages and dependencies
### _If runned on university computers_
#### 1 Add course module
`module add courses/TSBB19`

#### 2 Do once
`conda init bash`

#### 3 Use for example conda virtual environment
`conda activate tsbb19`

### _Your own computer_
#### 1 Create a new environment and install packages
`conda create --name [my_env] --file requirements.txt`

#### 2 Activate new environment
`conda activate [my_env]`


## Run the model

### Models are located in autoencoder.py
### Dataloaders are located in dataset_loader.py
### Main file to be runned i main.ipynb

#### In main file
#### 1. Choose dataset
- **Dataset 1**, Uncomment the following lines
    - `anomaly_dir = './../data/desert/anomaly/'`
    - `normal_dir = './../data/desert/normal/'`
- **Dataset 2**, Uncomment the following lines
    - `anomaly_dir = './../data/santa_clarita/anomaly/'`
    - `normal_dir = './../data/santa_clarita/normal/'`
- **New dataset**, or select the paths to anomalies and normals to your new customized dataset.

#### 2. Decide if you want to run on images or videos by setting SEQ = True for videos and SEQ = False for images
#### 3. Run cells until cell 5: Loss functions
Decide which loss function you want to use. (MSE and BCE has shown most promising results) 
#### 4. Run cell untill cell 9:
Decide learning rate, amplify and latent space dimensions
#### 5. Run cells until cell 10: Choose your autoencoder configuration
- **CAE4**: 4 convolutional layers, no max pooling
- **CAE3**: 3 convolutiona layers, no max pooling
- **CAEM**: 3 convolutional layers, max pooling
- **CAE0**: 3 convolutional layers, no max pooling, no linear code layer

#### 6. Run cells until cell 13: Decide number of training epochs
#### 7. Run cells until cell 34: Analyze ROC curve and set threshold 
