import os
from os import path
from zipfile import ZipFile
from tqdm import tqdm
import requests
import shutil

pretrained_dir = 'pretrained'

model_urls = {
    # Not working due to Tensorflow error ?
    'beetles': 'https://drive.google.com/uc?id=1BOluDQSMzKLgJ3tipAD3tfq5p6AEv_-C&confirm=t&uuid=e8b7e5d7-404e-41e6-b7c2-eb881adde69e',
    'ukiyoe': 'https://drive.google.com/uc?id=1_QysUKfed1-_x9e5off2WWJKp1yUcidu&confirm=t&uuid=d9027c30-ce7c-4edb-85fc-010e2548e623&at=ANzk5s4NAOyrXSkpVRvFMBhxESOr:1679599758528',

    # Working
    'churches': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-church-config-f.pkl',
    'horse': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-horse-config-f.pkl',
    'metfaces': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl'
}

def fetch_model(name: str) -> str:
    """
    Checks if a model has been downloaded, and downloads it if needed
    Returns the path of the model so it can be passed to `models.get_model()`
    """
    if (not path.exists(pretrained_dir)):
        os.makedirs(pretrained_dir)

    model_path = f'{pretrained_dir}/{name}.pkl'

    if (not path.exists(model_path)):
        response = requests.get(model_urls[name], stream=True)
        with open(model_path, 'bx') as f:
            for data in tqdm(response.iter_content(1024), desc=f'Downloading {name} model from {model_urls[name]}'):
                f.write(data)

    return model_path


def fetch_LELSD_requirements():
    """
    Download the LELSD repo as a ZIP, and extract the correct files from it
    """
    url = 'https://github.com/IVRL/LELSD/archive/refs/heads/main.zip'
    zfname = 'LELSD.zip'
    top_level_dir = 'LELSD-main'

    req_folders = [
        'models',
        'utils'
    ]

    if (not path.exists(zfname)):
        response = requests.get(url, stream=True)
        with open(zfname, 'bx') as f:
            for data in tqdm(response.iter_content(1024), desc='Downloading repo'):
                f.write(data)

    # Unzip archive
    to_unzip = []
    for folder in req_folders:
        if (not path.exists(folder)):
            to_unzip.append(f'{top_level_dir}/{folder}')
    
    with ZipFile(zfname) as zip_file:
        for zip_info in tqdm(zip_file.infolist(), desc="Unzipping required folders from archive"):
            if (any([(folder in zip_info.filename) for folder in to_unzip])):
                zip_file.extract(zip_info)
                shutil.move(zip_info.filename, zip_info.filename[zip_info.filename.index('/') + 1:])

    print('\nCleaning up...')
    if (path.exists(top_level_dir)):
        shutil.rmtree(top_level_dir)

    print('Done.')