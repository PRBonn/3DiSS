# Towards Generating Realistic 3D Semantic Training Data for Autonomous Driving

## Dependencies

Installing python (we have used python 3.9) packages pre-requisites:

`sudo apt install build-essential python3-dev libopenblas-dev`

`pip install -r requirements.txt`

Installing MinkowskiEngine:

`pip install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps`

To setup the code run the following command on the code main directory:

`pip install -U -e .`

## Conda Installation

You can also install the dependencies with conda environment:
`conda create --name 3diss python=3.9 && conda activate 3diss`

Then again, installing python packages pre-requisites:

`sudo apt install build-essential python3-dev libopenblas-dev`

`pip install -r requirements.txt`

And installing MinkowskiEngine:

`pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps`
**NOTE**: At the moment, MinkowskiEngine is not compatible with python 3.10+, see this [issue](https://github.com/NVIDIA/MinkowskiEngine/issues/526#issuecomment-1855119728)

## SemanticKITTI Dataset

The SemanticKITTI dataset has to be download from the official [site](http://www.semantic-kitti.org/dataset.html#download) and extracted in the following structure:

```
./lidiff/
└── Datasets/
    └── SemanticKITTI
        └── dataset
          └── sequences
            ├── 00/
            │   ├── velodyne/
            |   |       ├── 000000.bin
            |   |       ├── 000001.bin
            |   |       └── ...
            │   └── labels/
            |       ├── 000000.label
            |       ├── 000001.label
            |       └── ...
            ├── 08/ # for validation
            ├── 11/ # 11-21 for testing
            └── 21/
                └── ...
```

## Ground truth generation

To generate the ground complete scenes you can run the `sem_map_from_scans.py` script. This will use the dataset scans and poses to generate the sequence map to be used as ground truth during training:

```
python tools/sem_map_from_scans.py --path Datasets/SemanticKITTI/dataset/sequences/
```

Once the sequences map is generated you can then train the VAE and diffusion models

## VAE Training

To train the VAE you can run the following command:

`python vae_train.py`

In case you want to change the VAE training config you can edit the `config/vae_config.yaml` file. After the VAE is trained you can run the VAE refinement training with:

`python vae_train.py --config config/vae_refine.yaml`

Which will do the refinement training **only** on the VAE decoder weights.

## Diffusion Training

After the VAE is trained you can run the folowing command to train the unconditional DDPM:

`python diff_train.py --vae_weights experiments/VAE/default/version_0/checkpoint/VAE_epoch\=49.ckpt`

By default, the diffusion training is set to be trained as an unconditional DDPM. For the LiDAR scan conditioning training you can run:

`python diff_train.py --vae_weights experiments/VAE/default/version_0/checkpoint/VAE_epoch\=49.ckpt --config config/diff_cond_config.yaml --condition single_scan`

Which will train the model conditioned to the dataset LiDAR point clouds.

## Model Weights

You can download the trained model weights from the following links:

- VAE: Coming soon!
- VAE Refined: Coming soon!
- Unconditional DDPM: Coming soon!
- Conditional DDPM: Coming soon!

## Diffusion Inference

For running the unconditional scene generation we provide a pipeline where both the diffusion and VAE trained models are loaded and used to generate a novel scene. You can run the pipeline with the command:

`python tools/diff_pipeline.py --diff DIFF_CKPT --vae VAE_REFINE_CKPT -T DENOISING_STEPS

To run the pipeline for the conditional scene generation you can run:

`python tools/diff_pipeline.py --path PATH_TO_SCANS --diff DIFF_CKPT --vae VAE_REFINE_CKPT -T DENOISING_STEPS -s CONDITIONING_WEIGHT --condition single_scan`

Where the LiDAR point clouds used as condition should be placed in the `diss/Datasets/test/` directoty. We provide one scan as example in `diss/Datasets/test/` so you can directly test it out with our trained model by just running the code above. By default, the `DENOISING_STEPS` and `CONDITIONING_WEIGHT` parameters are set respectively to `1000` and `2.0`, the same as used in the paper. 

## Citation

If you use this repo, please cite as :

```bibtex
@article{nunes2025arxiv,
    author = {Lucas Nunes and Rodrigo Marcuzzi and Jens Behley and Cyrill Stachniss},
    title = {{Towards Generating Realistic 3D Semantic Training Data for Autonomous Driving}},
    journal = arxiv,
    year = {2025}
}




