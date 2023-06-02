
import models
from utils.stylegan2_utils import StyleGAN2SampleGenerator
from utils.segmentation_utils import FaceSegmentation, StuffSegmentation, GANLinearSegmentation
#from utils.loss_utils import AbstractLoss
from utils.prompt_utils import PromptAnalyzer
from eclipsed import eCLIPsed

DEFAULT_TRAINING_PARAMS = {
    'model': 'ffhq',
    'segmentation_model': 'face_bisenet',
    'segmentation_type': 'FaceSegmentation',
    'segmentation_parts': [
        ["skin", ["skin"]],
        ["left brow", ["l_brow"]],
        ["right brow", ["r_brow"]],
        ["left eye", ["l_eye"]],
        ["right eye", ["r_eye"]],
        ["eyes", ["eye_g"]],
        ["left ear", ["l_ear"]],
        ["right ear", ["r_ear"]],
        ["earrings", ["ear_r"]],
        ["nose", ["nose"]],
        ["mouth", ["mouth"]],
        ["upper lip", ["u_lip"]],
        ["lower lip", ["l_lip"]],
        ["neck", ["neck"]],
        ["lower neck", ["neck_l"]], 
        ["clothes", ["cloth"]],
        ["hair", ["hair"]],
        ["hat", ["hat"]]
    ],


    'latent_spaces': ['W+'],
    'loss_functions': ['L2'],
    'mask_aggregations': ['average'],
    'nums_latent_dirs': [1],
    'learning_rate': 0.001,
    'min_abs_alpha_value': 0.1,
    'min_alpha_value': 0.0,
    'max_alpha_value': 1.0, 
    'batch_size': 2,
    'device': 'cuda'
}


def train_eCLIPsed_model(training_config,
                      analyzer: PromptAnalyzer,
                      single_image_seed: int = None,
                      clip_model_name: str = 'ViT-B/32',
                      clip_loss_lambda: float = 1.0,
                      localization_loss_lambda: float = 1.0,
                      gamma_correlation: float = 1.0,
                      unit_norm=False,
                      min_abs_alpha_value=0.1,
                      snapshot_interval = 200):
    
    G2 = models.get_model("stylegan2", f"pretrained/{training_config['model']}.pkl")
    stylegan2_sample_generator = StyleGAN2SampleGenerator(G=G2, device=training_config['device'])

    exp_dir = 'out'
    segmentation_part_idx = analyzer.part_idx

    seg_model = models.get_model(training_config['segmentation_model'], f"pretrained/{training_config['segmentation_model']}.pth")

    if (training_config['segmentation_type'] == 'FaceSegmentation'):
        segmentation = FaceSegmentation(face_bisenet=seg_model, device=training_config['device'])


    lr = training_config['learning_rate']
    min_alpha_value = training_config['min_alpha_value']
    max_alpha_value = training_config['max_alpha_value']
    min_abs_alpha_value = training_config['min_abs_alpha_value']
    batch_size = training_config['batch_size']

    onehot_temperature = 0.001
    localization_layers = list(range(1, 18))
    localization_layer_weights = None

    for latent_space in training_config['latent_spaces']:
        for loss_function in training_config['loss_functions']:
            for mask_aggregation in training_config['mask_aggregations']:

                for num_latent_dirs in training_config['nums_latent_dirs']:
                    for part_name, face_parts in [training_config['segmentation_parts'][segmentation_part_idx]]:

                        log_dir = f'{exp_dir}/eclipsed_stylegan2_' + training_config['model'] \
                            + f'/{latent_space}_{loss_function}_{mask_aggregation}/{num_latent_dirs}D/' \
                            +training_config['segmentation_model'] + f'/{analyzer.prompt_dir}'
                        eclipsed = eCLIPsed(device=training_config['device'],
                                    localization_layers=localization_layers,
                                    semantic_parts=face_parts,
                                    localization_loss_function=loss_function,
                                    localization_layer_weights=localization_layer_weights,
                                    mode='foreground',
                                    mask_aggregation=mask_aggregation,
                                    n_layers=18,
                                    latent_dim=512,
                                    num_latent_dirs=num_latent_dirs,
                                    learning_rate=lr,
                                    batch_size=batch_size,
                                    unit_norm=unit_norm,
                                    latent_space=latent_space,
                                    onehot_temperature=onehot_temperature,
                                    min_alpha_value=min_alpha_value,
                                    max_alpha_value=max_alpha_value,
                                    min_abs_alpha_value=min_abs_alpha_value,
                                    log_dir=log_dir
                                    )

                        eclipsed.fit(stylegan2_sample_generator, segmentation, #clip_model,
                                  analyzer = analyzer,
                                  num_batches=200 * num_latent_dirs,
                                  single_image_seed = single_image_seed,
                                  clip_model_name=clip_model_name,
                                  clip_loss_lambda=clip_loss_lambda,
                                  localization_loss_lambda=localization_loss_lambda,
                                  gamma_correlation=gamma_correlation,
                                  num_lr_halvings=3,
                                  batch_size=training_config['batch_size'],
                                  pgbar=True, summary=True, snapshot_interval=snapshot_interval)
                        eclipsed.save()
