# Models and nomenclature

## Training for experiments

All in `model/experiments_models`

- base_model_beta_variation: 5 versions for beta in `[0.01, 0.001, 0.0001, 0.00001, 0]`. 

- experiment_latent_dim_{instrument}
    - beta_vae_latentdim_{n}, 5 versions each
    - vae_latentdim_{n} (2,3,4,6,8), 5 versions each

- experiment_latent_dim_base_model
    - vae_latentdim_{n} (2,3,4,6,8), 5 versions each
    - beta_vae_latentdim_{n} (2,3,4,6,8), 5 versions each

## Training for inference

All in `model/inference_models`

- base_model
    - base_model_no_beta: unique version (version_0)
    - base_model_beta_{b}: unique version (version_0) with b=0.001

- instruments_from_checkpoint
    - {instrument}_from_checkpoint_no_beta
    - {instrument}_from_checkpoint_beta_{b} with b=0.001

- instruments_from_scratch
    - {instrument}_from_scratch_no_beta
    - {instrument}_from_scratch_beta_{b} with b=0.001
