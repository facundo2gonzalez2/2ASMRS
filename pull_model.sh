# Pull base model
rm -rf /home/facu/Documents/2ASMRS/model/inference_models/base_model/base_model_no_beta
ssh venus "tar czf - -C /home/fgbarnator/2ASMRS/model/inference_models/base_model base_model_no_beta" | tar xzvf - -C /home/facu/Documents/2ASMRS/model/inference_models/base_model

# Pull instruments from checkpoint models
# ssh venus "tar czf - -C /home/fgbarnator/2ASMRS/model/inference_models/instruments_from_checkpoint bass_from_checkpoint_no_beta" | tar xzvf - -C /home/facu/Documents/2ASMRS/model/inference_models/instruments_from_checkpoint
# ssh venus "tar czf - -C /home/fgbarnator/2ASMRS/model/inference_models/instruments_from_checkpoint guitar_from_checkpoint_no_beta" | tar xzvf - -C /home/facu/Documents/2ASMRS/model/inference_models/instruments_from_checkpoint
# ssh venus "tar czf - -C /home/fgbarnator/2ASMRS/model/inference_models/instruments_from_checkpoint voice_from_checkpoint_no_beta" | tar xzvf - -C /home/facu/Documents/2ASMRS/model/inference_models/instruments_from_checkpoint
# ssh venus "tar czf - -C /home/fgbarnator/2ASMRS/model/inference_models/instruments_from_checkpoint piano_from_checkpoint_no_beta" | tar xzvf - -C /home/facu/Documents/2ASMRS/model/inference_models/instruments_from_checkpoint

# Pull instruments from scratch models
rm -rf /home/facu/Documents/2ASMRS/model/inference_models/instruments_from_scratch/piano_from_scratch_no_beta
rm -rf /home/facu/Documents/2ASMRS/model/inference_models/instruments_from_scratch/guitar_from_scratch_no_beta
rm -rf /home/facu/Documents/2ASMRS/model/inference_models/instruments_from_scratch/voice_from_scratch_no_beta
rm -rf /home/facu/Documents/2ASMRS/model/inference_models/instruments_from_scratch/bass_from_scratch_no_beta

ssh venus "tar czf - -C /home/fgbarnator/2ASMRS/model/inference_models/instruments_from_scratch piano_from_scratch_no_beta" | tar xzvf - -C /home/facu/Documents/2ASMRS/model/inference_models/instruments_from_scratch
ssh venus "tar czf - -C /home/fgbarnator/2ASMRS/model/inference_models/instruments_from_scratch guitar_from_scratch_no_beta" | tar xzvf - -C /home/facu/Documents/2ASMRS/model/inference_models/instruments_from_scratch
ssh venus "tar czf - -C /home/fgbarnator/2ASMRS/model/inference_models/instruments_from_scratch voice_from_scratch_no_beta" | tar xzvf - -C /home/facu/Documents/2ASMRS/model/inference_models/instruments_from_scratch
ssh venus "tar czf - -C /home/fgbarnator/2ASMRS/model/inference_models/instruments_from_scratch bass_from_scratch_no_beta" | tar xzvf - -C /home/facu/Documents/2ASMRS/model/inference_models/instruments_from_scratch

# Pull data_instruments
rm -rf /home/facu/Documents/2ASMRS/model/data_instruments
ssh venus "tar czf - -C /home/fgbarnator/2ASMRS/model data_instruments" | tar xzvf - -C /home/facu/Documents/2ASMRS/model/