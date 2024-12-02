# ECE1508_Gas_Price_Pred
ECE1508 Gas Price Prediction Project by Jean-Louis Keyrouz and Adham Ragab

## Running the model
To run the model:
```bash
python3.11 final_model_shap.py
```

## Configuration 

### Final Training Parameters
    NUM_EPOCHS_FINAL = 30      # Number of epochs for final training
    PATIENCE_FINAL = 3         # Early Stopping patience

### Data Parameters
    TRAIN_WEEKS = 327          # Number of weeks for training
    VAL_WEEKS = 52             # Number of weeks for validation

### Hyperparameters
    best_seq_length = 2        # Example sequence length
    best_hdim = 15             # Example hidden dimension
    best_batch_size = 8        # Example batch size

## run_model_val is for hyperparameter tuning with similar parameters