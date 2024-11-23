# ECE1508_Gas_Price_Pred
ECE1508 Gas Price Prediction Project by Jean-Louis Keyrouz and Adham Ragab

## Running the model
To run the model:
```bash
python3.11 run_model.py
```

## Factors to play around with:
- Size of prediction window (time sequence length)
- Batch size (Total period considered)
- Relationship between number of epochs and hidden state size
- Inclusion/Exclusion of different features (e.g. include date, remove CPI, etc)
- Learning rate with optimizer
