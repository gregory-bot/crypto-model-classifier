import pandas as pd

data = pd.read_csv(r"C:\Users\HP\Desktop\crypto-classifier\data\processed\crypto_data_feature_engineered.csv")

data["future_return"] = data["close"].pct_change().shift(-1)

def label(row):
    if row["future_return"] > 0.02:
        return 2      
    elif row["future_return"] < -0.02:
        return 0      
    else:
        return 1        

data["label"] = data.apply(label, axis=1)
data['label_name'] = data['label'].map({0:'SELL', 1:'HOLD', 2:'BUY'})

print(data.head())
#save the labeled data to a new csv file
data.to_csv(r"C:\Users\HP\Desktop\crypto-classifier\data\processed\crypto_data_labeled.csv", index=False)
print("Labeled data saved to crypto_data_labeled.csv")
