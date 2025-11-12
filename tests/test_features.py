from src.pipeline.features import build_feature_frame
import pandas as pd

def test_feature_build_minimal():
    df = pd.DataFrame({
        "timestamp":["2024-01-01T00:00:00","2024-01-01T00:05:00"],
        "customer_id":[1,1],
        "merchant_id":[5,6],
        "amount":[100.0,150.0],
        "country":["IN","IN"],
        "mcc":["5411","5411"],
        "channel":["POS","ECOM"],
        "device_id":["d1","d1"],
        "label":[0,0]
    })
    out = build_feature_frame(df)
    assert "hour" in out.columns
    assert "tx_count_1h" in out.columns
    assert "merchant_risk" in out.columns
