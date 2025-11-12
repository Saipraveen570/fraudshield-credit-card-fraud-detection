from src.serving.schema import Transaction

def test_schema_fields():
    tx = Transaction(customer_id=1, merchant_id=2, amount=99.0, country="IN", mcc="5411", channel="POS", device_id="d5")
    assert tx.amount == 99.0
