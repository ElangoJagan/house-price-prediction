from src.utils import Utils

test_obj = {'name': 'House_price_pred', 'version': 1}

Utils.save_object("artifacts/test.pkl", test_obj)
loaded = Utils.load_object("artifacts/test.pkl")

print(loaded)