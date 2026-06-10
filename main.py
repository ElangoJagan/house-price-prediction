from src.utils import utils

test_obj = {'name': 'House_price_pred', 'version': 1}

utils.save_object("artifacts/test.pkl", test_obj)
loaded = utils.load_object("artifacts/test.pkl")

print(loaded)