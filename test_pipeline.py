import unittest
from model_pipeline import prepare_data, train_model, evaluate_model
from model_pipeline import save_data, load_data


class TestPipeline(unittest.TestCase):

    def test_prepare_data(self):
        X_train, X_test, y_train, y_test = prepare_data("churn-bigml-80.csv")
        self.assertEqual(X_train.shape[0], 0.8 * (X_train.shape[0] + X_test.shape[0]))

    def test_training(self):
        X_train, X_test, y_train, y_test = prepare_data("churn-bigml-80.csv")
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)

    def test_save_and_load_data(self):
        X_train, X_test, y_train, y_test = prepare_data("churn-bigml-80.csv")
        save_data(X_train, X_test, y_train, y_test)
        X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded = load_data()
        self.assertEqual(X_train.shape, X_train_loaded.shape)


if __name__ == "__main__":
    unittest.main()
