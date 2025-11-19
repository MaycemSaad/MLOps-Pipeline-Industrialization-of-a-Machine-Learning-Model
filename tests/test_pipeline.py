# test_pipeline.py
import unittest
import pandas as pd
import os
import datetime
from model_pipeline import prepare_data, train_model, evaluate_model
from sklearn.linear_model import LogisticRegression

class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n📝 Initialisation des tests...")
        if not os.path.exists('logs'):
            os.makedirs('logs')
        if not os.path.exists('reports'):
            os.makedirs('reports')
        
        cls.log_file = f'logs/test_logs.txt'
        cls.report_file = f'reports/report_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.html'

    def setUp(self):
        # Données factices pour les tests
        print("\n🔄 Préparation des données factices...")
        self.data_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'Churn': [0, 1, 0, 1, 0]
        })
        self.data_test = pd.DataFrame({
            'feature1': [6, 7, 8],
            'feature2': [1, 2, 3],
            'Churn': [1, 0, 1]
        })
        
        self.data_train.to_csv('train_temp.csv', index=False)
        self.data_test.to_csv('test_temp.csv', index=False)

    def log_result(self, test_name, result):
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.datetime.now()} - {test_name} : {'✅' if result else '❌'}\n")

    def test_prepare_data(self):
        print("\n🔎 Test de préparation des données...")

        # 🔹 On passe un seul fichier car prepare_data ne prend qu'un argument
        data, data_test, _, _ = prepare_data('train_temp.csv')

        print("\n📌 DataFrame préparé :")
        print(data.head())

        print("\n📏 Dimensions obtenues pour le training :", data.shape)
        print("\n📏 Dimensions obtenues pour le test :", data_test.shape)
        print("\n📏 Dimensions attendues pour le training : (4, 2)")
        print("\n📏 Dimensions attendues pour le test : (1, 2)")

        # Vérification des dimensions attendues après le split
        self.assertEqual(data.shape, (4, 2))
        self.assertEqual(data_test.shape, (1, 2))

    def test_train_model(self):
	print("\n🔎 Test d'entraînement du modèle...")
        X_train = self.data_train.drop(columns=['Churn'])
        y_train = self.data_train['Churn']
        model = train_model(X_train, y_train)
        self.assertIsInstance(model, RandomForestClassifier)

    def test_evaluate_model(self):
        print("\n🔎 Test d'évaluation du modèle...")
        X_train = self.data_train.drop(columns=['Churn'])
        y_train = self.data_train['Churn']
        model = train_model(X_train, y_train)
        X_test = self.data_test.drop(columns=['Churn'])
        y_test = self.data_test['Churn']
        evaluate_model(model, X_test, y_test)
        self.log_result("test_evaluate_model", True)

    def tearDown(self):
        print("\n🗑️ Nettoyage des fichiers temporaires...")
        os.remove('train_temp.csv')
        os.remove('test_temp.csv')

    @classmethod
    def tearDownClass(cls):
        print("\n📝 Génération du rapport HTML...")
        with open(cls.report_file, 'w') as f:
            f.write(f"<html><body><h1>Rapport de tests - {datetime.datetime.now()}</h1>")
            with open(cls.log_file, 'r') as log:
                f.write("<pre>")
                f.write(log.read())
                f.write("</pre>")
            f.write("</body></html>")
        print(f"\n📂 Rapport généré : {cls.report_file}")

if __name__ == '__main__':
    unittest.main()

