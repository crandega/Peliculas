import unittest
import pandas as pd
from data_loader import DataLoader

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Configurar una ruta de archivo de prueba
        self.file_path = 'src/IMDB.csv'
        self.loader = DataLoader(self.file_path)

    def test_load_data_success(self):
        # Verificar si el DataFrame se carga correctamente
        df = self.loader.load_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "El DataFrame debería tener datos.")

    def test_add_relevant_info(self):
        # Crear un DataFrame de prueba para verificar la adición de información relevante
        df = pd.DataFrame({
            'Cast': ['Director A', 'Director B'],
            'Info': ['100M', '200M']
        })
        df_result = self.loader._add_relevant_info(df)
        self.assertIn('informacion_relevante', df_result.columns)
        self.assertEqual(df_result['informacion_relevante'][0], 'Director: Director A, Valor ganado: 100M')

if __name__ == '__main__':
    unittest.main()