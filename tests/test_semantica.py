import unittest
from semantica import SemanticSearch

class TestSemanticSearch(unittest.TestCase):

    def setUp(self):
        self.search_engine = SemanticSearch()

    def test_encode_data(self):
        # Prueba para la creación de embeddings
        data = ['test sentence 1', 'test sentence 2']
        embeddings = self.search_engine.encode_data(data)
        self.assertEqual(len(embeddings), 2, "Debería haber 2 embeddings generados.")

    def test_find_similarities(self):
        # Prueba para calcular las similitudes
        data = ['test sentence 1', 'test sentence 2']
        embeddings = self.search_engine.encode_data(data)
        query = 'test sentence 1'
        similarities = self.search_engine.find_similarities(query, embeddings)
        self.assertEqual(len(similarities), len(data), "El número de similitudes debe coincidir con el número de datos.")

if __name__ == '__main__':
    unittest.main()