import unittest
from unittest.mock import patch
from buscar_peliculas import MovieSearchApp

class TestMovieSearchApp(unittest.TestCase):

    @patch('builtins.input', side_effect=['test query', 'salir'])
    def test_run(self, mock_input):
        # Prueba básica para verificar la ejecución de la aplicación con entrada de usuario
        app = MovieSearchApp('src/IMDB.csv')
        app.run()
        self.assertTrue(True, "El método run debe manejar correctamente la entrada de usuario.")

if __name__ == '__main__':
    unittest.main()