import unittest

from origami.cli.utils import create_projection, filter_data


class TestCreateProjection(unittest.TestCase):
    def test_include_fields(self):
        result = create_projection(include_fields="name,age,city")
        expected = {"name": 1, "age": 1, "city": 1}
        self.assertEqual(result, expected)

    def test_exclude_fields(self):
        result = create_projection(exclude_fields="address,phone")
        expected = {"address": 0, "phone": 0}
        self.assertEqual(result, expected)

    def test_include_and_exclude_fields(self):
        with self.assertRaises(ValueError):
            create_projection(include_fields="name,age", exclude_fields="name,city")

    def test_empty_input(self):
        result = create_projection()
        self.assertEqual(result, {})

    def test_empty_include_fields(self):
        result = create_projection(include_fields="")
        self.assertEqual(result, {})

    def test_empty_exclude_fields(self):
        result = create_projection(exclude_fields="")
        self.assertEqual(result, {})

    def test_whitespace_in_fields(self):
        result = create_projection(include_fields=" name , age ")
        expected = {"name": 1, "age": 1}
        self.assertEqual(result, expected)


class TestFilterData(unittest.TestCase):
    def setUp(self):
        self.data = [
            {"name": "John", "age": 30, "city": "New York"},
            {"name": "Alice", "age": 25, "city": "London"},
            {"name": "Bob", "age": 35, "city": "Paris"},
        ]

    def test_keep_fields(self):
        projection = {"name": 1, "age": 1}
        result = filter_data(self.data, projection)
        expected = [{"name": "John", "age": 30}, {"name": "Alice", "age": 25}, {"name": "Bob", "age": 35}]
        self.assertEqual(result, expected)

    def test_remove_fields(self):
        projection = {"city": 0}
        result = filter_data(self.data, projection)
        expected = [{"name": "John", "age": 30}, {"name": "Alice", "age": 25}, {"name": "Bob", "age": 35}]
        self.assertEqual(result, expected)

    def test_empty_projection(self):
        projection = {}
        result = filter_data(self.data, projection)
        self.assertEqual(result, self.data)

    def test_all_fields_removed(self):
        projection = {"name": 0, "age": 0, "city": 0}
        result = filter_data(self.data, projection)
        expected = [{}, {}, {}]
        self.assertEqual(result, expected)

    def test_non_existent_field(self):
        projection = {"name": 1, "non_existent": 1}
        result = filter_data(self.data, projection)
        expected = [{"name": "John"}, {"name": "Alice"}, {"name": "Bob"}]
        self.assertEqual(result, expected)

    def test_invalid_projection_mixed_values(self):
        projection = {"name": 1, "age": 0}
        with self.assertRaises(ValueError):
            filter_data(self.data, projection)

    def test_invalid_projection_non_binary(self):
        projection = {"name": 1, "age": 2}
        with self.assertRaises(ValueError):
            filter_data(self.data, projection)

    def test_empty_data(self):
        projection = {"name": 1}
        result = filter_data([], projection)
        self.assertEqual(result, [])

    def test_all_fields_kept(self):
        projection = {"name": 1, "age": 1, "city": 1}
        result = filter_data(self.data, projection)
        self.assertEqual(result, self.data)


if __name__ == "__main__":
    unittest.main()
