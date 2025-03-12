from data_generator.data_generator import DataGenerator

def test_generate_person():
    generator = DataGenerator()
    person = generator.generate_person()
    assert isinstance(person, dict)
    assert "name" in person
    assert "email" in person

def test_generate_company():
    generator = DataGenerator()
    company = generator.generate_company()
    assert isinstance(company, dict)
    assert "name" in company
    assert "website" in company

def test_generate_dataset():
    generator = DataGenerator()
    dataset = generator.generate_dataset("person", 5)
    assert isinstance(dataset, list)
    assert len(dataset) == 5