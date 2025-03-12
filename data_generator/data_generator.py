import random
from faker import Faker

fake = Faker()

class DataGenerator:
    def generate_person(self):
        """Generuje dane osobowe"""
        return {
            "name": fake.name(),
            "email": fake.email(),
            "address": fake.address(),
            "phone": fake.phone_number(),
            "job": fake.job()
        }
    
    def generate_company(self):
        """Generuje dane firmowe"""
        return {
            "name": fake.company(),
            "catchphrase": fake.catch_phrase(),
            "business": fake.bs(),
            "address": fake.address(),
            "website": fake.domain_name()
        }
    
    def generate_dataset(self, type_name, count=10):
        """Generuje zestaw danych"""
        if type_name == "person":
            return [self.generate_person() for _ in range(count)]
        elif type_name == "company":
            return [self.generate_company() for _ in range(count)]
        else:
            raise ValueError(f"Unknown data type: {type_name}")