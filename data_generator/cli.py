import json
import click
from data_generator import DataGenerator

@click.command()
@click.argument('type_name', type=click.Choice(['person', 'company']))
@click.option('--count', '-c', default=10, help='Liczba rekordów do wygenerowania')
@click.option('--output', '-o', default='data.json', help='Nazwa pliku wyjściowego')
def generate(type_name, count, output):
    """Generuje dane testowe i zapisuje je do pliku JSON."""
    generator = DataGenerator()
    data = generator.generate_dataset(type_name, count)
    
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    click.echo(f"Wygenerowano {count} rekordów typu {type_name} do pliku {output}")

if __name__ == '__main__':
    generate()