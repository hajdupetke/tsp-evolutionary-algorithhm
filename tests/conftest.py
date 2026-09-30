from pathlib import Path
import pytest
from src.utils import load_cities

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# (filename, expected city count, first city coordinates)
TSP_DATASETS = [
    ("burma14.tsp", 14, (16.47, 96.1)),
    ("berlin52.tsp", 52, (565.0, 575.0)),
    ("kroA100.tsp", 100, (1380.0, 939.0)),
]

@pytest.fixture(params=TSP_DATASETS, ids=[d[0] for d in TSP_DATASETS])
def tsp_instance(request):
    filename, expected_n, first_city = request.param
    cities = load_cities(str(DATA_DIR / filename))
    return {
        "name": filename,
        "cities": cities,
        "n": expected_n,
        "first_city": first_city,
    }