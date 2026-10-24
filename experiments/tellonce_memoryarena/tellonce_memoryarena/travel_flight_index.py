"""Query the public flight table without loading a full copy in every task.

Build the index once before running MemoryArena (about 0.5 GB, a few minutes):

    python travel_flight_index.py build --database DATABASE_DIR [--root INDEX_DIR]

The default index directory is $TELLONCE_FLIGHT_INDEX or
~/.cache/tellonce-experiments/travel-flight-index.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3

INDEX_ROOT = Path(os.environ.get('TELLONCE_FLIGHT_INDEX',
                                 Path.home()/'.cache/tellonce-experiments/travel-flight-index')).resolve()
COLUMNS = ['Flight Number', 'Price', 'DepTime', 'ArrTime', 'ActualElapsedTime',
           'FlightDate', 'OriginCityName', 'DestCityName', 'Distance']


def file_identity(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


class IndexedFlights:
    def __init__(self, *, source_sha256, root=INDEX_ROOT):
        root = Path(root)
        manifest = json.loads((root/'manifest.json').read_bytes())
        path = root/'flights.sqlite'
        if (path.is_symlink() or root.resolve() != root or manifest['schema'] != 1
                or manifest['source_sha256'] != source_sha256
                or manifest['columns'] != COLUMNS or file_identity(path) != manifest['file_identity']):
            raise ValueError('flight index does not match the public flight table')
        self.dtypes = manifest['dtypes']
        self.connection = sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True)
        self.connection.execute('PRAGMA cache_size=-1024')
        self.connection.execute('PRAGMA query_only=ON')

    def run(self, origin, destination, departure_date):
        import pandas as pd
        columns = ','.join('"'+c+'"' for c in COLUMNS)
        rows = self.connection.execute(
            f'SELECT {columns} FROM flights WHERE "OriginCityName"=? AND "DestCityName"=? '
            'AND "FlightDate"=? ORDER BY seq', (origin,destination,departure_date)).fetchall()
        if not rows:
            return f'There is no flight from {origin} to {destination} on {departure_date}.'
        return pd.DataFrame.from_records(rows,columns=COLUMNS).astype(self.dtypes)

    def close(self):
        self.connection.close()


def public_executor(database, *, source_sha256, root=INDEX_ROOT):
    # Retain the upstream execute()/formatting and the five small tool APIs.
    # Only the redundant full-table flight allocation is replaced.
    from travel_planner_env import tool_executor as original
    executor = object.__new__(original.ToolExecutor)
    executor.db_path = str(database)
    executor.flights = IndexedFlights(source_sha256=source_sha256, root=root)
    executor.restaurants = original.Restaurants(path=f'{database}/restaurants/clean_restaurant_2022.csv')
    executor.accommodations = original.Accommodations(path=f'{database}/accommodations/clean_accommodations_2022.csv')
    executor.attractions = original.Attractions(path=f'{database}/attractions/attractions.csv')
    executor.distance_matrix = original.GoogleDistanceMatrix(path=f'{database}/googleDistanceMatrix/distance.csv')
    executor.cities = original.Cities(path=f'{database}/background/citySet_with_states.txt')
    return executor


def build_index(database, root=INDEX_ROOT):
    """Write the flight rows in table order, exactly as the upstream Flights tool loads them."""
    import pandas as pd
    source = Path(database)/'flights/clean_Flights_2022.csv'
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root/'flights.sqlite'
    if path.exists():
        raise SystemExit(f'{path} already exists')
    frame = pd.read_csv(source).dropna()[COLUMNS]
    connection = sqlite3.connect(path)
    columns = ','.join('"'+c+'"' for c in COLUMNS)
    types = {'int64': 'INTEGER', 'float64': 'REAL'}
    connection.execute('CREATE TABLE flights (seq INTEGER PRIMARY KEY,' + ','.join(
        '"'+c+'" '+types.get(str(frame[c].dtype), 'TEXT') for c in COLUMNS) + ')')
    connection.executemany(f'INSERT INTO flights (seq,{columns}) VALUES (?{",?"*len(COLUMNS)})',
        ((i, *[v.item() if hasattr(v, 'item') else v for v in row]) for i, row in
         enumerate(frame.itertuples(index=False, name=None))))
    connection.execute('CREATE INDEX lookup ON flights ("OriginCityName","DestCityName","FlightDate",seq)')
    connection.commit()
    connection.close()
    manifest = {'schema': 1, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'columns': COLUMNS, 'dtypes': {c: str(frame[c].dtype) for c in COLUMNS},
                'rows': len(frame), 'file_identity': file_identity(path)}
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the indexed flight table for MemoryArena travel tasks.')
    parser.add_argument('action', choices=['build'])
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=INDEX_ROOT)
    args = parser.parse_args()
    print(build_index(args.database, args.root))
