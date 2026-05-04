import pandas as pd
import sqlite3
import io



def read_csv_from_sql(file_name, db_name):
    conn = sqlite3.connect(f"file:{db_name}.db?mode=ro", uri=True)
    cursor = conn.cursor()

    # Query the file content
    cursor.execute("SELECT content FROM files WHERE name = ?", (file_name,))
    result = cursor.fetchone()
    conn.close()

    if result:
        # Convert binary content back to a DataFrame
        content = result[0]
        return pd.read_csv(io.BytesIO(content))
    else:
        raise FileNotFoundError(f"File '{file_name}' not found in the database.")
    

def read_sequence_from_sql(name, db_path):
    conn = sqlite3.connect(db_path)

    cursor = conn.execute(
        "SELECT seq FROM sequences WHERE name = ?",
        (name,)
    )
    
    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0]
    else:
        raise FileNotFoundError(f"Sample '{name}' not found in the database.")
    

def read_structure_from_sql(name, db_path):
    conn = sqlite3.connect(db_path)

    cursor = conn.execute(
        "SELECT structure FROM structures WHERE name = ?",
        (name,)
    )
    
    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0]
    else:
        raise FileNotFoundError(f"Sample '{name}' not found in the database.")
