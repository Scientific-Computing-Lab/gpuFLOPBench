import sqlite3


def get_thread_ids_from_sqlite(full_path: str, success_only: bool = False) -> list[str]:
    # open the sqlite file and read the column of thread_id
    conn = sqlite3.connect(full_path)
    cursor = conn.cursor()
    
    # there are duplicate thead_ids, we want the unique ones
    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
    thread_ids = cursor.fetchall()

    unique_thread_ids = [str(id[0]) for id in thread_ids]

    conn.close()
    return unique_thread_ids