#!/usr/bin/env python3
import sqlite3
import pandas as pd

# Path to your iMessage database (ensure you have Full Disk Access granted)
DB_PATH = "chat.db"

def extract_messages(db_path: str) -> pd.DataFrame:
    """
    Extracts messages from the iMessage SQLite database.

    Apple's timestamps are stored as nanoseconds since 2001-01-01.
    This function converts them to human-readable local time.

    Returns:
        DataFrame: Columns include message_id, timestamp, sender, text, and chat_id.
    """
    # Connect to the iMessage database
    conn = sqlite3.connect(db_path)
    
    # SQL query to extract messages and convert the date
    query = """
    SELECT
        message.ROWID AS message_id,
        datetime(message.date / 1000000000 + strftime('%s','2001-01-01'), 'unixepoch', 'localtime') AS timestamp,
        handle.id AS sender,
        message.text,
        chat.chat_identifier AS chat_id
    FROM
        message
    LEFT JOIN
        handle ON message.handle_id = handle.ROWID
    LEFT JOIN
        chat_message_join ON message.ROWID = chat_message_join.message_id
    LEFT JOIN
        chat ON chat.ROWID = chat_message_join.chat_id
    WHERE
        message.text IS NOT NULL
    ORDER BY
        timestamp ASC;
    """
    
    # Read the query results into a pandas DataFrame
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def main():
    try:
        df = extract_messages(DB_PATH)
        if df.empty:
            print("No messages found in the database.")
        else:
            print("Preview of extracted messages:")
            print(df.head())
            # Export the messages to a CSV file
            df.to_csv("messages.csv", index=False)
            print("✅ Exported messages to messages.csv")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

if __name__ == "__main__":
    main()
