
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# SQL Server Settings
SERVER = "DESKTOP-UA1AG5E"
DATABASE = "EpiShieldDB"
DRIVER = "ODBC Driver 17 for SQL Server"

# Windows Authentication
connection_string = (
    f"DRIVER={{{DRIVER}}};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

# Encode connection string
connection_url = (
    "mssql+pyodbc:///?odbc_connect="
    + quote_plus(connection_string)
)

# Create database engine
engine = create_engine(
    connection_url,
    fast_executemany=True
)


def test_connection():

    try:
        with engine.connect() as connection:

            result = connection.exec_driver_sql(
                "SELECT DB_NAME()"
            )

            database_name = result.scalar()

            print("Connection successful!")
            print(f"Connected Database: {database_name}")

    except Exception as error:
        print("Connection failed!")
        print(error)


if __name__ == "__main__":
    test_connection()
