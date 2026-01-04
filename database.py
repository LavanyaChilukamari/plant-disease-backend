import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Ramlalkala",
    "database": "plant_disease",
    "auth_plugin": "mysql_native_password"
}


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def init_db():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # ---- USERS TABLE ----
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id CHAR(36) PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # ---- SCANS TABLE ----
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id CHAR(36) NOT NULL,
                disease_name VARCHAR(255),
                confidence FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT fk_user
                    FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("DB initialized successfully")

    except Error as e:
        print("DB init error:", e)


def ensure_user(user_id: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id FROM users WHERE id = %s",
        (user_id,)
    )
    exists = cursor.fetchone()

    if not exists:
        cursor.execute(
            "INSERT INTO users (id) VALUES (%s)",
            (user_id,)
        )
        conn.commit()

    cursor.close()
    conn.close()


def save_scan(user_id: str, disease: str, confidence: float):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO scans (user_id, disease_name, confidence)
        VALUES (%s, %s, %s)
    """, (user_id, disease, confidence))

    conn.commit()
    cursor.close()
    conn.close()
