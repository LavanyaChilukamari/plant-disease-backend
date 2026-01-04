import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Ramlalkala",
    database="plant_disease",
    auth_plugin="mysql_native_password"
)

print("DB CONNECTED OK")
conn.close()
