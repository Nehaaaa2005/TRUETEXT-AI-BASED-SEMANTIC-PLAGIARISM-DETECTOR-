import sqlite3
import bcrypt

conn = sqlite3.connect("truetext.db")
cursor = conn.cursor()

admin_password = "MySecureAdmin@123"

hashed_password = bcrypt.hashpw(admin_password.encode(), bcrypt.gensalt())

try:
    cursor.execute(
        "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
        ("neha_admin", hashed_password, "admin")
    )
    conn.commit()
    print("✅ Admin created")
except:
    print("⚠️ Admin already exists")

conn.close()