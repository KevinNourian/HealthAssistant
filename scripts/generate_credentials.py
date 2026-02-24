#!/usr/bin/env python3
"""
Setup script to generate credentials.yaml with hashed passwords
Run this once before starting the app: python generate_credentials.py
"""

import yaml
import bcrypt

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Define users with plain text passwords
users = {
    'alice': {
        'email': 'alice@example.com',
        'name': 'Alice',
        'password': hash_password('temp123')
    },
    'bob': {
        'email': 'bob@example.com',
        'name': 'Bob',
        'password': hash_password('temp456')
    },
    'charlie': {
        'email': 'charlie@example.com',
        'name': 'Charlie',
        'password': hash_password('temp789')
    },
    'kevin': {                           # ← ADD YOUR USER HERE
        'email': 'kevin@example.com',
        'name': 'Kevin',
        'password': hash_password('temp123')  # ← Your password
    }
}

# Create credentials structure
credentials = {
    'credentials': {
        'usernames': users
    },
    'cookie': {
        'expiry_days': 0,
        'key': 'health_assistant_app_key',
        'name': 'health_assistant_cookie'
    },
    'preauthorized': {
        'emails': ['alice@example.com', 'bob@example.com', 'charlie@example.com']
    }
}

# Save to YAML file
with open('credentials.yaml', 'w') as file:
    yaml.dump(credentials, file, default_flow_style=False, sort_keys=False)

print("✅ Passwords hashed successfully!")
print(f"\n{'='*50}")
print("credentials.yaml created successfully!")
print(f"{'='*50}")
print("\n📋 Mock users created:\n")
print("  Username: alice")
print("  Password: temp123")
print("  Name: Alice")
print("  Email: alice@example.com\n")

print("  Username: bob")
print("  Password: temp456")
print("  Name: Bob")
print("  Email: bob@example.com\n")

print("  Username: charlie")
print("  Password: temp789")
print("  Name: Charlie")
print("  Email: charlie@example.com\n")

print("✅ Ready to run: streamlit run main.py")
