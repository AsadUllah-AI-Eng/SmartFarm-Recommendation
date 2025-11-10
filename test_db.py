from app import app, db, User

def test_database():
    with app.app_context():
        # Drop all tables to start fresh
        db.drop_all()
        # Create all tables
        db.create_all()
        
        # Create a test user
        test_user = User(
            first_name="Test",
            last_name="User",
            email="test@example.com",
            country="Test Country"
        )
        test_user.set_password("test123")
        
        # Add and commit the test user
        db.session.add(test_user)
        db.session.commit()
        
        # Verify the user was created
        user = User.query.filter_by(email="test@example.com").first()
        if user:
            print("Test user created successfully!")
            print(f"User details: {user.first_name} {user.last_name} ({user.email})")
        else:
            print("Failed to create test user!")

if __name__ == "__main__":
    test_database() 