from flask import Blueprint, render_template, redirect, session, url_for, request, flash
from flask_login import LoginManager, login_user, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from db import db
from datetime import datetime
from cryptography.fernet import Fernet
import os

auth_bp = Blueprint('auth', __name__)

# Initialize encryption
FERNET_KEY = os.getenv('ENCRYPTION_KEY')
cipher = Fernet(FERNET_KEY)

class User():
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.email = user_data['email']
        self._is_admin = user_data.get('is_admin', False)
        self.password_hash = user_data.get('password', '')
        self.created_at = user_data.get('created_at', datetime.utcnow())
        self.learning_style = user_data.get('learning_style', {})
        self.stats = user_data.get('stats', {
            'avg_score': 0,
            'weak_concepts': []
        })
        self.encrypted_api_key = user_data.get('encrypted_api_key')

    @property
    def api_key(self):
        """Decrypt and return the API key"""
        if self.encrypted_api_key:
            try:
                return cipher.decrypt(self.encrypted_api_key).decode()
            except:
                return None
        return None

    @property
    def is_admin(self):
        return self._is_admin 

    @staticmethod
    def get(user_id):
        user_data = db.users.find_one({'_id': ObjectId(user_id)})
        return User(user_data) if user_data else None

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return self.id

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email').strip().lower()
        password = request.form.get('password')
        api_key = request.form.get('api_key', '').strip()

        if not api_key:
            flash('API key is required', 'error')
            return redirect(url_for('auth.signup'))

        if db.users.find_one({'email': {'$regex': f'^{email}$', '$options': 'i'}}):
            flash('Email already exists', 'error')
            return redirect(url_for('auth.signup'))

        encrypted_key = cipher.encrypt(api_key.encode())

        new_user = {
            'email': email,
            'password': generate_password_hash(password),
            'encrypted_api_key': encrypted_key,
            'created_at': datetime.utcnow(),
            'is_admin': False,
            'progress': session.get('guest_progress', {})
        }

        user_id = db.users.insert_one(new_user).inserted_id
        user = User(db.users.find_one({'_id': user_id}))
        login_user(user)

        if 'user_id' in session and session['user_id'].startswith('anon_'):
            db.migrate_progress_data(session['user_id'], str(user.id))
        
        session['user_id'] = str(user.id)
        session['api_key'] = api_key  # Store raw key in session

        if 'guest_progress' in session:
            db.users.update_one(
                {'_id': ObjectId(user.id)},
                {'$set': {'progress': session['guest_progress']}}
            )
            session.pop('guest_progress')

        return redirect(url_for('index'))

    return render_template('signup.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email').strip().lower()
        password = request.form.get('password')
        user_data = db.users.find_one({'email': email})

        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data)
            login_user(user)

            if 'user_id' in session and session['user_id'].startswith('anon_'):
                db.migrate_progress_data(session['user_id'], str(user.id))

            session['user_id'] = str(user.id)
            session['api_key'] = user.api_key  # Decrypted key from DB

            if user.is_admin:
                return redirect(url_for('admin.dashboard'))
            return redirect(url_for('progress_dashboard'))
            
        flash('Invalid email or password', 'error')
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    logout_user()
    session.pop('api_key', None)
    session.pop('user_id', None)
    return redirect(url_for('index'))

def init_login_manager(app):
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        user_data = db.users.find_one({'_id': ObjectId(user_id)})
        return User(user_data) if user_data else None