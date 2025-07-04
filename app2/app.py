from flask import Flask, render_template, redirect, url_for, request, flash, session, g, send_from_directory, send_file
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import shutil
import tempfile
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import model as face_model
import predict as face_predict
import base64
import cv2
from flask_socketio import SocketIO, emit, join_room, leave_room
import zipfile
import io
import re
import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this in production

login_manager = LoginManager()
login_manager.init_app(app)

DATABASE = os.path.join(os.path.dirname(__file__), 'users.db')
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

socketio = SocketIO(app)

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

class User(UserMixin):
    def __init__(self, id, username, password_hash, unique_id):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.unique_id = unique_id

    @staticmethod
    def get(user_id):
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        if user:
            return User(user['id'], user['username'], user['password_hash'], user['unique_id'])
        return None

    @staticmethod
    def get_by_username(username):
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user:
            return User(user['id'], user['username'], user['password_hash'], user['unique_id'])
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.before_request
def create_tables():
    if not hasattr(g, 'db_initialized'):
        conn = get_db()
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            unique_id TEXT UNIQUE NOT NULL
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS friend_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_user_id INTEGER NOT NULL,
            to_user_id INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending', 'accepted', 'rejected')),
            UNIQUE(from_user_id, to_user_id)
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS friendships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            friend_id INTEGER NOT NULL,
            UNIQUE(user_id, friend_id)
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS shared_photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id INTEGER NOT NULL,
            receiver_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            person_name TEXT,
            person_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id INTEGER NOT NULL,
            receiver_id INTEGER NOT NULL,
            content TEXT,
            msg_type TEXT CHECK(msg_type IN ('text', 'image')) NOT NULL DEFAULT 'text',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS prediction_credits (
            user_id INTEGER,
            date TEXT,
            count INTEGER,
            PRIMARY KEY (user_id, date)
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS credit_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            amount INTEGER NOT NULL,
            reason TEXT,
            status TEXT NOT NULL CHECK(status IN ('pending', 'accepted', 'rejected')),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()
        # Ensure admin user exists
        user = conn.execute('SELECT * FROM users WHERE username = ?', ('admin@sandhya',)).fetchone()
        if not user:
            conn.execute('INSERT INTO users (username, password_hash, unique_id) VALUES (?, ?, ?)',
                ('admin@sandhya', generate_password_hash('subhradip'), '0000'))
            conn.commit()
        conn.close()
        g.db_initialized = True

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/train', methods=['GET', 'POST'])
@login_required
def train():
    if request.method == 'POST':
        person_name = request.form['person_name']
        files = request.files.getlist('images')
        if not person_name or not files or files[0].filename == '':
            flash('Please provide a name and at least one image.')
            return redirect(url_for('train'))
        temp_dir = tempfile.mkdtemp(dir=UPLOAD_FOLDER)
        image_paths = []
        for file in files:
            if not file or not file.filename:
                continue
            filename = secure_filename(file.filename)
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)
            image_paths.append(file_path)
        model_path = face_model.train_and_save_model(image_paths, person_name)
        shutil.rmtree(temp_dir)
        if model_path:
            flash(f'Model for {person_name} trained and saved!')
        else:
            flash('Training failed. No valid faces found.')
        return redirect(url_for('train'))
    return render_template('train.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    # Credit system: 10 images per day for normal users, unlimited for admin
    is_admin = current_user.username == 'admin@sandhya' and current_user.unique_id == '0000'
    max_per_day = 10
    today = datetime.date.today().isoformat()
    conn = get_db()
    remaining = max_per_day
    if not is_admin:
        row = conn.execute('SELECT count FROM prediction_credits WHERE user_id = ? AND date = ?', (current_user.id, today)).fetchone()
        used = row['count'] if row else 0
        remaining = max_per_day - used
    else:
        remaining = '∞'
    grouped_results = {}
    result_imgs = {}
    original_files = {}
    if request.method == 'POST':
        files = request.files.getlist('images')
        num_images = len([f for f in files if f and f.filename])
        if not is_admin:
            row = conn.execute('SELECT count FROM prediction_credits WHERE user_id = ? AND date = ?', (current_user.id, today)).fetchone()
            used = row['count'] if row else 0
            if used + num_images > max_per_day:
                flash(f'Credit limit reached: You can only predict {max_per_day} images per day. Remaining: {max(0, max_per_day - used)}')
                conn.close()
                return redirect(url_for('predict'))
            # Update credits
            if row:
                conn.execute('UPDATE prediction_credits SET count = count + ? WHERE user_id = ? AND date = ?', (num_images, current_user.id, today))
            else:
                conn.execute('INSERT INTO prediction_credits (user_id, date, count) VALUES (?, ?, ?)', (current_user.id, today, num_images))
            conn.commit()
        if not files or files[0].filename == '':
            flash('Please upload at least one image.')
            return redirect(url_for('predict'))
        models = face_predict.load_models()
        for file in files:
            if not file or not file.filename:
                continue
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            predictions, image = face_predict.predict_faces(file_path, models)
            if image is not None:
                for person, accuracy, loc in predictions:
                    if person == 'unknown':
                        continue
                    if person not in grouped_results:
                        grouped_results[person] = []
                        original_files[person] = []
                    grouped_results[person].append(filename)
                    original_files[person].append(file_path)
                # Save result image with boxes for display
                result_img = face_predict.draw_results(image, predictions)
                _, buffer = cv2.imencode('.jpg', result_img)
                result_imgs[filename] = base64.b64encode(buffer).decode('utf-8')
    # Fetch friends for the modal
    conn = get_db()
    friends = conn.execute('SELECT u.id, u.username, u.unique_id FROM friendships f JOIN users u ON f.friend_id = u.id WHERE f.user_id = ?', (current_user.id,)).fetchall()
    conn.close()
    # Fetch user's credit requests
    conn2 = get_db()
    my_requests = conn2.execute('SELECT amount, reason, status, timestamp FROM credit_requests WHERE user_id = ? ORDER BY timestamp DESC', (current_user.id,)).fetchall()
    conn2.close()
    return render_template('predict.html', grouped_results=grouped_results, result_imgs=result_imgs, original_files=original_files, friends=friends, remaining=remaining, my_requests=my_requests)

@app.route('/friends', methods=['GET', 'POST'])
@login_required
def friends():
    conn = get_db()
    message = None
    # Handle search and send request
    if request.method == 'POST':
        search_id = request.form.get('search_id')
        if search_id:
            user = conn.execute('SELECT * FROM users WHERE unique_id = ?', (search_id,)).fetchone()
            if user:
                # Check if already friends
                is_friend = conn.execute('SELECT 1 FROM friendships WHERE user_id = ? AND friend_id = ?', (current_user.id, user['id'])).fetchone()
                already_sent = conn.execute('SELECT 1 FROM friend_requests WHERE from_user_id = ? AND to_user_id = ?', (current_user.id, user['id'])).fetchone()
                if user['id'] == current_user.id:
                    message = 'You cannot follow yourself.'
                elif is_friend:
                    message = f'You are already friends with {user["username"]}.'
                elif already_sent:
                    message = 'Follow request already sent.'
                else:
                    conn.execute('INSERT INTO friend_requests (from_user_id, to_user_id, status) VALUES (?, ?, ?)', (current_user.id, user['id'], 'pending'))
                    conn.commit()
                    message = f'Follow request sent to {user["username"]}.'
            else:
                message = 'No user found with that ID.'
        # Accept/reject logic
        if 'accept_id' in request.form:
            req_id = request.form.get('accept_id')
            req = conn.execute('SELECT * FROM friend_requests WHERE id = ? AND to_user_id = ?', (req_id, current_user.id)).fetchone()
            if req:
                # Add to friendships both ways (A->B and B->A)
                conn.execute('INSERT OR IGNORE INTO friendships (user_id, friend_id) VALUES (?, ?)', (current_user.id, req['from_user_id']))
                conn.execute('INSERT OR IGNORE INTO friendships (user_id, friend_id) VALUES (?, ?)', (req['from_user_id'], current_user.id))
                conn.execute('UPDATE friend_requests SET status = ? WHERE id = ?', ('accepted', req_id))
                conn.commit()
                message = 'Friend request accepted. You are now friends!'
        if 'reject_id' in request.form:
            req_id = request.form.get('reject_id')
            req = conn.execute('SELECT * FROM friend_requests WHERE id = ? AND to_user_id = ?', (req_id, current_user.id)).fetchone()
            if req:
                conn.execute('UPDATE friend_requests SET status = ? WHERE id = ?', ('rejected', req_id))
                conn.commit()
                message = 'Friend request rejected.'
    # Pending requests for current user
    pending_requests = conn.execute('''SELECT fr.id, u.username, u.unique_id FROM friend_requests fr JOIN users u ON fr.from_user_id = u.id WHERE fr.to_user_id = ? AND fr.status = 'pending' ''', (current_user.id,)).fetchall()
    # Friends list
    friends = conn.execute('''SELECT u.username, u.unique_id FROM friendships f JOIN users u ON f.friend_id = u.id WHERE f.user_id = ?''', (current_user.id,)).fetchall()
    conn.close()
    return render_template('friends.html', pending_requests=pending_requests, friends=friends, message=message)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.get_by_username(username)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        unique_id = generate_unique_id()
        password_hash = generate_password_hash(password)
        try:
            conn = get_db()
            conn.execute('INSERT INTO users (username, password_hash, unique_id) VALUES (?, ?, ?)',
                         (username, password_hash, unique_id))
            conn.commit()
            conn.close()
            flash(f'Signup successful! Your unique ID is {unique_id}. Please log in.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose another.')
    return render_template('signup.html')

def generate_unique_id():
    import random
    conn = get_db()
    while True:
        unique_id = '{:04d}'.format(random.randint(0, 9999))
        exists = conn.execute('SELECT 1 FROM users WHERE unique_id = ?', (unique_id,)).fetchone()
        if not exists:
            conn.close()
            return unique_id

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/saved_models', methods=['GET', 'POST'])
@login_required
def saved_models():
    model_dir = os.path.join(os.path.dirname(__file__), 'face_models')
    if request.method == 'POST':
        model_file = request.form.get('model_file')
        if model_file:
            file_path = os.path.join(model_dir, model_file)
            if os.path.exists(file_path):
                os.remove(file_path)
                flash(f'Model {model_file} deleted.')
            else:
                flash('Model file not found.')
        return redirect(url_for('saved_models'))
    if not os.path.exists(model_dir):
        models = []
    else:
        models = [os.path.basename(f) for f in glob.glob(os.path.join(model_dir, '*_model.pkl'))]
    return render_template('saved_models.html', models=models)

@app.context_processor
def inject_user_id():
    unique_id = None
    if current_user.is_authenticated:
        unique_id = getattr(current_user, 'unique_id', None)
    return dict(current_user_id=unique_id)

@app.route('/send_photos', methods=['POST'])
@login_required
def send_photos():
    conn = get_db()
    friend_username = request.form.get('friend_username')
    person = request.form.get('person')
    files = request.form.getlist('files')
    if not friend_username or not files:
        flash('Please select a friend and at least one photo.')
        return redirect(url_for('predict'))
    # Look up friend user ID
    user = conn.execute('SELECT id FROM users WHERE username = ?', (friend_username,)).fetchone()
    if not user or user['id'] == current_user.id:
        flash('Invalid friend selected.')
        return redirect(url_for('predict'))
    receiver_id = user['id']
    shared_dir = os.path.join(os.path.dirname(__file__), 'static', 'shared_photos')
    os.makedirs(shared_dir, exist_ok=True)
    sent = 0
    for file in files:
        src = os.path.join(UPLOAD_FOLDER, file)
        if os.path.exists(src):
            dest = os.path.join(shared_dir, f'{current_user.id}_{receiver_id}_{file}')
            import shutil
            shutil.copy(src, dest)
            conn.execute('INSERT INTO shared_photos (sender_id, receiver_id, filename, person_name, person_id) VALUES (?, ?, ?, ?, ?)',
                (current_user.id, receiver_id, os.path.basename(dest), person, '',))
            sent += 1
    conn.commit()
    conn.close()
    if sent:
        flash(f'Shared {sent} photo(s) with {friend_username} successfully!')
    else:
        flash('No photos shared. Please try again.')
    return redirect(url_for('predict'))

@app.route('/inbox')
@login_required
def inbox():
    conn = get_db()
    # Get all shared photos sent to current user, grouped by sender
    photos = conn.execute('''SELECT sp.id, sp.filename, sp.person_name, sp.person_id, sp.timestamp, u.username as sender_name, u.unique_id as sender_id
                            FROM shared_photos sp JOIN users u ON sp.sender_id = u.id
                            WHERE sp.receiver_id = ?
                            ORDER BY sp.timestamp DESC''', (current_user.id,)).fetchall()
    # Group by sender
    grouped = {}
    for row in photos:
        key = (row['sender_name'], row['sender_id'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)
    conn.close()
    return render_template('inbox.html', grouped=grouped)

@app.route('/download_photo/<int:photo_id>/<string:ext>')
@login_required
def download_photo(photo_id, ext):
    conn = get_db()
    photo = conn.execute('SELECT * FROM shared_photos WHERE id = ? AND receiver_id = ?', (photo_id, current_user.id)).fetchone()
    conn.close()
    if not photo:
        flash('Photo not found or access denied.')
        return redirect(url_for('inbox'))
    # Photos are stored in static/shared_photos/
    photo_dir = os.path.join(os.path.dirname(__file__), 'static', 'shared_photos')
    file_path = os.path.join(photo_dir, photo['filename'])
    if not os.path.exists(file_path):
        flash('File not found on server.')
        return redirect(url_for('inbox'))
    # Convert to requested format if needed
    if ext.lower() not in ['jpg', 'jpeg', 'png']:
        flash('Invalid file format.')
        return redirect(url_for('inbox'))
    import cv2
    img = cv2.imread(file_path)
    temp_path = os.path.join(photo_dir, f'temp_{photo_id}.{ext}')
    cv2.imwrite(temp_path, img)
    from flask import send_file
    response = send_file(temp_path, as_attachment=True)
    # Do NOT delete the temp file immediately; let the OS clean up or add a scheduled cleanup if needed
    return response

@socketio.on('join')
def handle_join(data):
    room = str(data['room'])
    join_room(room)

@socketio.on('leave')
def handle_leave(data):
    room = str(data['room'])
    leave_room(room)

@socketio.on('send_message')
def handle_send_message(data):
    sender_id = data['sender_id']
    receiver_id = data['receiver_id']
    content = data['content']
    msg_type = data.get('msg_type', 'text')
    # Save to DB
    conn = get_db()
    conn.execute('INSERT INTO messages (sender_id, receiver_id, content, msg_type) VALUES (?, ?, ?, ?)',
                 (sender_id, receiver_id, content, msg_type))
    conn.commit()
    conn.close()
    # Emit to both users' rooms
    emit('receive_message', data, to=str(sender_id))
    emit('receive_message', data, to=str(receiver_id))

@socketio.on('send_image')
def handle_send_image(data):
    # data: sender_id, receiver_id, image (base64)
    handle_send_message(data)

@app.route('/chat_history/<int:friend_id>')
@login_required
def chat_history(friend_id):
    conn = get_db()
    messages = conn.execute('''SELECT * FROM messages WHERE (sender_id = ? AND receiver_id = ?) OR (sender_id = ? AND receiver_id = ?) ORDER BY timestamp ASC''',
        (current_user.id, friend_id, friend_id, current_user.id)).fetchall()
    conn.close()
    return {'messages': [dict(m) for m in messages]}

@app.route('/chat')
@login_required
def chat():
    conn = get_db()
    friends = conn.execute('SELECT u.id, u.username, u.unique_id FROM friendships f JOIN users u ON f.friend_id = u.id WHERE f.user_id = ?', (current_user.id,)).fetchall()
    conn.close()
    return render_template('chat.html', friends=friends)

@app.route('/download_all_photos', methods=['POST'])
@login_required
def download_all_photos():
    conn = get_db()
    photos = conn.execute('SELECT filename FROM shared_photos WHERE receiver_id = ?', (current_user.id,)).fetchall()
    conn.close()
    photo_dir = os.path.join(os.path.dirname(__file__), 'static', 'shared_photos')
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for photo in photos:
            file_path = os.path.join(photo_dir, photo['filename'])
            if os.path.exists(file_path):
                zipf.write(file_path, arcname=photo['filename'])
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='all_photos.zip')

@app.route('/clear_data', methods=['POST'])
@login_required
def clear_data():
    conn = get_db()
    # Delete all shared/received photos for this user
    photos = conn.execute('SELECT filename FROM shared_photos WHERE sender_id = ? OR receiver_id = ?', (current_user.id, current_user.id)).fetchall()
    photo_dir = os.path.join(os.path.dirname(__file__), 'static', 'shared_photos')
    for photo in photos:
        file_path = os.path.join(photo_dir, photo['filename'])
        if os.path.exists(file_path):
            os.remove(file_path)
    conn.execute('DELETE FROM shared_photos WHERE sender_id = ? OR receiver_id = ?', (current_user.id, current_user.id))
    # Delete all chat messages for this user
    conn.execute('DELETE FROM messages WHERE sender_id = ? OR receiver_id = ?', (current_user.id, current_user.id))
    conn.commit()
    conn.close()
    flash('All your shared/received photos and chat messages have been cleared. Models are not affected.')
    return redirect(url_for('home'))

@app.route('/clear_inbox', methods=['POST'])
@login_required
def clear_inbox():
    conn = get_db()
    # Delete all received photos for this user
    photos = conn.execute('SELECT filename FROM shared_photos WHERE receiver_id = ?', (current_user.id,)).fetchall()
    photo_dir = os.path.join(os.path.dirname(__file__), 'static', 'shared_photos')
    for photo in photos:
        file_path = os.path.join(photo_dir, photo['filename'])
        if os.path.exists(file_path):
            os.remove(file_path)
    conn.execute('DELETE FROM shared_photos WHERE receiver_id = ?', (current_user.id,))
    conn.commit()
    conn.close()
    flash('All received photos have been cleared from your inbox.')
    return redirect(url_for('inbox'))

@app.route('/rename_model', methods=['POST'])
@login_required
def rename_model():
    old_model = request.form.get('old_model')
    new_model = request.form.get('new_model')
    if not old_model or not new_model:
        flash('Invalid model name.')
        return redirect(url_for('saved_models'))
    # Sanitize new name (no special chars, no .pkl)
    new_model = re.sub(r'[^a-zA-Z0-9_\-]', '', new_model)
    if not new_model:
        flash('Invalid new model name.')
        return redirect(url_for('saved_models'))
    model_dir = os.path.join(os.path.dirname(__file__), 'face_models')
    old_path = os.path.join(model_dir, old_model)
    new_path = os.path.join(model_dir, f'{new_model}_model.pkl')
    if not os.path.exists(old_path):
        flash('Original model file not found.')
        return redirect(url_for('saved_models'))
    if os.path.exists(new_path):
        flash('A model with that name already exists.')
        return redirect(url_for('saved_models'))
    os.rename(old_path, new_path)
    flash(f'Model renamed to {new_model}_model.pkl!')
    return redirect(url_for('saved_models'))

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin_panel():
    if not (current_user.username == 'admin@sandhya' and getattr(current_user, 'unique_id', None) == '0000'):
        flash('Admin access only.')
        return redirect(url_for('home'))
    conn = get_db()
    # Handle user deletion
    if request.method == 'POST':
        user_id = request.form.get('delete_user_id')
        if user_id and str(user_id) != str(current_user.id):
            conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.commit()
            flash('User deleted.')
        # Handle user ID edit
        edit_user_id = request.form.get('edit_user_id')
        new_unique_id = request.form.get('new_unique_id')
        if edit_user_id and new_unique_id:
            if not (new_unique_id.isdigit() and len(new_unique_id) == 4):
                flash('User ID must be exactly 4 digits.')
            else:
                exists = conn.execute('SELECT id FROM users WHERE unique_id = ? AND id != ?', (new_unique_id, edit_user_id)).fetchone()
                if exists:
                    flash('That ID is already taken by another user.')
                else:
                    conn.execute('UPDATE users SET unique_id = ? WHERE id = ?', (new_unique_id, edit_user_id))
                    conn.commit()
                    flash('User ID updated successfully!')
        # Handle credit request approval/rejection
        approve_id = request.form.get('approve_request_id')
        reject_id = request.form.get('reject_request_id')
        if approve_id:
            req = conn.execute('SELECT * FROM credit_requests WHERE id = ?', (approve_id,)).fetchone()
            if req and req['status'] == 'pending':
                # Add credits to user for today (subtract from used count)
                today = datetime.date.today().isoformat()
                row = conn.execute('SELECT count FROM prediction_credits WHERE user_id = ? AND date = ?', (req['user_id'], today)).fetchone()
                if row:
                    conn.execute('UPDATE prediction_credits SET count = count - ? WHERE user_id = ? AND date = ?', (req['amount'], req['user_id'], today))
                else:
                    conn.execute('INSERT INTO prediction_credits (user_id, date, count) VALUES (?, ?, ?)', (req['user_id'], today, -req['amount']))
                conn.execute('UPDATE credit_requests SET status = ? WHERE id = ?', ('accepted', approve_id))
                conn.commit()
                flash('Credit request approved and points added.')
        if reject_id:
            req = conn.execute('SELECT * FROM credit_requests WHERE id = ?', (reject_id,)).fetchone()
            if req and req['status'] == 'pending':
                conn.execute('UPDATE credit_requests SET status = ? WHERE id = ?', ('rejected', reject_id))
                conn.commit()
                flash('Credit request rejected.')
    users = conn.execute('SELECT id, username, unique_id FROM users').fetchall()
    # Fetch all credit requests with user info
    credit_requests = conn.execute('''SELECT cr.id, cr.user_id, cr.amount, cr.reason, cr.status, cr.timestamp, u.username, u.unique_id
        FROM credit_requests cr JOIN users u ON cr.user_id = u.id ORDER BY cr.timestamp DESC''').fetchall()
    conn.close()
    return render_template('admin.html', users=users, credit_requests=credit_requests)

@login_manager.unauthorized_handler
def unauthorized_callback():
    flash('Please log in to access this page.')
    return redirect(url_for('login'))

@app.route('/request_credits', methods=['POST'])
@login_required
def request_credits():
    amount = request.form.get('amount')
    reason = request.form.get('reason', '').strip()
    if not amount or not amount.isdigit() or int(amount) <= 0:
        flash('Please enter a valid amount.')
        return redirect(url_for('predict'))
    amount = int(amount)
    conn = get_db()
    # Only allow one pending request per user
    pending = conn.execute('SELECT 1 FROM credit_requests WHERE user_id = ? AND status = ?', (current_user.id, 'pending')).fetchone()
    if pending:
        flash('You already have a pending request.')
        conn.close()
        return redirect(url_for('predict'))
    conn.execute('INSERT INTO credit_requests (user_id, amount, reason, status) VALUES (?, ?, ?, ?)', (current_user.id, amount, reason, 'pending'))
    conn.commit()
    conn.close()
    flash('Credit request submitted to admin.')
    return redirect(url_for('predict'))

if __name__ == '__main__':
    socketio.run(app, debug=True) 