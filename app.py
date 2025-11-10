from flask import Flask, request, render_template, redirect, url_for, flash, make_response
from werkzeug.urls import url_parse
from werkzeug.utils import secure_filename
import numpy as np
import pandas
import sklearn
import pickle
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User
import os
import logging
import traceback
from sqlalchemy.exc import SQLAlchemyError

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load models
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','rb'))
fertilizer_model = pickle.load(open('fertilizer-recommendation-system.pkl','rb'))

# Load fertilizer recommendation model and encoders
soil_type_encoder = pickle.load(open('soil_type_encoder.pkl','rb'))
crop_type_encoder = pickle.load(open('crop_type_encoder.pkl','rb'))

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Disable caching of static files and templates
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = 'dev_key_123'  # Fixed key for development
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Flask extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Add no-cache headers to all responses
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@login_manager.user_loader
def load_user(user_id):
    try:        return User.query.get(int(user_id))
    except Exception as e:
        logger.error(f"Error loading user: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def init_db():
    """Initialize the database and create all tables"""
    try:
        with app.app_context():
            db.create_all()
            logger.info("Database tables created successfully!")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/')
def home():
    # Check if this is an auto-transition from login
    auto_transition = request.args.get('auto_transition', False)
    return render_template('temp.html', active_page="dashboard", auto_transition=auto_transition)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            password = request.form.get('password')
            
            logger.debug(f"Login attempt for email: {email}")
            
            if not email or not password:
                flash('Please provide both email and password')
                return render_template('temp.html', active_page="login")
            
            try:
                user = User.query.filter_by(email=email).first()
                logger.debug(f"Found user: {user}")
                
                if user and user.check_password(password):
                    login_user(user, remember=True)
                    logger.info(f"Login successful for user: {user.email}")
                    flash('Login successful! Welcome back.')
                    
                    # Redirect to home with auto-transition flag
                    return redirect(url_for('home', auto_transition=True))
                else:
                    if not user:
                        logger.warning(f"No user found with email: {email}")
                        flash('Invalid email address')
                    else:
                        logger.warning(f"Invalid password for user: {email}")
                        flash('Invalid password')
                    return render_template('temp.html', active_page="login")
                
            except SQLAlchemyError as e:
                logger.error(f"Database error during login: {str(e)}")
                logger.error(traceback.format_exc())
                flash('Database error occurred. Please try again.')
                return render_template('temp.html', active_page="login")
            
        except Exception as e:
            logger.error(f"Unexpected error during login: {str(e)}")
            logger.error(traceback.format_exc())
            flash('An unexpected error occurred. Please try again.')
            return render_template('temp.html', active_page="login")
    
    # GET request - Check if this is an auto-transition from signup
    auto_transition = request.args.get('auto_transition', False)
    return render_template('temp.html', active_page="login", auto_transition=auto_transition)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        try:
            # Get form data
            first_name = request.form.get('first_name')
            last_name = request.form.get('last_name')
            email = request.form.get('email')
            password = request.form.get('password')
            country = request.form.get('country')
            
            logger.debug(f"Signup attempt - Email: {email}, Name: {first_name} {last_name}")
            
            # Validate required fields
            if not all([first_name, last_name, email, password, country]):
                missing = []
                if not first_name: missing.append('First Name')
                if not last_name: missing.append('Last Name')
                if not email: missing.append('Email')
                if not password: missing.append('Password')
                if not country: missing.append('Country')
                flash(f'Missing required fields: {", ".join(missing)}')
                return render_template('temp.html', active_page="signup")
            
            try:
                # Check if user already exists
                if User.query.filter_by(email=email).first():
                    flash('Email already registered')
                    return render_template('temp.html', active_page="signup")
                
                # Create new user
                new_user = User(
                    first_name=first_name,
                    last_name=last_name,
                    email=email,
                    country=country
                )
                new_user.set_password(password)
                
                # Save to database
                db.session.add(new_user)
                db.session.commit()
                
                logger.info(f"New user created successfully: {email}")
                flash('Account created successfully! Please login with your credentials.')
                
                # Redirect to login page with auto-transition flag
                return redirect(url_for('login', auto_transition=True))
                
            except SQLAlchemyError as e:
                db.session.rollback()
                logger.error(f"Database error during signup: {str(e)}")
                logger.error(traceback.format_exc())
                flash('Database error occurred. Please try again.')
                return render_template('temp.html', active_page="signup")
            
        except Exception as e:
            logger.error(f"Unexpected error during signup: {str(e)}")
            logger.error(traceback.format_exc())
            flash('An unexpected error occurred. Please try again.')
            return render_template('temp.html', active_page="signup")
    
    # GET request
    return render_template('temp.html', active_page="signup")

@app.route('/logout')
@login_required
def logout():
    try:
        logout_user()
        return redirect(url_for('home'))
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return redirect(url_for('home'))

@app.route("/predict", methods=['POST'])
@login_required
def predict():
    try:
        # Get form data
        form_data = {
            'Nitrogen': request.form['Nitrogen'],
            'Phosporus': request.form['Phosporus'],
            'Potassium': request.form['Potassium'],
            'Temperature': request.form['Temperature'],
            'Humidity': request.form['Humidity'],
            'pH': request.form['pH'],
            'Rainfall': request.form['Rainfall']
        }

        # Convert to feature list
        feature_list = [
            float(form_data['Nitrogen']),
            float(form_data['Phosporus']),
            float(form_data['Potassium']),
            float(form_data['Temperature']),
            float(form_data['Humidity']),
            float(form_data['pH']),
            float(form_data['Rainfall'])
        ]

        single_pred = np.array(feature_list).reshape(1, -1)

        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)
        prediction = model.predict(sc_mx_features)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = "{} is the best crop to be cultivated right there".format(crop)
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        
        logger.info(f"Prediction made successfully for user: {current_user.email}")
        
        # Return the template with prediction result but without form data to clear the form
        return render_template('temp.html', 
                             prediction_result=result, 
                             active_page="prediction")
                             
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        flash('An error occurred during prediction. Please try again.')
        return render_template('temp.html', 
                             active_page="prediction")

@app.route("/predict_fertilizer", methods=['POST'])
@login_required
def predict_fertilizer():
    try:
        # Get form data
        form_data = {
            'Temperature': float(request.form['Temperature']),
            'Humidity': float(request.form['Humidity']),
            'Moisture': float(request.form['Moisture']),
            'Soil_Type': int(request.form['Soil_Type']),
            'Crop_Type': int(request.form['Crop_Type']),
            'Nitrogen': float(request.form['Nitrogen']),
            'Potassium': float(request.form['Potassium']),
            'Phosphorous': float(request.form['Phosphorous'])
        }

        # Convert to feature list
        feature_list = [
            form_data['Temperature'],
            form_data['Humidity'],
            form_data['Moisture'],
            form_data['Soil_Type'],
            form_data['Crop_Type'],
            form_data['Nitrogen'],
            form_data['Potassium'],
            form_data['Phosphorous']
        ]

        # Make prediction
        features = np.array(feature_list).reshape(1, -1)
        prediction = fertilizer_model.predict(features)

        # Define fertilizer dictionary
        fertilizer_dict = {
            0: "10-26-26",
            1: "14-35-14",
            2: "17-17-17",
            3: "20-20",
            4: "28-28",
            5: "DAP",
            6: "Urea"
        }

        if prediction[0] in fertilizer_dict:
            fertilizer = fertilizer_dict[prediction[0]]
            result = f"The recommended fertilizer is: {fertilizer}"
            
            # Add descriptions for the fertilizers
            descriptions = {
                "10-26-26": "A balanced NPK fertilizer with high phosphorus content, ideal for root development and flowering.",
                "14-35-14": "High in phosphorus, good for early growth stages and root development.",
                "17-17-17": "A perfectly balanced NPK fertilizer, good for general purpose use.",
                "20-20": "Balanced nitrogen and phosphorus, good for vegetative growth.",
                "28-28": "High nitrogen and phosphorus content, ideal for leafy crops.",
                "DAP": "Di-Ammonium Phosphate, high in phosphorus and nitrogen, good for early growth stages.",
                "Urea": "High nitrogen content, excellent for leafy growth and vegetative development."
            }
            
            result += f"\n\nDescription: {descriptions[fertilizer]}"
        else:
            result = "Sorry, we could not determine the best fertilizer with the provided data."
        
        logger.info(f"Fertilizer prediction made successfully for user: {current_user.email}")
        
        return render_template('temp.html', 
                             fertilizer_prediction=result,
                             active_page="fertilizer")
                             
    except Exception as e:
        logger.error(f"Fertilizer prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        flash('An error occurred during fertilizer prediction. Please try again.')
        return render_template('temp.html', 
                             active_page="fertilizer")

@app.route('/predict_disease', methods=['GET', 'POST'])
@login_required
def predict_disease():
    if request.method == 'POST':
        try:
            # Check if a file was uploaded
            if 'file' not in request.files:
                flash('No file uploaded')
                return render_template('temp.html', active_page="disease")
            
            file = request.files['file']
            if file.filename == '':
                flash('No file selected')
                return render_template('temp.html', active_page="disease")
            
            if file and allowed_file(file.filename):
                # Save the file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Preprocess the image
                img = load_img(filepath, target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # Normalize
                
                # Make prediction
                prediction = disease_model.predict(img_array)
                predicted_class = np.argmax(prediction[0])
                
                # Map class index to disease name
                disease_classes = [
                    'Apple___Apple_scab',
                    'Apple___Black_rot',
                    'Apple___Cedar_apple_rust',
                    'Apple___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight',
                    'Corn_(maize)___healthy',
                    'Grape___Black_rot',
                    'Grape___Esca_(Black_Measles)',
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy',
                    'Potato___Early_blight',
                    'Potato___Late_blight',
                    'Potato___healthy',
                    'Tomato___Bacterial_spot',
                    'Tomato___Early_blight',
                    'Tomato___Late_blight',
                    'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot',
                    'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]
                
                disease = disease_classes[predicted_class]
                confidence = float(prediction[0][predicted_class])
                
                # Get treatment recommendations
                treatments = {
                    'Apple___Apple_scab': 'Apply fungicides early in the season. Remove infected leaves. Maintain good air circulation.',
                    'Apple___Black_rot': 'Remove infected fruit and cankers. Prune during dry weather. Apply fungicides.',
                    'Apple___Cedar_apple_rust': 'Remove nearby cedar trees if possible. Apply fungicides in spring.',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Rotate crops. Use resistant varieties. Apply fungicides if severe.',
                    'Corn_(maize)___Common_rust_': 'Plant resistant hybrids. Apply fungicides. Avoid high humidity.',
                    'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant hybrids. Rotate crops. Apply fungicides if needed.',
                    'Grape___Black_rot': 'Remove infected fruit. Improve air circulation. Apply fungicides.',
                    'Grape___Esca_(Black_Measles)': 'Remove infected vines. Avoid vine stress. No effective treatment available.',
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply fungicides. Improve air circulation. Remove infected leaves.',
                    'Potato___Early_blight': 'Remove infected leaves. Rotate crops. Apply fungicides.',
                    'Potato___Late_blight': 'Use resistant varieties. Apply fungicides preventively. Monitor weather.',
                    'Tomato___Bacterial_spot': 'Use disease-free seeds. Rotate crops. Apply copper-based sprays.',
                    'Tomato___Early_blight': 'Remove lower infected leaves. Improve air circulation. Apply fungicides.',
                    'Tomato___Late_blight': 'Use resistant varieties. Apply fungicides preventively. Monitor weather.',
                    'Tomato___Leaf_Mold': 'Improve air circulation. Reduce humidity. Apply fungicides.',
                    'Tomato___Septoria_leaf_spot': 'Remove infected leaves. Mulch well. Apply fungicides.',
                    'Tomato___Spider_mites Two-spotted_spider_mite': 'Use insecticidal soaps. Increase humidity. Introduce predatory mites.',
                    'Tomato___Target_Spot': 'Improve air circulation. Remove infected leaves. Apply fungicides.',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whiteflies. Use resistant varieties. Remove infected plants.',
                    'Tomato___Tomato_mosaic_virus': 'Remove infected plants. Control insects. Use virus-free seeds.'
                }
                
                treatment = treatments.get(disease, 'No specific treatment available. Consult a local agricultural expert.')
                
                if 'healthy' in disease.lower():
                    result = f"Good news! Your plant appears to be healthy (Confidence: {confidence:.2%})"
                else:
                    result = f"Detected Disease: {disease.replace('___', ' - ').replace('_', ' ')}\nConfidence: {confidence:.2%}\n\nRecommended Treatment:\n{treatment}"
                
                # Save the image path for display
                image_path = os.path.join('uploads', filename)
                
                logger.info(f"Disease prediction made successfully for user: {current_user.email}")
                
                return render_template('temp.html',
                                     active_page="disease",
                                     disease_prediction=result,
                                     image_path=image_path)
                                     
        except Exception as e:
            logger.error(f"Disease prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            flash('An error occurred during disease prediction. Please try again.')
            return render_template('temp.html', active_page="disease")
            
    # GET request
    return render_template('temp.html', active_page="disease")

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('temp.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    logger.error(f"Internal server error: {str(error)}")
    return render_template('temp.html', error="An internal error occurred"), 500

if __name__ == "__main__":
    try:
        # Initialize the database
        with app.app_context():
            init_db()
            logger.info("Database initialized successfully")
        
        # Run the Flask application with debug mode and reloader
        app.run(
            debug=True,  # Enable debug mode
            use_reloader=True,  # Enable automatic reloading
            host='0.0.0.0',  # Listen on all available interfaces
            port=5000,  # Use port 5000
            threaded=True  # Enable threading
        )
    except Exception as e:
        logger.error(f"Failed to start the application: {str(e)}")
        logger.error(traceback.format_exc())