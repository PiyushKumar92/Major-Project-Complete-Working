# AI-Powered Person Detection System

A comprehensive surveillance system that uses AI and machine learning to detect and track missing persons in CCTV footage.

## Features

- **Location-based Matching**: Automatically matches missing person cases with surveillance footage based on location data
- **Face Recognition**: Advanced face detection and recognition using multiple AI models
- **GPU CNN Detection**: High-performance face detection using GPU-accelerated CNN models
- **Multi-modal Recognition**: Combines face recognition with clothing analysis
- **Real-time Analysis**: Process surveillance footage in real-time
- **Admin Dashboard**: Comprehensive admin panel for case management
- **Automated Verification**: AI-powered confidence scoring and auto-verification

## Technology Stack

- **Backend**: Python Flask
- **Database**: SQLite/PostgreSQL
- **AI/ML**: 
  - OpenCV for computer vision
  - face_recognition library
  - Custom GPU CNN models
  - TensorFlow/PyTorch (optional)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Cloud Services**: AWS Rekognition (optional)

## Installation

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)
- GPU support (optional, for better performance)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-person-detection.git
   cd ai-person-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

5. **Initialize Database**
   ```bash
   python init_db.py
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

- `SECRET_KEY`: Flask secret key for sessions
- `DATABASE_URL`: Database connection string
- `AWS_ACCESS_KEY_ID`: AWS credentials (if using AWS Rekognition)
- `FACE_RECOGNITION_TOLERANCE`: Face matching sensitivity (0.3-0.6)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for auto-verification

### Face Recognition Settings

Adjust these parameters in your `.env` file:
- Lower tolerance = stricter matching
- Higher confidence threshold = fewer false positives

## Usage

### Admin Panel

1. Access admin panel at `/admin`
2. Upload missing person cases with photos
3. Add surveillance footage with location data
4. Trigger AI analysis
5. Review detection results

### API Endpoints

- `POST /api/cases` - Create new missing person case
- `POST /api/footage` - Upload surveillance footage
- `GET /api/analysis/{match_id}` - Get analysis results
- `POST /api/trigger-analysis` - Start AI analysis

## Project Structure

```
├── app.py                 # Main Flask application
├── models.py             # Database models
├── admin.py              # Admin panel routes
├── ai_location_matcher.py # Location matching algorithm
├── aws_rekognition_matcher.py # Face recognition engine
├── gpu_cnn_detector.py   # GPU CNN face detection
├── static/               # Static files (CSS, JS, uploads)
├── templates/            # HTML templates
├── requirements.txt      # Python dependencies
└── .env.example         # Environment configuration template
```

## Security Features

- Environment-based configuration
- Secure file upload handling
- Input validation and sanitization
- Session management
- Admin authentication
- Sensitive data exclusion from version control

## Performance Optimization

- GPU acceleration for face detection
- Efficient frame sampling (configurable intervals)
- Batch processing for multiple cases
- Caching for frequently accessed data
- Optimized database queries

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This system is designed for legitimate law enforcement and security purposes only. Users are responsible for compliance with local privacy laws and regulations.

## Support

For support and questions, please open an issue on GitHub or contact the development team.

## Acknowledgments

- OpenCV community for computer vision tools
- face_recognition library by Adam Geitgey
- Flask framework developers
- Contributors and testers