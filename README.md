# Resume Analyzer

A Tool for Resume Analysis, Predictions, and Recommendations

## About the Project

A tool which parses information from a resume using natural language processing and finds the keywords, clusters them onto sectors based on their keywords, and provides recommendations, predictions, and analytics to applicants/recruiters based on keyword matching.


## Features

### Client Features:
- Extract location and miscellaneous data from resumes
- Parse basic information, skills, and keywords
- Provide skill recommendations
- Predict job roles
- Suggest courses and certificates
- Offer resume tips and ideas
- Calculate overall resume score
- Recommend interview and resume tip videos

### Admin Features:
- View all applicant data in tabular format
- Export user data to CSV
- Access uploaded resumes
- View user feedback and ratings
- Analytical pie charts for:
  - Ratings
  - Predicted job roles
  - Experience levels
  - Resume scores
  - User demographics

### Feedback System:
- User feedback form
- Rating system (1-5)
- Overall ratings visualization
- Comment history

## Technical Requirements

### Prerequisites:
1. Python (3.9.12)
2. MySQL
3. Visual Studio Code (Recommended)
4. Visual Studio build tools for C++

## Installation and Setup
### Clone the Repository
### activate virtual environment:
```bash
cd venvapp/Scripts
activate
cd../..
cd App
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
### Running the Application:
```bash
streamlit run App.py
```

### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss proposed changes.
