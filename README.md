# AI Resume Analyzer using Generative AI

An AI-powered resume analysis application built using Python, Streamlit, and Google Gemini API. It compares resumes with job descriptions and generates personalized feedback to help candidates improve their resumes.

## Features

- PDF resume upload and text extraction.
- AI-powered resume and job description analysis.
- Estimated job-match score.
- Matching and missing skills identification.
- Resume strengths and weaknesses.
- Personalized improvement suggestions.
- Downloadable analysis report.
- Automatic retries for temporary API failures.

## Tech Stack

- **Programming Language:** Python
- **Frontend:** Streamlit
- **Generative AI:** Google Gemini API
- **PDF Processing:** PyPDF2
- **API Integration:** Google GenAI SDK

## How to Run

1. Clone the GitHub repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Create a `.env` file and add `GEMINI_API_KEY=your_api_key`.
4. Run the application using `python -m streamlit run app.py`.

## How It Works

1. Upload a PDF resume.
2. Paste the target job description.
3. Click Analyze Resume.
4. View the AI-generated analysis and recommendations.
5. Download the analysis report.

**Note:** The job-match score is an AI-generated estimate, not an official ATS score.