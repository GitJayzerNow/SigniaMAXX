# SigniaMAXX – Real-Time ASL Letter & Number Recognizer

SigniaMAXX is a real-time American Sign Language (ASL) recognition system that lets you sign letters (A–Z) and numbers (0–9) in front of a camera and instantly see them recognized on screen.

Built with **Next.js**, **Python**, and **Microsoft SQL Server**, SigniaMAXX turns your webcam into an interactive ASL interpreter for fingerspelling, making it easier to learn, practice, and communicate using sign language.

## Features

- 🖥️ **Live camera input** – Recognizes signs directly from your webcam in real time  
- 🔤 **Full alphabet support** – Detects ASL letters **A–Z**  
- 🔢 **Number support** – Detects ASL numbers **0–9**  
- ⚡ **Low-latency inference** – Optimized for smooth, responsive recognition  
- 🧠 **Deep learning–based** – Uses a trained model for robust hand gesture classification  
- 🌐 **Modern web frontend** – Responsive UI built with **Next.js**  
- 🗄️ **Persistent storage** – Uses **Microsoft SQL Server** to store sessions, logs, and user data  

## How It Works

1. The **Next.js** frontend accesses your webcam and streams video frames to the backend.  
2. A **Python** service processes each frame:
   - Preprocessing (cropping, resizing, normalization)  
   - Inference using a trained deep learning model  
3. The model classifies the hand gesture into:
   - Letters: `A`–`Z`
   - Numbers: `0`–`9`  
4. The predicted character is sent back to the frontend and displayed live.  
5. Recognition results and metadata are stored in a **Microsoft SQL Server** database for history, analytics, and user sessions.

## Tech Stack

- **Frontend:** Next.js (React), TypeScript/JavaScript  
- **Backend / ML:** Python (e.g., FastAPI / Flask / custom service)  
- **Computer Vision:** OpenCV  
- **Deep Learning:** TensorFlow / Keras or PyTorch  
- **Database:** Microsoft SQL Server  
- **APIs:** REST (or GraphQL) between Next.js and Python backend  

## Installation

### Prerequisites

- Node.js & npm / yarn  
- Python 3.x  
- Microsoft SQL Server instance  
- A webcam  

### Clone the Repository

```bash
git clone https://github.com/GitJayzerNow/SigniaMAXX.git
cd SigniaMAXX
```

### Frontend (Next.js)

```bash
cd frontend          # or wherever your Next.js app lives
npm install
```

Create a `.env.local` file with your backend API URL and any other config:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Backend (Python)

```bash
cd backend           # or your Python project folder
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Configure your database connection (example using environment variables):

```env
DATABASE_URL=mssql+pyodbc://<user>:<password>@<server>/<database>?driver=ODBC+Driver+17+for+SQL+Server
```

Ensure your SQL Server instance is running and the database is created.

## Usage

### Start the Backend

```bash
cd backend
# Activate venv if not already active
python app.py
```

This starts the Python API/ML service (e.g., on `http://localhost:8000`).

### Start the Frontend

```bash
cd frontend
npm run dev
```

Open your browser at `http://localhost:3000` (or the port shown in your terminal).

- Allow camera access when prompted.  
- Point your camera at your hand.  
- Form ASL letters (A–Z) or numbers (0–9) within the frame.  
- The recognized character appears live on the UI.  
- Sessions and results are logged to your **Microsoft SQL Server** database.

## Project Structure

Example layout (adjust to your actual structure):

```text
SigniaMAXX/
├─ frontend/              # Next.js app
│  ├─ pages/
│  ├─ components/
│  ├─ .env.local
│  └─ package.json
├─ backend/               # Python API + ML
│  ├─ app.py
│  ├─ model/
│  ├─ utils/
│  ├─ requirements.txt
│  └─ .env
├─ database/
│  └─ schema.sql         # SQL scripts for tables (sessions, logs, etc.)
└─ README.md
```

## Database Schema (Overview)

Typical tables you might have in **Microsoft SQL Server**:

- `Users` – user accounts / profiles  
- `Sessions` – recognition sessions (start/end time, device info)  
- `RecognitionLogs` – per-frame or per-prediction logs (timestamp, predicted character, confidence, session_id)  

You can add or reference your `schema.sql` file for exact details.

## Future Improvements

- Support for full words and sentences (sequence modeling)  
- On-screen keyboard / text output from recognized signs  
- Mobile app version (Flutter / React Native)  
- Speech output (text-to-speech) for accessibility  
- Better hand tracking with MediaPipe or similar  
- Analytics dashboard using data stored in SQL Server  

## License

Add your chosen license here, e.g.:

```text
MIT License – see LICENSE file for details.
```

## Acknowledgments

- ASL dataset providers and open-source contributors  
- Libraries: Next.js, React, OpenCV, TensorFlow/PyTorch, and SQL Server drivers  

---

Made with ❤️ for accessible communication and ASL learning.
