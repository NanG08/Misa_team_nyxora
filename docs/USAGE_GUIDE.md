## 🚀 Quick Start

### Prerequisites

- Node.js 16 or higher
- npm or yarn
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Misa
```

2. **Setup Frontend**
```bash
cd website
npm install
cp .env.example .env.local
```

Edit `.env.local`:
```env
VITE_API_URL=http://localhost:5000/api
VITE_GEMINI_API_KEY=your_gemini_api_key_here
```

3. **Setup Backend**
```bash
cd backend
npm install
cp .env.example .env
```

Edit `backend/.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=5000
ALLOWED_ORIGINS=http://localhost:5173
```

4. **Run the Application**

Terminal 1 (Backend):
```bash
cd backend
npm run dev
```

Terminal 2 (Frontend):
```bash
npm run dev
```

5. **Access the Application**
- Frontend: http://localhost:5173
- Backend API: http://localhost:5000
- Health Check: http://localhost:5000/health