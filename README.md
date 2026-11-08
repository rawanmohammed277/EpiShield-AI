# EpiShield AI 🦠

**EpiShield AI** is an AI-powered epidemic risk prediction and resource allocation system. It analyzes disease data, forecasts future cases, identifies high-risk regions, and assigns priority scores to help decision-makers distribute limited medical resources efficiently. The system also compares different allocation scenarios to support data-driven decisions and improve epidemic preparedness.

## Run the data pipeline interface

Start the FastAPI backend from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn Src.Backend.app:app --reload
```

Open `http://127.0.0.1:8000/ui/` to use the interface. It shows the latest run status, total rows read and loaded, row counts by source file and database table, the last execution time, and any failure message. While a run is active, the page checks `/pipeline/status` every two seconds. The backend API documentation is at `http://127.0.0.1:8000/docs`.

The interface can also be hosted separately. Add `?api=http://127.0.0.1:8000` to its URL and set `FRONTEND_ORIGINS` in the backend environment to the frontend's origin so the browser's cross-origin request is allowed.

The current backend permits one successful pipeline run per database. Use a fresh empty test database before running the pipeline again.
