# VibeMap

VibeMap analyzes Spotify playlists and creates smaller, vibe-matched playlists using audio embeddings and clustering. It runs on your computer; there is no hosted VibeMap backend. While in use, your computer connects to Spotify, Deezer, and Hugging Face as needed.

## Run Locally

### Requirements

- Python 3.9 or newer
- A Spotify account and Spotify Developer app
- Git, or download and extract the repository ZIP from GitHub
- Internet access; the CLAP model downloads the first time it is needed

### 1. Download the project

Clone the repository:

```bash
git clone <repository-url>
cd VibeMap
```

Or download and extract the ZIP from GitHub, then open a terminal in the extracted project folder.

### 2. Configure Spotify

Create an app in the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) and add this exact Redirect URI to the app settings:

```text
http://127.0.0.1:5000/callback
```

Use `127.0.0.1`, not `localhost`, for the local callback.

### 3. Install dependencies

From the project folder, create and activate a virtual environment, then install the requirements.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 4. Add your credentials

Copy `.env.example` to `.env` (`Copy-Item .env.example .env` in PowerShell, or `cp .env.example .env` on macOS/Linux). Add your Spotify Client ID and Client Secret to `.env`:

```env
FLASK_ENV=development
FLASK_SECRET_KEY=choose-a-long-random-value
SPOTIFY_CLIENT_ID=your-client-id
SPOTIFY_CLIENT_SECRET=your-client-secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:5000/callback
PORT=5000
```

Do not share or commit `.env` or your Spotify Client Secret.

### 5. Start VibeMap

With the virtual environment active, run:

```bash
python app.py
```

Open **http://127.0.0.1:5000**. Keep the terminal open while using the app; press `Ctrl+C` to stop it. The CLAP model downloads the first time it is needed and may take several minutes to load.

## How It Works

The local Flask app serves the web interface and its API from the same address. Playlist processing runs on your computer. The app uses Spotify for authentication and playlist changes, Deezer for audio previews, and Hugging Face to download the CLAP model.

## Troubleshooting

- **Spotify redirect mismatch:** confirm the dashboard URI is exactly `http://127.0.0.1:5000/callback` and matches `.env`.
- **Port 5000 is busy:** stop the other process using it, or change `PORT` and register the matching callback URL with Spotify.
- **Missing credentials:** confirm `.env` is in the project folder and contains your Spotify app's Client ID and Client Secret.
- **Model installation or memory errors:** confirm the model download completed and close other memory-intensive applications before retrying.
- **Some songs are skipped:** not every track has an audio preview available from Deezer.

## Project Files

- `app.py` - Local Flask server and API routes
- `index.html` - Web interface served by Flask
- `vibe_processor.py` - Audio analysis and playlist clustering
- `requirements.txt` - Python dependencies
- `.env.example` - Local configuration template

## License

MIT. See [LICENSE](LICENSE).
