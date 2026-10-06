# VibeMap

VibeMap turns a Spotify playlist into smaller playlists grouped by musical vibe. Choose a playlist from your Spotify account or paste a Spotify playlist URL. VibeMap analyzes the tracks and groups them with K-means clustering. Choose how many groups to create, or leave the number unspecified and VibeMap will choose it automatically. The resulting playlists are added to your Spotify profile, where you can rename or delete each one.

## How It Works

1. **Choose a playlist:** Select one of your Spotify playlists or provide its Spotify URL.
2. **Group tracks by vibe:** VibeMap uses audio feature extractions and K-means clustering to split the tracks into groups. Set the number of groups yourself, or let VibeMap choose it automatically.
3. **Manage the results:** The new playlists are created in your Spotify account. Rename or delete them from the VibeMap.

VibeMap runs on your computer; there is no hosted VibeMap backend. The local Flask app serves the web interface and API. While in use, your computer connects to Spotify for authentication and playlist changes, Deezer for audio previews, and Hugging Face to download the CLAP model.

## Run Locally

### Requirements

- Python 3.11 (required by the pinned PyTorch and NumPy versions)
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

From the project folder, create a Python 3.11 virtual environment, then install the requirements. The Windows commands use the environment's Python directly, so activation is not required.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe --version
.\.venv311\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
```

Confirm the version command prints Python 3.11 before installing packages. If the environment was created with another Python version, rebuild it with the 3.11 launcher:

```powershell
py -3.11 -m venv --clear .venv311
.\.venv311\Scripts\python.exe --version
.\.venv311\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
```

macOS/Linux Terminal:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --no-cache-dir -r requirements.txt
```

### 4. Add your credentials

Copy `.env.example` to `.env` (`Copy-Item .env.example .env` in PowerShell, or `cp .env.example .env` on macOS/Linux). Add your Spotify Client ID and Client Secret to `.env`, and (change the port number in PORT and the Spotify URI if desired):

```env
FLASK_ENV=development
FLASK_SECRET_KEY=choose-a-long-random-value
SPOTIFY_CLIENT_ID=your-client-id
SPOTIFY_CLIENT_SECRET=your-client-secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:5000/callback
PORT=5000
```

Do not share `.env` or your Spotify Client Secret.

### 5. Start VibeMap

Run VibeMap with the Python from the virtual environment:

Windows PowerShell:

```powershell
.\.venv311\Scripts\python.exe app.py
```

macOS/Linux:

```bash
python app.py
```

Open **http://127.0.0.1:5000**. Keep the terminal open while using the app; press `Ctrl+C` to stop it. The CLAP model downloads the first time it is needed and may take several minutes to load.

## Troubleshooting

- **Spotify redirect mismatch:** confirm the dashboard URI is exactly `http://127.0.0.1:5000/callback` and matches `.env`.
- **Port 5000 is busy:** stop the other process using it, or change `PORT` and register the matching callback URL with Spotify.
- **Missing credentials:** confirm `.env` is in the project folder and contains your Spotify app's Client ID and Client Secret.
- **PyTorch `WinError 126`:** check that `./.venv311/Scripts/python.exe --version` reports Python 3.11. A venv created with Python 3.14 is incompatible with the pinned `torch==2.1.1`; rebuild the venv using `py -3.11 -m venv --clear .venv311`.
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
