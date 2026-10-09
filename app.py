from flask import Flask, jsonify, request, session, redirect, send_from_directory
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv
import json
from vibe_processor import VibeProcessor
from functools import wraps
import secrets

load_dotenv()

app = Flask(__name__, static_folder='assets', static_url_path='/assets')
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))

# Spotify Configuration
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI', 'http://127.0.0.1:5000/callback')

# Initialize VibeProcessor
vibe_processor = VibeProcessor()

@app.route('/', methods=['GET'])
def index():
    """Serve the VibeMap interface from the local app server."""
    return send_from_directory(app.root_path, 'index.html')

# Handle CORS preflight explicitly
@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response, 200

# Add CORS headers to every response
@app.after_request
def after_request(response):
    try:
        origin = request.headers.get('Origin', '*')
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Expose-Headers'] = 'Content-Type'
    except Exception as e:
        print(f"Error in after_request: {e}")
    return response

def get_auth_token():
    """Get auth token from session or Authorization header"""
    # Try session first (from callback)
    if 'spotify_token' in session:
        return session['spotify_token']
    
    # Try Authorization header (Bearer token from frontend)
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header[7:]  # Remove 'Bearer ' prefix
    
    return None

def refresh_access_token_if_needed():
    """Refresh Spotify access token if it's expired or about to expire"""
    import time
    
    if 'spotify_token' not in session:
        return False
    
    # Check if token is about to expire (within 5 minutes)
    token_expires = session.get('spotify_token_expires', 0)
    if token_expires > time.time() + 300:  # 5 minutes buffer
        return True
    
    # Try to refresh token
    refresh_token = session.get('spotify_refresh_token')
    if not refresh_token:
        print("No refresh token available, token refresh not possible")
        return False
    
    try:
        sp_oauth = SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI
        )
        
        token_info = sp_oauth.refresh_access_token(refresh_token)
        session['spotify_token'] = token_info['access_token']
        session['spotify_token_expires'] = token_info.get('expires_at')
        if 'refresh_token' in token_info:
            session['spotify_refresh_token'] = token_info['refresh_token']
        
        print("Successfully refreshed Spotify access token")
        return True
    except Exception as e:
        print(f"Failed to refresh token: {e}")
        return False

def require_auth(f):
    """Decorator to check if user is authenticated"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return f(*args, **kwargs)
        token = get_auth_token()
        if not token:
            return jsonify({'error': 'Not authenticated'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': vibe_processor.model_loaded,
        'message': 'App is running. Model will load on first playlist processing.'
    })

@app.route('/progress', methods=['GET'])
@require_auth
def get_progress():
    """Get current processing progress"""
    return jsonify({
        'current_song': vibe_processor.current_song,
        'current_song_image': vibe_processor.current_song_image,
        'current_index': vibe_processor.current_index,
        'total_songs': vibe_processor.total_songs,
        'skipped_count': len(vibe_processor.skipped_tracks),
        'progress_percent': int((vibe_processor.current_index / max(vibe_processor.total_songs, 1)) * 100)
    })

@app.route('/login', methods=['GET'])
def login():
    """Redirect user to Spotify authorization page"""
    sp_oauth = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope='user-library-read playlist-modify-public playlist-read-private playlist-modify-private',
        show_dialog=True
    )
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback', methods=['GET'])
def callback():
    """Handle Spotify callback"""
    code = request.args.get('code')
    
    if not code:
        return jsonify({'error': 'No authorization code received'}), 400
    
    try:
        sp_oauth = SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope='user-library-read playlist-modify-public playlist-read-private playlist-modify-private'
        )
        
        token_info = sp_oauth.get_access_token(code)
        session['spotify_token'] = token_info['access_token']
        session['spotify_refresh_token'] = token_info.get('refresh_token')
        session['spotify_token_expires'] = token_info.get('expires_at')
        session['user_id'] = spotipy.Spotify(auth=token_info['access_token']).me()['id']
        
        return f"""
        <html>
            <head><title>Authorization successful</title></head>
            <body>
                <h1>Authorization successful!</h1>
                <p>Redirecting you back to VibeMap...</p>
                <script>
                    window.opener.postMessage({{'type': 'spotify_auth_success', 'token': '{token_info['access_token']}'}},'*');
                    window.close();
                </script>
            </body>
        </html>
        """
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/process-playlist', methods=['POST', 'OPTIONS'])
@require_auth
def process_playlist():
    """Main endpoint to process a playlist"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'error': 'Invalid JSON payload'}), 400
        
        playlist_name = data.get('playlist_name')
        playlist_id = data.get('playlist_id')
        playlist_url = data.get('playlist_url')
        num_clusters = data.get('num_clusters')
        
        if not playlist_name and not playlist_id:
            return jsonify({'error': 'Playlist name or ID required'}), 400
        
        # Refresh token if needed before processing
        refresh_access_token_if_needed()
        
        token = get_auth_token()
        # Get user_id from session if available, or extract from token
        user_id = session.get('user_id')
        if not user_id:
            # Extract user_id from Spotify API if not in session
            sp = spotipy.Spotify(auth=token)
            user_id = sp.me()['id']
        
        # Process the playlist
        result = vibe_processor.process_playlist(
            token=token,
            user_id=user_id,
            playlist_name=playlist_name,
            playlist_id=playlist_id,
            num_clusters=num_clusters,
            refresh_token_callback=refresh_access_token_if_needed
        )
        
        return jsonify(result)
    except Exception as e:
        print(f"Error processing playlist: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/get-user', methods=['GET'])
@require_auth
def get_user():
    """Get current user info"""
    try:
        refresh_access_token_if_needed()
        token = get_auth_token()
        sp = spotipy.Spotify(auth=token)
        user = sp.me()
        return jsonify({
            'display_name': user['display_name'],
            'user_id': user['id'],
            'external_urls': user['external_urls']
        })
    except spotipy.exceptions.SpotifyException as e:
        print(f"Spotify auth error in /get-user: {e}")
        return jsonify({'error': 'Spotify authentication failed'}), 401
    except Exception as e:
        print(f"Error in /get-user: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/get-user-playlists', methods=['GET'])
@require_auth
def get_user_playlists():
    """Get user's playlists"""
    try:
        refresh_access_token_if_needed()
        token = get_auth_token()
        sp = spotipy.Spotify(auth=token)
        results = sp.current_user_playlists(limit=50)
        playlists = results['items']
        
        # Get all playlists if more than 50
        while results['next']:
            results = sp.next(results)
            playlists.extend(results['items'])
        
        return jsonify([{
            'id': p['id'],
            'name': p['name'],
            'tracks': {
                'total': p['tracks']['total']
            }
        } for p in playlists])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete-playlist', methods=['POST'])
@require_auth
def delete_playlist():
    """Delete a playlist from user's Spotify account"""
    try:
        token = get_auth_token()
        sp = spotipy.Spotify(auth=token)
        
        data = request.get_json()
        playlist_id = data.get('playlist_id')
        
        if not playlist_id:
            return jsonify({'error': 'Playlist ID is required'}), 400
        
        # Unfollow the playlist (this deletes it if user owns it)
        sp.current_user_unfollow_playlist(playlist_id)
        
        return jsonify({
            'success': True,
            'message': f'Playlist {playlist_id} has been deleted'
        })
    except spotipy.exceptions.SpotifyException as e:
        print(f"Spotify error deleting playlist: {e}")
        return jsonify({'error': f'Failed to delete playlist: {str(e)}'}), 400
    except Exception as e:
        print(f"Error deleting playlist: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rename-playlist', methods=['POST'])
@require_auth
def rename_playlist():
    """Rename a playlist"""
    try:
        token = get_auth_token()
        sp = spotipy.Spotify(auth=token)
        
        data = request.get_json()
        playlist_id = data.get('playlist_id')
        new_name = data.get('new_name')
        
        if not playlist_id or not new_name:
            return jsonify({'error': 'Playlist ID and new name are required'}), 400
        
        sp.playlist_change_details(playlist_id, name=new_name)
        
        return jsonify({
            'success': True,
            'message': f'Playlist renamed to {new_name}'
        })
    except spotipy.exceptions.SpotifyException as e:
        print(f"Spotify error renaming playlist: {e}")
        return jsonify({'error': f'Failed to rename playlist: {str(e)}'}), 400
    except Exception as e:
        print(f"Error renaming playlist: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/toggle-playlist-privacy', methods=['POST'])
@require_auth
def toggle_playlist_privacy():
    """Toggle privacy setting of a playlist"""
    try:
        token = get_auth_token()
        sp = spotipy.Spotify(auth=token)
        
        data = request.get_json()
        playlist_id = data.get('playlist_id')
        
        if not playlist_id:
            return jsonify({'error': 'Playlist ID is required'}), 400
        
        # Get current privacy setting
        playlist = sp.playlist(playlist_id)
        current_privacy = playlist.get('public', True)
        new_privacy = not current_privacy
        
        # Update privacy setting
        sp.playlist_change_details(playlist_id, public=new_privacy)
        
        privacy_status = 'public' if new_privacy else 'private'
        return jsonify({
            'success': True,
            'message': f'Playlist is now {privacy_status}',
            'is_public': new_privacy
        })
    except spotipy.exceptions.SpotifyException as e:
        print(f"Spotify error toggling privacy: {e}")
        return jsonify({'error': f'Failed to toggle privacy: {str(e)}'}), 400
    except Exception as e:
        print(f"Error toggling privacy: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get-playlist-tracks', methods=['POST'])
@require_auth
def get_playlist_tracks():
    """Get top tracks from a playlist"""
    try:
        token = get_auth_token()
        sp = spotipy.Spotify(auth=token)
        
        data = request.get_json()
        playlist_id = data.get('playlist_id')
        limit = data.get('limit', 10)
        
        if not playlist_id:
            return jsonify({'error': 'Playlist ID is required'}), 400
        
        # Get tracks from playlist
        results = sp.playlist_items(playlist_id, limit=min(limit, 50))
        tracks = results['items']
        
        track_data = []
        for item in tracks:
            if item['track']:
                track = item['track']
                # Get album artwork
                image_url = None
                if track.get('album') and track['album'].get('images') and len(track['album']['images']) > 0:
                    image_url = track['album']['images'][0]['url']
                
                track_data.append({
                    'name': track['name'],
                    'artist': track['artists'][0]['name'] if track.get('artists') else 'Unknown',
                    'image_url': image_url,
                    'url': track.get('external_urls', {}).get('spotify'),
                    'id': track['id']
                })
        
        return jsonify({'tracks': track_data[:limit]})
    except Exception as e:
        print(f"Error getting playlist tracks: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/logout', methods=['POST'])
def logout():
    """Logout user"""
    session.clear()
    return jsonify({'status': 'logged out'})

@app.errorhandler(404)
def not_found(e):
    response = jsonify({'error': 'Endpoint not found'})
    response.status_code = 404
    return response

@app.errorhandler(500)
def server_error(e):
    response = jsonify({'error': 'Internal server error'})
    response.status_code = 500
    return response

if __name__ == '__main__':
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(debug=debug, host='127.0.0.1', port=int(os.getenv('PORT', 5000)))
