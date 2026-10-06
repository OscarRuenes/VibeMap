import spotipy
from spotipy.oauth2 import SpotifyOAuth
import torch
import librosa
import numpy as np
from transformers import ClapModel, ClapProcessor
import requests
import os
import warnings
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import defaultdict

# Suppress ID3v2 warnings from librosa/mpg123
warnings.filterwarnings('ignore', message='.*ID3v2.*')
warnings.filterwarnings('ignore', category=UserWarning)
import logging
logging.getLogger('librosa').setLevel(logging.ERROR)
logging.getLogger('audioread').setLevel(logging.ERROR)

# Suppress mpg123 stderr output at system level
import io
import contextlib

class SuppressStderr:
    """Context manager to suppress stderr output"""
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = io.StringIO()
        return self
    
    def __exit__(self, *args):
        sys.stderr = self._original_stderr

class VibeProcessor:
    """Handles all the ML processing for VibeMap"""
    
    def __init__(self):
        """Initialize the processor (model loads lazily on first use)"""
        print("Initializing VibeProcessor...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Progress tracking
        self.current_song = None
        self.current_song_image = None
        self.current_index = 0
        self.total_songs = 0
        self.skipped_tracks = []
        
        # Model will be loaded lazily on first use
        self.model = None
        self.processor = None
        self.model_loaded = False
        self.use_lightweight = False
        print("✓ VibeProcessor ready (model will load on first playlist processing)")
        
        # Vibe library for descriptive matching
        self.vibe_library = {
            # --- POP & ELECTRONIC ---
            "Pop": "Upbeat radio music with catchy melodies, female or male vocals, and polished production",
            "Synth-Pop": "80s style synthesizers, retro electronic drums, new wave vibe, catchy beat",
            "Hyperpop": "Distorted bass, pitched up vocals, glitchy electronic textures, very fast tempo",
            "Indie Pop": "Jangly electric guitars, soft vocals, catchy melody, bedroom studio sound",
            "EDM": "High energy electronic dance music, heavy bass drop, synthesizer lead, club atmosphere",
            "House": "Four-on-the-floor drum beat, 120 bpm, groovy bassline, soulful vocals, dance floor",
            "Techno": "Repetitive mechanical rhythm, synthesized textures, minimal melody, dark atmosphere",
            "Lo-Fi": "Low fidelity hip hop beat, vinyl crackle noise, slow tempo, relaxing piano loop",

            # --- ROCK & ALTERNATIVE ---
            "Rock": "Distorted electric guitars, heavy drums, energetic male vocals, classic rock band sound",
            "Indie Rock": "Alternative rock band, raw guitar sound, garage recording feel, energetic but unpolished",
            "Punk": "Fast tempo, aggressive power chords, shouting vocals, high energy, raw production",
            "Psychedelic": "Reverb-heavy guitars, delay effects, trippy atmosphere, 60s rock vibe",
            "Soft Rock": "Acoustic guitars, light drums, smooth vocals, relaxing rock ballad",
            "Metal": "Heavily distorted down-tuned guitars, aggressive drumming, double kick pedal, screaming vocals",

            # --- HIP HOP & R&B ---
            "Hip Hop": "Boom bap drum beat, rapping male vocals, sampling culture, rhythmic flow",
            "Trap": "808 sub bass, rapid hi-hat rolls, autotuned vocals, modern rap production",
            "R&B": "Smooth soulful vocals, slow groove, electronic beats, romantic atmosphere",
            "Neo-Soul": "Jazzy chords, laid back hip hop beat, soulful vocals, organic instrumentation",

            # --- MOODS ---
            "Happy": "Major key, uptempo, bright instrumentation, clapping, joyful atmosphere",
            "Sad": "Minor key, slow tempo, melancholic piano or strings, emotional vocals",
            "Aggressive": "Loud volume, distorted textures, fast tempo, shouting, chaotic energy",
            "Chill": "Slow tempo, soft dynamics, minimal instrumentation, relaxing atmosphere",
            "Dark": "Low frequencies, ominous drone, dissonant chords, creepy atmosphere, minor key",
            "Euphoric": "Uplifting chord progression, large soundstage, crescendo, emotional high",
            "Romantic": "Slow tempo, strings section, soft piano, intimate vocals, love song vibe",

            # --- CONTEXTS ---
            "Workout": "High bpm, consistent rhythm, heavy bass, motivating energy, gym music",
            "Party": "Loud, upbeat, danceable rhythm, celebratory vibe, crowd noise textures",
            "Focus": "Instrumental only, no vocals, repetitive calm pattern, background music",
            "Night Drive": "Synthwave aesthetic, steady beat, atmospheric pads, neon city vibe",
            "Summer": "Tropical percussion, bright acoustic guitar, upbeat, sunny atmosphere",
            "Rainy Day": "Soft acoustic instruments, sound of rain, melancholic jazz, cozy vibe"
        }
        
    def load_model(self):
        """Lazily load CLAP model on first use"""
        if self.model_loaded:
            return True
        if os.getenv("CLAP_DISABLED", "").lower() in {"1", "true", "yes"}:
            print("CLAP disabled via CLAP_DISABLED=1. Using lightweight embeddings.")
            self.use_lightweight = True
            return False
        
        try:
            model_name = os.getenv("CLAP_MODEL_NAME", "laion/clap-htsat-unfused")
            print(f"Loading CLAP model from HuggingFace ({model_name}) (this may take a minute)...")
            self.model = ClapModel.from_pretrained(
                model_name,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.processor = ClapProcessor.from_pretrained(model_name)
            self.model_loaded = True
            print("✓ CLAP model loaded successfully")
            return True
        except Exception as e:
            print(f"⚠ Warning: Could not load CLAP model: {e}")
            self.model = None
            self.processor = None
            self.model_loaded = False
            self.use_lightweight = True
            return False

    def get_lightweight_embedding(self, audio_path):
        """Fallback embedding using MFCC statistics (no CLAP)"""
        try:
            with SuppressStderr():
                audio_data, sr = librosa.load(audio_path, sr=22050)
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
            mfcc_mean = mfcc.mean(axis=1)
            mfcc_var = mfcc.var(axis=1)
            features = np.concatenate([mfcc_mean, mfcc_var])
            return features.reshape(1, -1)
        except Exception as e:
            print(f"Error generating lightweight embedding: {e}")
            return None

    def download_preview(self, track_name, artist_name, output_filename="temp.mp3"):
        """Download 30s preview from Deezer API with multiple retry strategies"""
        base_url = "https://api.deezer.com/search"
        
        # Try different search strategies
        search_queries = [
            f'artist:"{artist_name}" track:"{track_name}"',  # Exact match
            f'{artist_name} {track_name}',                    # Fuzzy match
            track_name,                                        # Just track name
        ]

        try:
            if os.path.exists(output_filename):
                os.remove(output_filename)

            for query in search_queries:
                try:
                    params = {'q': query, 'limit': 5}  # Get top 5 results to increase chance
                    response = requests.get(base_url, params=params, timeout=10)
                    data = response.json()

                    if 'data' not in data or not data['data']:
                        continue

                    # Try each result until we find one with a preview
                    for result in data['data']:
                        preview_url = result.get('preview')
                        if preview_url:
                            audio_response = requests.get(preview_url, timeout=10)
                            if audio_response.status_code == 200 and len(audio_response.content) > 1000:
                                with open(output_filename, 'wb') as f:
                                    f.write(audio_response.content)
                                return output_filename
                except Exception as e:
                    continue
            
            # If no preview found, return None
            return None

        except Exception as e:
            print(f"Error downloading {track_name}: {e}")
            return None

    def get_embedding(self, audio_path):
        """Generate audio embedding using CLAP model"""
        try:
            with SuppressStderr():
                audio_data, _ = librosa.load(audio_path, sr=48000)
            audio_data = audio_data[:48000 * 30]

            inputs = self.processor(audios=audio_data, return_tensors="pt", sampling_rate=48000).to(self.device)

            with torch.no_grad():
                outputs = self.model.get_audio_features(**inputs)

            return outputs.cpu().numpy()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def get_embeddings_batch(self, audio_paths, batch_size=4):
        """Generate embeddings for multiple audio files in batches (faster)"""
        if self.use_lightweight or not self.model_loaded or not self.model or not self.processor:
            return [self.get_lightweight_embedding(path) for path in audio_paths]

        embeddings = []
        
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i+batch_size]
            batch_audio = []
            
            try:
                for path in batch_paths:
                    # Suppress stderr to hide ID3 warnings from librosa/mpg123
                    with SuppressStderr():
                        audio_data, _ = librosa.load(path, sr=48000)
                    audio_data = audio_data[:48000 * 30]
                    batch_audio.append(audio_data)
                
                if not batch_audio:
                    continue
                
                # Pad audio to same length
                max_len = max(len(a) for a in batch_audio)
                batch_audio = [np.pad(a, (0, max_len - len(a))) for a in batch_audio]
                
                inputs = self.processor(audios=batch_audio, return_tensors="pt", sampling_rate=48000).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.get_audio_features(**inputs)
                
                embeddings.extend(outputs.cpu().numpy())
                
            except Exception as e:
                print(f"Batch processing error: {e}")
                # Fall back to individual processing for this batch
                for path in batch_paths:
                    vec = self.get_embedding(path)
                    if vec is not None:
                        embeddings.append(vec)
        
        return embeddings

    def get_spotify_audio_features(self, sp, track_ids):
        """Fetch Spotify audio features for tracks (BPM, energy, danceability, etc.)"""
        audio_features_dict = {}
        
        # Spotify API limits to 100 IDs per request
        batch_size = 100
        for i in range(0, len(track_ids), batch_size):
            batch_ids = track_ids[i:i+batch_size]
            try:
                features = sp.audio_features(batch_ids)
                for feature in features:
                    if feature and feature.get('id'):
                        audio_features_dict[feature['id']] = feature
            except Exception as e:
                print(f"Warning: Could not fetch audio features: {e}")
        
        return audio_features_dict

    def find_playlist_id(self, sp, playlist_name):
        """Find playlist ID by name"""
        playlists = sp.current_user_playlists(limit=50)

        while playlists:
            for playlist in playlists['items']:
                if playlist['name'].lower() == playlist_name.lower():
                    return playlist['id']

            if playlists['next']:
                playlists = sp.next(playlists)
            else:
                playlists = None

        return None

    def get_all_tracks(self, sp, playlist_id):
        """Get all tracks from a playlist"""
        results = sp.playlist_items(playlist_id)
        tracks = results['items']

        while results['next']:
            results = sp.next(results)
            tracks.extend(results['items'])

        return tracks

    def cluster_songs(self, dataset, num_clusters=None):
        """Cluster songs using KMeans (combines CLAP embeddings with Spotify audio features)"""
        if len(dataset) < 2:
            return {0: dataset}

        # Prepare CLAP embeddings
        embeddings = np.array([d['embedding'].flatten() for d in dataset])
        X_norm = normalize(embeddings)
        
        # Extract and normalize Spotify audio features
        audio_features_list = []
        feature_names = ['tempo', 'energy', 'danceability', 'valence', 'acousticness', 
                        'instrumentalness', 'liveness', 'speechiness']
        
        for item in dataset:
            features = []
            if 'audio_features' in item and item['audio_features']:
                af = item['audio_features']
                # Normalize tempo to 0-1 range (typical range: 60-200 BPM)
                tempo = af.get('tempo', 120) / 200.0
                features = [
                    tempo,
                    af.get('energy', 0.5),
                    af.get('danceability', 0.5),
                    af.get('valence', 0.5),
                    af.get('acousticness', 0.5),
                    af.get('instrumentalness', 0.5),
                    af.get('liveness', 0.5),
                    af.get('speechiness', 0.5)
                ]
            else:
                # Default if no features available
                features = [0.5] * 8
            audio_features_list.append(features)
        
        audio_features = np.array(audio_features_list)
        audio_features_norm = normalize(audio_features)
        
        # Combine CLAP embeddings (30% weight) with audio features (70% weight)
        combined = np.hstack([
            X_norm * 0.3,
            audio_features_norm * 0.7
        ])
        
        # Apply PCA for dimensionality reduction
        n_features = combined.shape[1]
        n_components = min(25, len(dataset), n_features)
        if n_components < 2:
            n_components = min(len(dataset), n_features)
        pca = PCA(n_components=n_components)
        X_final = pca.fit_transform(combined)

        # Determine K if not provided
        if num_clusters is None or num_clusters < 2:
            max_k = min(10, len(dataset))
            best_score = -1
            optimal_k = 2

            for k in range(2, max_k):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_final)
                score = silhouette_score(X_final, labels)

                if score > best_score:
                    best_score = score
                    optimal_k = k
        else:
            optimal_k = min(num_clusters, len(dataset) - 1)

        # Final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_final)

        # Group by cluster
        clusters = {i: [] for i in range(optimal_k)}
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(dataset[idx])

        return clusters

    def name_clusters(self, clusters):
        """Assign descriptive names to clusters based on vibe matching"""
        if not self.model_loaded or not self.model or not self.processor:
            return {
                c_id: {
                    'name': f"Cluster {c_id + 1}",
                    'vibes': [],
                    'size': len(songs)
                }
                for c_id, songs in clusters.items()
            }

        display_labels = list(self.vibe_library.keys())
        descriptive_prompts = list(self.vibe_library.values())

        # Embed descriptions
        text_inputs = self.processor(text=descriptive_prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            label_features = self.model.get_text_features(**text_inputs)

        label_matrix = label_features.cpu().numpy()
        label_matrix = normalize(label_matrix)

        cluster_names = {}

        for c_id, songs in clusters.items():
            if not songs:
                continue

            song_matrix = np.array([s['embedding'].flatten() for s in songs])
            song_matrix = normalize(song_matrix)

            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(song_matrix, label_matrix)

            cluster_vote_tally = defaultdict(float)

            for i in range(len(songs)):
                scores = sim_matrix[i]
                top_5_indices = np.argsort(scores)[-5:][::-1]

                for idx in top_5_indices:
                    score = scores[idx]
                    if score > 0:
                        label_name = display_labels[idx]
                        cluster_vote_tally[label_name] += score

            final_scores = []
            for label, total_impact in cluster_vote_tally.items():
                std_score = total_impact / len(songs)
                final_scores.append((label, std_score))

            sorted_votes = sorted(final_scores, key=lambda x: x[1], reverse=True)

            if not sorted_votes:
                final_name = "Eclectic Mix"
            else:
                top_3_labels = [item[0] for item in sorted_votes[:3]]
                final_name = " + ".join(top_3_labels)

            cluster_names[c_id] = {
                'name': final_name,
                'vibes': [(label, float(score)) for label, score in sorted_votes[:3]],
                'size': len(songs)
            }

        return cluster_names

    def ensure_unique_vibes(self, cluster_names):
        """Ensure each vibe appears only once across ALL playlists (global uniqueness)"""
        if not cluster_names:
            return cluster_names
        
        # Collect all vibes with their scores and cluster IDs
        vibe_candidates = []  # List of dicts with cluster_id, vibe_name, score
        
        for c_id, cluster_info in cluster_names.items():
            for i, (vibe_label, vibe_score) in enumerate(cluster_info['vibes']):
                vibe_candidates.append({
                    'cluster_id': c_id,
                    'vibe': vibe_label,
                    'score': float(vibe_score),
                    'rank': i,  # 0 for top choice, 1 for second, etc.
                    'cluster_size': cluster_info['size']
                })
        
        # Sort by score (descending) and cluster size to prioritize high-scoring vibes from larger clusters
        vibe_candidates.sort(key=lambda x: (-x['score'], -x['cluster_size'], -x['rank']))
        
        assigned_vibes = {}  # cluster_id -> [vibe1, vibe2, vibe3]
        used_vibes = set()  # Track which vibes have been used globally
        
        # First pass: greedy assignment for first vibe per cluster (ensure each cluster gets at least 1 unique vibe)
        for candidate in [v for v in vibe_candidates if v['rank'] == 0]:
            c_id = candidate['cluster_id']
            vibe = candidate['vibe']
            
            if c_id not in assigned_vibes:
                assigned_vibes[c_id] = []
            
            # Skip if cluster already has a vibe or vibe is used
            if vibe not in used_vibes and len(assigned_vibes[c_id]) == 0:
                assigned_vibes[c_id].append(vibe)
                used_vibes.add(vibe)
        
        # Second pass: fill remaining vibes for clusters that don't have their first choice
        for candidate in vibe_candidates:
            c_id = candidate['cluster_id']
            vibe = candidate['vibe']
            
            if c_id not in assigned_vibes:
                assigned_vibes[c_id] = []
            
            # Assign if vibe is new and cluster needs more vibes
            if vibe not in used_vibes and len(assigned_vibes[c_id]) < 3:
                assigned_vibes[c_id].append(vibe)
                used_vibes.add(vibe)
        
        # Third pass: if any cluster doesn't have a vibe, assign from available vibes
        all_vibes = set(self.vibe_library.keys())
        for c_id in cluster_names:
            if c_id not in assigned_vibes or len(assigned_vibes[c_id]) == 0:
                assigned_vibes[c_id] = []
                # Find any available vibe not used
                for vibe in all_vibes - used_vibes:
                    assigned_vibes[c_id].append(vibe)
                    used_vibes.add(vibe)
                    break
                # If no available vibes, use a generic one
                if len(assigned_vibes[c_id]) == 0:
                    assigned_vibes[c_id].append("Unique Mix")
        
        # Build result with globally unique vibes
        updated_names = {}
        for c_id, vibes in assigned_vibes.items():
            updated_names[c_id] = {
                'name': " + ".join(vibes[:3]) if vibes else "Unique Mix",
                'vibes': [(v, 1.0 / (i + 1)) for i, v in enumerate(vibes[:3])],
                'size': cluster_names[c_id]['size']
            }
        
        return updated_names

    def process_playlist(self, token, user_id, playlist_name=None, playlist_id=None, num_clusters=None, refresh_token_callback=None):
        """Main processing function"""
        self.refresh_token_callback = refresh_token_callback
        self.current_song = None
        self.current_song_image = None
        self.current_index = 0
        self.total_songs = 0
        self.skipped_tracks = []
        try:
            fast_mode = os.getenv("AUDIO_FEATURES_ONLY", "").lower() in {"1", "true", "yes"}
            
            sp = spotipy.Spotify(auth=token)
            
            # Get user info for display name
            user_info = sp.me()
            user_id = user_info['id']
            username = user_info.get('display_name', user_id)
            
            # Find or use provided playlist ID
            if playlist_id:
                # Use the provided playlist ID directly
                pid = playlist_id
                print(f"Using provided playlist ID: {pid}")
            else:
                # Find playlist by name
                print(f"Searching for playlist: {playlist_name}")
                pid = self.find_playlist_id(sp, playlist_name)
            
            if not pid:
                raise ValueError(f"Playlist not found")
            
            # Get original playlist's privacy setting and creator info
            original_playlist = sp.playlist(pid)
            is_public = original_playlist.get('public', True)
            original_playlist_name = original_playlist.get('name', 'Unknown')
            
            # Get creator info
            creator_name = None
            if original_playlist.get('owner'):
                creator_name = original_playlist['owner'].get('display_name', original_playlist['owner'].get('id'))
            
            print(f"Original playlist: {original_playlist_name}")
            print(f"Original playlist creator: {creator_name}")
            print(f"Original playlist is {'public' if is_public else 'private'}")
            
            # Get all tracks
            tracks = self.get_all_tracks(sp, pid)
            print(f"Found {len(tracks)} tracks")
            
            if len(tracks) < 2:
                raise ValueError("Playlist must have at least 2 tracks")
            
            # Initialize progress tracking
            self.total_songs = len(tracks)
            self.current_index = 0

            # Load the model after setting the playlist total so progress is meaningful during startup.
            if not fast_mode and not self.model_loaded:
                if os.getenv("CLAP_DISABLED", "").lower() in {"1", "true", "yes"}:
                    self.use_lightweight = True
                    print("⚠ CLAP disabled. Using lightweight embeddings (MFCC).")
                elif not self.load_model():
                    print("⚠ Falling back to lightweight embeddings (MFCC).")
            
            # Process tracks and generate embeddings
            dataset = []
            downloaded_files = []

            if fast_mode:
                print("⚠ AUDIO_FEATURES_ONLY enabled. Skipping audio previews and embeddings.")
                # Initialize progress tracking
                self.total_songs = len(tracks)
                self.skipped_tracks = []

                spotify_ids = []
                track_info = []
                for i, item in enumerate(tracks):
                    if not item or not isinstance(item, dict):
                        self.skipped_tracks.append('Unknown')
                        continue
                    track = item.get('track') or {}
                    if not track:
                        self.skipped_tracks.append(track.get('name', 'Unknown'))
                        continue

                    name = track['name']
                    artist = track['artists'][0]['name']
                    spotify_id = track['id']
                    spotify_ids.append(spotify_id)
                    track_info.append((name, artist, spotify_id))

                    self.current_index = i + 1
                    self.current_song = f"{name} - {artist}"

                print("Fetching Spotify audio features (BPM, energy, danceability, etc.)...")
                audio_features_dict = self.get_spotify_audio_features(sp, spotify_ids)

                for name, artist, spotify_id in track_info:
                    af = audio_features_dict.get(spotify_id) or {}
                    tempo = af.get('tempo', 120) / 200.0
                    features = np.array([
                        tempo,
                        af.get('energy', 0.5),
                        af.get('danceability', 0.5),
                        af.get('valence', 0.5),
                        af.get('acousticness', 0.5),
                        af.get('instrumentalness', 0.5),
                        af.get('liveness', 0.5),
                        af.get('speechiness', 0.5)
                    ], dtype=np.float32).reshape(1, -1)

                    dataset.append({
                        "name": name,
                        "artist": artist,
                        "embedding": features,
                        "spotify_id": spotify_id,
                        "audio_features": af
                    })

                print(f"✓ Prepared features for {len(dataset)} songs")
            else:
            
                print(f"Downloading previews for {len(tracks)} tracks...")
                for i, item in enumerate(tracks):
                    if not item or not isinstance(item, dict):
                        self.skipped_tracks.append('Unknown')
                        continue
                    track = item.get('track') or {}
                    if not track:
                        self.skipped_tracks.append(track.get('name', 'Unknown'))
                        continue
                    
                    name = track['name']
                    artist = track['artists'][0]['name']
                    
                    # Update progress
                    self.current_index = i + 1
                    self.current_song = f"{name} - {artist}"
                    
                    # Get album artwork
                    self.current_song_image = None
                    if track.get('album') and track['album'].get('images') and len(track['album']['images']) > 0:
                        self.current_song_image = track['album']['images'][0]['url']
                    
                    print(f"[{i+1}/{len(tracks)}] Downloading: {name} - {artist}", end='\r')
                    
                    # Download preview
                    file_path = self.download_preview(name, artist)
                    
                    if file_path:
                        downloaded_files.append({
                            'path': file_path,
                            'name': name,
                            'artist': artist,
                            'spotify_id': track['id'],
                            'image_url': self.current_song_image
                        })
                    else:
                        self.skipped_tracks.append(name)
            
                print(f"\n✓ Downloaded {len(downloaded_files)}/{len(tracks)} previews (skipped: {len(self.skipped_tracks)})")
                
                if not downloaded_files:
                    raise ValueError("Could not download any previews")
                
                # Generate embeddings in batches (faster than sequential)
                print("Generating embeddings (batch processing)...")
                file_paths = [f['path'] for f in downloaded_files]
                embeddings = self.get_embeddings_batch(file_paths, batch_size=2)
                
                # Fetch Spotify audio features for all tracks
                print("Fetching Spotify audio features (BPM, energy, danceability, etc.)...")
                spotify_ids = [f['spotify_id'] for f in downloaded_files]
                audio_features_dict = self.get_spotify_audio_features(sp, spotify_ids)
                
                # Combine embeddings with track info and audio features
                for i, (emb, file_info) in enumerate(zip(embeddings, downloaded_files)):
                    if emb is not None:
                        spotify_id = file_info['spotify_id']
                        dataset.append({
                            "name": file_info['name'],
                            "artist": file_info['artist'],
                            "embedding": emb,
                            "spotify_id": spotify_id,
                            "audio_features": audio_features_dict.get(spotify_id, {})
                        })
                        if (i + 1) % 10 == 0:
                            print(f"  Processed {i + 1}/{len(downloaded_files)} songs")
                
                # Cleanup
                for f in downloaded_files:
                    if os.path.exists(f['path']):
                        os.remove(f['path'])
                
                print(f"✓ Generated embeddings for {len(dataset)} songs")
            
            if len(dataset) < 2:
                raise ValueError("Could not generate embeddings for enough songs")
            
            # Cluster songs
            clusters = self.cluster_songs(dataset, num_clusters)
            
            # Name clusters
            cluster_names = self.name_clusters(clusters)
            
            # Ensure each playlist has a unique top vibe
            cluster_names = self.ensure_unique_vibes(cluster_names)
            
            # Refresh token before creating playlists (important for long operations)
            if self.refresh_token_callback:
                print("Refreshing Spotify token before creating playlists...")
                self.refresh_token_callback()
                # Get fresh Spotify client with new token
                sp = spotipy.Spotify(auth=token)
            
            # Create playlists on Spotify
            created_playlists = []
            for cluster_id, songs in clusters.items():
                if not songs:
                    continue
                
                vibe_info = cluster_names.get(cluster_id, {})
                playlist_name_created = f"{vibe_info.get('name', 'Cluster')} (VibeMap)"
                
                # Use original playlist name in description, fall back to creator name if available
                description_source = original_playlist_name
                if creator_name and creator_name != "Unknown":
                    description_source = f"{creator_name}'s {original_playlist_name} playlist"
                
                new_playlist = sp.user_playlist_create(
                    user=user_id,
                    name=playlist_name_created,
                    public=is_public,
                    collaborative=False,
                    description=f"{len(songs)} songs clustered by vibes from {description_source}."
                )
                
                # Explicitly update playlist privacy setting (Spotify API sometimes doesn't apply it on creation)
                sp.playlist_change_details(new_playlist['id'], public=is_public)
                
                print(f"Created playlist: {playlist_name_created} (public={is_public})")
                
                # Add tracks
                track_uris = [f"spotify:track:{s['spotify_id']}" for s in songs if s.get('spotify_id')]
                
                if track_uris:
                    batch_size = 100
                    for i in range(0, len(track_uris), batch_size):
                        batch = track_uris[i : i + batch_size]
                        sp.playlist_add_items(new_playlist['id'], batch)
                
                created_playlists.append({
                    'name': playlist_name_created,
                    'id': new_playlist['id'],
                    'song_count': len(songs),
                    'vibes': vibe_info.get('vibes', []),
                    'url': new_playlist['external_urls']['spotify'],
                    'sample_songs': [{'name': s['name'], 'artist': s['artist']} for s in songs[:5]]
                })
            
            return {
                'success': True,
                'message': f'Created {len(created_playlists)} playlists',
                'playlists': created_playlists,
                'total_songs_processed': len(dataset),
                'creator_name': creator_name,
                'original_playlist_name': original_playlist_name
            }
        
        except Exception as e:
            print(f"Error in process_playlist: {e}")
            return {
                'success': False,
                'error': str(e)
            }
