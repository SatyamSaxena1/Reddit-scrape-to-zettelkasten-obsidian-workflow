from fastapi import FastAPI
import praw
import webbrowser
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
import base64
import io
import prawcore
import json
import time
import requests
import os
import csv
import argparse
import getpass
import logging
import sys
from typing import Optional

app = FastAPI()

# Default Reddit credentials (fall back to environment variables or user input)
DEFAULT_CONFIG = {
    # IMPORTANT: Do NOT commit your app credentials to source control.
    # The repository should use environment variables or a local secrets store.
    'client_id': os.getenv('REDDIT_CLIENT_ID', ''),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET', ''),
    'user_agent': os.getenv('REDDIT_USER_AGENT', 'MySavedPostsFetcher/1.0'),
    'username': os.getenv('REDDIT_USERNAME', ''),
    'password': os.getenv('REDDIT_PASSWORD', '')
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load your saved links from the Reddit data export file
def load_saved_links(file_path):
    """Load saved links from either a JSON export (list of dicts with 'permalink')
    or from a CSV with a 'permalink' column.
    Returns a list of permalink strings (full URL or /r/... path).
    """
    links = []
    if file_path.lower().endswith('.csv'):
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'permalink' in row and row['permalink']:
                    links.append(row['permalink'].strip())
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Assuming the file is a list like [{"permalink": "/r/..."}, ...]
        for item in data:
            if isinstance(item, dict) and 'permalink' in item:
                links.append(item['permalink'])
            elif isinstance(item, str):
                links.append(item)
    return links

def init_reddit(config: dict) -> praw.Reddit:
    """Initialize and return a praw.Reddit instance from config dict.
    config keys: client_id, client_secret, user_agent, username(optional), password(optional)
    """
    reddit_kwargs = {
        'client_id': config.get('client_id'),
        'client_secret': config.get('client_secret'),
        'user_agent': config.get('user_agent')
    }
    if config.get('username') and config.get('password'):
        reddit_kwargs['username'] = config.get('username')
        reddit_kwargs['password'] = config.get('password')
    reddit = praw.Reddit(**reddit_kwargs)
    return reddit


def oauth_authorize(reddit: praw.Reddit, scopes=None, redirect_port: int = 8080, timeout: int = 120):
    """Start an OAuth authorization flow. Opens a browser and starts a temporary local HTTP server
    to capture the redirect with the code. Falls back to asking the user to paste the 'code' if
    the local server cannot be started.
    Returns True on success (refresh token available via reddit.read_only = False), False otherwise.
    """
    if scopes is None:
        scopes = ['identity', 'read', 'history']

    state = 'reddit_scraper_state'
    redirect_uri = f'http://localhost:{redirect_port}/authorize_callback'
    # PRAW expects reddit to be instantiated with that redirect_uri set
    reddit.config.redirect_uri = redirect_uri
    auth_url = reddit.auth.url(scopes, state, 'permanent')

    print('\nOpen this URL in your browser to authorize the application:')
    print(auth_url)
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    # Try to start a simple HTTP server to catch the redirect
    code_container = {'code': None}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            from urllib import parse
            qs = parse.urlparse(self.path).query
            params = dict(parse.parse_qsl(qs))
            if 'state' in params and params.get('state') == state and 'code' in params:
                code_container['code'] = params['code']
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'<html><body><h1>Authorization successful. You can close this window.</h1></body></html>')
            else:
                self.send_response(400)
                self.end_headers()

        def log_message(self, format, *args):
            return

    try:
        server = HTTPServer(('localhost', redirect_port), Handler)
        server.timeout = 1
        start = time.time()
        print(f"Listening on http://localhost:{redirect_port}/ for redirect (timeout {timeout}s)...")
        while time.time() - start < timeout and code_container['code'] is None:
            server.handle_request()
        server.server_close()
    except OSError:
        # Port busy or cannot bind: fallback to manual paste
        print('Could not start local server to receive redirect; please paste the code parameter from the redirected URL')
        code_container['code'] = None

    if not code_container['code']:
        code = input('Paste the `code` parameter from the URL after authorization (or blank to cancel): ').strip()
    else:
        code = code_container['code']

    if not code:
        print('Authorization cancelled')
        return False

    try:
        reddit.auth.authorize(code)
        print('Authorization complete — tokens stored in the praw instance')
        return True
    except Exception as e:
        print(f'Authorization failed: {e}')
        return False


def inspect_for_403_reason(permalink: str) -> str:
    """Do a direct GET to the permalink (with a normal browser UA) and try to infer a reason for 403/denial.
    Returns a short human-readable reason string.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; RedditScraper/1.0)'}
        resp = requests.get(permalink, headers=headers, timeout=15)
        status = resp.status_code
        if status == 403:
            # Look for common phrases
            txt = resp.text.lower()
            if 'private community' in txt or 'this community is private' in txt:
                return 'subreddit is private'
            if 'you must be 18 years old to view this community' in txt or 'over 18' in txt or 'nsfw' in txt:
                return 'age-gated / nsfw content (login required)'
            if 'you are not allowed to view this community' in txt:
                return 'access restricted (moderator only or banned)'
            return f'403 response from reddit (unknown reason)'
        elif status == 404:
            return 'not found (404) — possibly removed or deleted'
        else:
            return f'HTTP {status} response; check manually'
    except Exception as e:
        return f'could not inspect permalink (HTTP GET failed): {e}'


def fetch_all(reddit: praw.Reddit, saved_file: str = "saved_posts.csv", output_dir: str = "output", limit: Optional[int] = None, combined_file: Optional[str] = None, inline_media: bool = False, inline_media_max_bytes: int = 200000):
    os.makedirs(output_dir, exist_ok=True)
    links = load_saved_links(saved_file)
    if limit is not None:
        links = links[:limit]
    results = []
    combined_fh = None
    if combined_file:
        combined_fh = open(combined_file, 'w', encoding='utf-8')
    for link in links:
        # Extract post ID from permalink (e.g., "abc123" from "/r/subreddit/comments/abc123/...")
        try:
            post_id = link.split('/comments/')[1].split('/')[0]
        except Exception:
            # Try alternate: some permalinks might already be just the id
            post_id = link.strip().split('/')[-1]

        entry = {'id': post_id, 'permalink': link, 'status_code': None, 'error': None}
        try:
            submission = reddit.submission(id=post_id)
            # Fetch and expand all comments (handles pagination)
            submission.comments.replace_more(limit=None)
            comments = [comment.body for comment in submission.comments.list()]
            
            # Download images/videos if present
            images = []
            # preview may not exist; guard access
            if getattr(submission, 'preview', None) and isinstance(submission.preview, dict) and 'images' in submission.preview:
                for img in submission.preview['images']:
                    img_url = img['source']['url']
                    img_filename = f"{post_id}_{os.path.basename(img_url).split('?')[0]}"
                    img_path = os.path.join(output_dir, img_filename)
                    try:
                        resp = requests.get(img_url, timeout=20)
                        resp.raise_for_status()
                        data = resp.content
                        # Inline small images if requested
                        if inline_media and len(data) <= inline_media_max_bytes:
                            b64 = base64.b64encode(data).decode('ascii')
                            images.append({'type': 'inline', 'filename': img_filename, 'data_base64': b64, 'mime': resp.headers.get('Content-Type')})
                        else:
                            with open(img_path, 'wb') as f:
                                f.write(data)
                            images.append({'type': 'file', 'path': img_path, 'url': img_url})
                    except Exception as e:
                        logger.warning(f"failed to download {img_url}: {e}")
            elif submission.url and submission.url.endswith(('.jpg', '.png', '.gif', '.mp4')):  # Direct media link
                media_url = submission.url
                media_filename = f"{post_id}_{os.path.basename(media_url).split('?')[0]}"
                media_path = os.path.join(output_dir, media_filename)
                try:
                    resp = requests.get(media_url, timeout=20)
                    resp.raise_for_status()
                    data = resp.content
                    if inline_media and len(data) <= inline_media_max_bytes:
                        b64 = base64.b64encode(data).decode('ascii')
                        images.append({'type': 'inline', 'filename': media_filename, 'data_base64': b64, 'mime': resp.headers.get('Content-Type')})
                    else:
                        with open(media_path, 'wb') as f:
                            f.write(data)
                        images.append({'type': 'file', 'path': media_path, 'url': media_url})
                except Exception as e:
                    logger.warning(f"failed to download {media_url}: {e}")
            
            # Save post data as JSON
            data = {
                'id': post_id,
                'permalink': link,
                'title': submission.title,
                'selftext': submission.selftext,
                'author': submission.author.name if submission.author else '[deleted]',
                'score': submission.score,
                'comments': comments,
                'media': images,
                'fetched_at': time.asctime()
            }
            # Write per-post JSON file for convenience
            with open(os.path.join(output_dir, f"{post_id}.json"), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # Also append to combined JSONL if requested
            if combined_fh:
                combined_fh.write(json.dumps(data, ensure_ascii=False) + '\n')
            entry['status_code'] = 200
            entry['error'] = None
            logger.info(f"Saved {post_id} -> {os.path.join(output_dir, f'{post_id}.json')}")
            
            time.sleep(0.6)  # Buffer to respect 100 QPM rate limit
        except Exception as e:
            # More descriptive error handling / code mapping
            err_msg = str(e)
            if '403' in err_msg or 'forbidden' in err_msg.lower():
                code = 403
            elif '404' in err_msg:
                code = 404
            elif 'timed out' in err_msg.lower() or 'timeout' in err_msg.lower():
                code = 408
            else:
                code = 500
            entry['status_code'] = code
            entry['error'] = err_msg
            logger.error(f"Error fetching {post_id}: {err_msg} (code {code})")
            # append error to a persistent log
            with open(os.path.join(output_dir, 'errors.log'), 'a', encoding='utf-8') as lf:
                lf.write(f"{time.asctime()} - {post_id} - {code} - {err_msg}\n")
        results.append(entry)
    # Save a summary
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as sf:
        json.dump({'processed': len(links), 'results': results}, sf, ensure_ascii=False, indent=2)
    if combined_fh:
        combined_fh.close()
    return {"status": "Done processing posts", "processed": len(links), 'results': results}


def interactive_config():
    print('Interactive setup for Reddit scraper')
    cfg = {}
    use_env = input('Use environment variables for Reddit credentials if present? [Y/n]: ').strip().lower()
    if use_env in ['', 'y', 'yes']:
        cfg.update(DEFAULT_CONFIG)
    else:
        cfg['client_id'] = input(f"Reddit client_id [{DEFAULT_CONFIG['client_id']}]: ") or DEFAULT_CONFIG['client_id']
        cfg['client_secret'] = input(f"Reddit client_secret [{'<hidden>' if DEFAULT_CONFIG['client_secret'] else ''}]: ") or DEFAULT_CONFIG['client_secret']
        cfg['user_agent'] = input(f"user_agent [{DEFAULT_CONFIG['user_agent']}]: ") or DEFAULT_CONFIG['user_agent']
        want_auth = input('Do you want to provide Reddit username/password for private/age-gated content? [y/N]: ').strip().lower()
        if want_auth in ['y', 'yes']:
            cfg['username'] = input('username: ')
            cfg['password'] = getpass.getpass('password (hidden): ')
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Reddit saved posts scraper')
    parser.add_argument('--saved-file', default='saved_posts.csv', help='Path to saved links (CSV or JSON)')
    parser.add_argument('--output-dir', default='output', help='Directory to store outputs')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of posts to process')
    parser.add_argument('--non-interactive', action='store_true', help='Run without interactive prompts (use env/defaults)')
    parser.add_argument('--install-requirements', action='store_true', help='Install requirements before running')
    parser.add_argument('--combined-file', default=None, help='Write output to a single combined JSONL file instead of per-post JSONs')
    parser.add_argument('--inline-media', action='store_true', help='Embed small media items inline (base64) into the combined output')
    parser.add_argument('--inline-media-max-bytes', type=int, default=200000, help='Max size (bytes) of media to inline; larger will be saved as external files with links')
    parser.add_argument('--oauth', action='store_true', help='Attempt OAuth interactive flow before scraping')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.install_requirements:
        logger.info('Installing requirements from requirements.txt...')
        # Use pip to install requirements (best-effort)
        os.system(f"{sys.executable} -m pip install -r requirements.txt")

    if args.non_interactive:
        cfg = DEFAULT_CONFIG
    else:
        cfg = interactive_config()

    reddit = init_reddit(cfg)
    # Optional OAuth browser flow if requested
    if args.oauth:
        try:
            oauth_authorize(reddit)
        except Exception as e:
            logger.error(f"OAuth flow failed: {e}")
    logger.info(f"Using output dir: {args.output_dir}")
    res = fetch_all(reddit, saved_file=args.saved_file, output_dir=args.output_dir, limit=args.limit)
    logger.info(f"Finished: processed={res.get('processed')}")
    print(json.dumps(res if isinstance(res, dict) else {'status': str(res)}, indent=2))

# Run the API locally: uvicorn reddit_saved_fetcher:app --reload
# Access via http://localhpipost:8000/fetch-all?saved_file=path/to/your/saved.json&output_dir=your_output_folder