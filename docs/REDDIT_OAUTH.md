# Reddit OAuth: Client ID, Secret, and How This App Authenticates

This project uses Reddit’s official API via PRAW. You need a Reddit application (free) to get credentials.

## 1) Create your Reddit app
- Go to: https://www.reddit.com/prefs/apps
- Click “create another app”
- Name: anything you like (e.g., Obsidian Zettelkasten Scraper)
- Type: Script (personal use)
- Redirect URI: http://localhost:8080/authorize_callback (used for OAuth flow)
- Save. You’ll get:
  - client_id (the short string under the app name)
  - client_secret (the long secret string)

## 2) Provide credentials to the scraper
The scraper reads from environment variables by default (non-interactive mode):
- REDDIT_CLIENT_ID
- REDDIT_CLIENT_SECRET
- REDDIT_USER_AGENT (e.g., "MySavedPostsFetcher/1.0")
- Optional for private/age-gated: REDDIT_USERNAME and REDDIT_PASSWORD

Windows PowerShell example:
```powershell
$env:REDDIT_CLIENT_ID = "your_client_id"
$env:REDDIT_CLIENT_SECRET = "your_client_secret"
$env:REDDIT_USER_AGENT = "ObsidianZettels/0.1 (by u/yourname)"
# Optional for private/NSFW subs
$env:REDDIT_USERNAME = "your_reddit_username"
$env:REDDIT_PASSWORD = "your_reddit_password"
```
Then run:
```powershell
python .\reddit_scraper.py --non-interactive --saved-file .\saved_posts.csv --output-dir output
```

If you omit --non-interactive, the app will prompt and can also attempt a browser OAuth flow with --oauth.

## 3) What’s happening under the hood
- PRAW is initialized with your client_id, client_secret, and user_agent.
- For many read-only endpoints, client credentials + user-agent are sufficient.
- For private/age-gated content, you can:
  - Provide username/password (password grant) OR
  - Use the browser OAuth flow with --oauth.
- The OAuth flow opens a URL, you sign in, and Reddit redirects back to localhost with a one-time code. The app exchanges it for tokens and continues.

## 4) Safety notes
- Never commit your secrets to Git.
- Prefer environment variables or a local .env file not tracked by Git.
- Reddit rate-limits apply; this tool intentionally pauses between requests.
