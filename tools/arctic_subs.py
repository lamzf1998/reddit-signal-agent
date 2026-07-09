import json
import os
import re
import time
import requests
from datetime import datetime, timedelta, timezone


def convert_unix_to_iso(unix_time: int | float) -> str:
    """Convert Unix timestamp to ISO-formatted string in Singapore timezone."""
    timezone_offset = timezone(timedelta(hours=8))
    local_time = datetime.fromtimestamp(unix_time, tz=timezone_offset)
    return local_time.isoformat(timespec='milliseconds')


def iso_to_timestamp_ms(iso_date: str) -> int:
    """Convert ISO 8601 datetime string to Unix timestamp in milliseconds."""
    dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    "Referer": "https://arctic-shift.photon-reddit.com/download-tool",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
}


def get_subreddit_posts(subreddit: str, after: int, before: int) -> dict:
    """Fetch all posts for a subreddit, paginating past the 1000 limit."""
    url = "https://arctic-shift.photon-reddit.com/api/posts/search"
    all_data = []
    current_after = after

    while True:
        params = {
            "subreddit": subreddit,
            "before": before,
            "limit": "auto",
            "sort": "asc",
            "after": current_after,
            "meta-app": "download-tool",
        }
        response = requests.get(url, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json().get("data", [])
        all_data.extend(data)

        if len(data) < 1000:
            break

        # Advance cursor past the last item's timestamp
        last_created = data[-1].get("created_utc", 0)
        current_after = int(last_created * 1000)
        print(f"    Fetched {len(all_data)} posts so far, paginating...")
        time.sleep(0.5)

    return {"data": all_data}


def get_subreddit_comments(subreddit: str, after: int, before: int) -> dict:
    """Fetch all comments for a subreddit, paginating past the 1000 limit."""
    url = "https://arctic-shift.photon-reddit.com/api/comments/search"
    all_data = []
    current_after = after

    while True:
        params = {
            "subreddit": subreddit,
            "before": before,
            "limit": "auto",
            "sort": "asc",
            "after": current_after,
            "meta-app": "download-tool",
        }
        response = requests.get(url, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json().get("data", [])
        all_data.extend(data)

        if len(data) < 1000:
            break

        # Advance cursor past the last item's timestamp
        last_created = data[-1].get("created_utc", 0)
        current_after = int(last_created * 1000)
        print(f"    Fetched {len(all_data)} comments so far, paginating...")
        time.sleep(0.5)

    return {"data": all_data}


def parse_arctic_post(post: dict, collection_time_iso: str) -> dict:
    """
    Parse an Arctic Shift post into the standard data format.

    Args:
        post: Raw post data from Arctic Shift API
        collection_time_iso: ISO timestamp of when data was collected
    """
    post_id = post.get("id", "")
    author = post.get("author", "")
    subreddit = post.get("subreddit", "")
    title = post.get("title", "")
    selftext = post.get("selftext", "") or ""
    created_utc = post.get("created_utc", 0)
    score = post.get("score", 0)
    upvote_ratio = post.get("upvote_ratio", 0.5)
    num_comments = post.get("num_comments", 0)
    url = post.get("url", "")
    link_flair_text = post.get("link_flair_text", "")
    subreddit_id = post.get("subreddit_id", "")[3:] if post.get("subreddit_id") else ""
    author_fullname = post.get("author_fullname", "")[3:] if post.get("author_fullname") else ""
    locked = post.get("locked", False)
    removed_by_category = post.get("removed_by_category", "") or ""

    # Calculate upvotes/downvotes from score and ratio
    if upvote_ratio != 0.5 and upvote_ratio != 0:
        num_upvotes = round(upvote_ratio * (score / (2 * upvote_ratio - 1)))
        num_downvotes = round((1 - upvote_ratio) * (score / (2 * upvote_ratio - 1)))
    else:
        num_upvotes = score
        num_downvotes = score

    # Build URLs
    post_iso_time = convert_unix_to_iso(created_utc) if created_utc else ""
    title_for_url = re.sub(r'[^\w\s]', '', title).replace(' ', '_').lower()
    thread_url = f"https://reddit.com/r/{subreddit}/comments/{post_id}/{title_for_url}/"
    post_url = thread_url
    redditor_url = f"https://reddit.com/user/{author}" if author else ""

    # Find URLs in content
    urls_list = re.findall(r'(https?://[^\s\)\]\}]+)', selftext) if selftext else []
    if url and url != post_url:
        urls_list.append(url)

    # Check for media
    post_has_image = post.get("is_reddit_media_domain", False) or "i.redd.it" in (url or "")
    post_has_video = post.get("is_video", False)
    media_count = 1 if (post_has_image or post_has_video) else 0

    return {
        'platform': 'Forum',
        'postType': 'Forum_Thread',
        'postRawID': str(post_id),
        'redditorID': str(author_fullname),
        'redditorName': str(author),
        'redditorURL': str(redditor_url),
        'redditorJoinedISOTime': None,  # Not available from Arctic Shift
        'redditorImageURL': "",
        'redditorSuspended': None,
        'redditorVerified': False,
        'redditorIsGold': False,
        'redditorPostKarma': 0,
        'redditorCommentKarma': 0,
        'postTitle': str(title),
        'postThreadUrl': str(thread_url),
        'postOrigThreadContent': str(selftext),
        'postUrl': str(post_url),
        'postISOTime': str(post_iso_time),
        'postParentID': "",
        'postSourceName': f"r/{subreddit}",
        'postSourceID': str(subreddit_id),
        'postFlairText': str(link_flair_text) if link_flair_text else "",
        'postContent': str(selftext),
        'postCounterData': [{
            'collectionISOTime': str(collection_time_iso),
            'postScore': score,
            'postUpvoteRatio': float(upvote_ratio),
            'postNumUpvotes': num_upvotes,
            'postNumDownvotes': num_downvotes,
            'postNumComments': num_comments,
            'postNumMedia': media_count
        }],
        'urlsList': urls_list,
        'postHasText': len(title) > 0 or len(selftext) > 0,
        'postHasImage': post_has_image,
        'postHasVideo': post_has_video,
        'postLocked': locked,
        'postRemovedCategory': removed_by_category
    }


def parse_arctic_comment(comment: dict, collection_time_iso: str) -> dict:
    """
    Parse an Arctic Shift comment into the standard data format.

    Args:
        comment: Raw comment data from Arctic Shift API
        collection_time_iso: ISO timestamp of when data was collected
    """
    comment_id = comment.get("id", "")
    author = comment.get("author", "")
    subreddit = comment.get("subreddit", "")
    body = comment.get("body", "") or ""
    created_utc = comment.get("created_utc", 0)
    score = comment.get("score", 0)
    permalink = comment.get("permalink", "")
    parent_id = comment.get("parent_id", "")[3:] if comment.get("parent_id") else ""
    subreddit_id = comment.get("subreddit_id", "")[3:] if comment.get("subreddit_id") else ""
    author_fullname = comment.get("author_fullname", "")[3:] if comment.get("author_fullname") else ""
    locked = comment.get("locked", False)

    # Build URLs
    comment_iso_time = convert_unix_to_iso(created_utc) if created_utc else ""
    redditor_url = f"https://reddit.com/user/{author}" if author else ""
    thread_url = f"https://reddit.com{permalink.rsplit('/', 2)[0]}/" if permalink else ""
    post_url = f"https://reddit.com{permalink}" if permalink else ""

    # Find URLs in content
    urls_list = re.findall(r'(https?://[^\s\)\]\}]+)', body) if body else []

    return {
        'platform': 'Forum',
        'postType': 'Forum_Comment',
        'postRawID': str(comment_id),
        'redditorID': str(author_fullname),
        'redditorName': str(author),
        'redditorURL': str(redditor_url),
        'redditorJoinedISOTime': None,  # Not available from Arctic Shift
        'redditorImageURL': "",
        'redditorSuspended': None,
        'redditorVerified': False,
        'redditorIsGold': False,
        'redditorPostKarma': 0,
        'redditorCommentKarma': 0,
        'postTitle': "",  # Not available for comments in Arctic Shift
        'postThreadUrl': str(thread_url),
        'postOrigThreadContent': "",  # Not available for comments in Arctic Shift
        'postUrl': str(post_url),
        'postISOTime': str(comment_iso_time),
        'postParentID': str(parent_id),
        'postSourceName': f"r/{subreddit}",
        'postSourceID': str(subreddit_id),
        'postFlairText': "",
        'postContent': str(body),
        'postCounterData': [{
            'collectionISOTime': str(collection_time_iso),
            'postScore': score,
            'postUpvoteRatio': 0.0,
            'postNumUpvotes': 0,
            'postNumDownvotes': 0,
            'postNumComments': 0,
            'postNumMedia': 0
        }],
        'urlsList': urls_list,
        'postHasText': len(body) > 0,
        'postHasImage': False,
        'postHasVideo': False,
        'postLocked': locked,
        'postRemovedCategory': ""
    }


def parse_arctic_posts(posts_response: dict) -> list[dict]:
    """Parse all posts from Arctic Shift API response."""
    singapore_tz = timezone(timedelta(hours=8))
    collection_time_iso = datetime.now(singapore_tz).isoformat(timespec='seconds')

    parsed = []
    for post in posts_response.get("data", []):
        parsed.append(parse_arctic_post(post, collection_time_iso))
    return parsed


def parse_arctic_comments(comments_response: dict) -> list[dict]:
    """Parse all comments from Arctic Shift API response."""
    singapore_tz = timezone(timedelta(hours=8))
    collection_time_iso = datetime.now(singapore_tz).isoformat(timespec='seconds')

    parsed = []
    for comment in comments_response.get("data", []):
        parsed.append(parse_arctic_comment(comment, collection_time_iso))
    return parsed


def process_subreddit(subreddit: str, after: int, before: int, output_dir: str, max_retries: int = 10) -> bool:
    """
    Process a single subreddit: fetch posts and comments, then save to JSON.

    Returns True if successful, False otherwise.
    """
    for attempt in range(1, max_retries + 1):
        try:
            # Fetch posts
            posts_raw = get_subreddit_posts(subreddit, after, before)
            posts_count = len(posts_raw.get('data', []))
            parsed_posts = parse_arctic_posts(posts_raw)

            # Fetch comments
            comments_raw = get_subreddit_comments(subreddit, after, before)
            comments_count = len(comments_raw.get('data', []))
            parsed_comments = parse_arctic_comments(comments_raw)

            # Save to JSON files
            with open(f"{output_dir}/{subreddit}_posts.json", "w", encoding="utf-8") as f:
                json.dump(parsed_posts, f, indent=2, ensure_ascii=False)

            with open(f"{output_dir}/{subreddit}_comments.json", "w", encoding="utf-8") as f:
                json.dump(parsed_comments, f, indent=2, ensure_ascii=False)

            print(f"  Saved {posts_count} posts, {comments_count} comments")
            return True

        except Exception as e:
            print(f"  Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(1)

    print(f"  Failed after {max_retries} attempts")
    return False


if __name__ == "__main__":
    output_dir = "subreddit_data"
    os.makedirs(output_dir, exist_ok=True)

    # Date range (millisecond timestamps)
    after = 1771459200000   # Feb 19, 2026
    before = 1772064000000  # Feb 26, 2026

    # Read subreddits from file
    subreddits = []
    with open("subreddits.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract subreddit name from URL (e.g. "https://www.reddit.com/r/Midjourney" -> "Midjourney")
                sub = line.rstrip("/").split("/r/")[-1]
                subreddits.append(sub)

    print(f"Found {len(subreddits)} subreddits to process")

    success_count = 0
    fail_count = 0

    for i, subreddit in enumerate(subreddits, 1):
        print(f"[{i}/{len(subreddits)}] Processing r/{subreddit}...")
        if process_subreddit(subreddit, after, before, output_dir):
            success_count += 1
        else:
            fail_count += 1
        time.sleep(1)  # Be nice to the API

    print(f"\nDone! Success: {success_count}, Failed: {fail_count}")
