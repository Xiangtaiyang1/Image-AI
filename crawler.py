import os
import sys
import time
import hashlib
import requests
from pathlib import Path
from urllib.parse import urlparse, urlunparse


class AirplaneImageCrawler:
    """使用 Wikimedia Commons 官方 API 抓取真实飞机照片，带 429 限流自动退避。"""

    def __init__(self, output_dir='images-unmarked', max_images=100, delay=3, keywords=[], object=''):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_images = max_images
        self.delay = delay
        self.downloaded_hashes = set()
        self.api = 'https://commons.wikimedia.org/w/api.php'
        self.keywords = keywords
        self.object = object
        # 带完整 UA 和 cookie 的复用 session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/120.0.0.0 Safari/537.36'),
            'Referer': 'https://commons.wikimedia.org/',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        })
        self._load_existing_hashes()

    def _load_existing_hashes(self):
        for f in self.output_dir.glob('*'):
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                try:
                    self.downloaded_hashes.add(hashlib.md5(f.read_bytes()).hexdigest())
                except Exception:
                    pass
        print('loaded {} existing hashes'.format(len(self.downloaded_hashes)))

    def _clean_url(self, url):
        return urlunparse(urlparse(url)._replace(query=''))

    def _get(self, url, params=None, max_retries=5):
        """带 429 退避的 GET 请求。"""
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    print('      rate limited (429), wait {}s...'.format(wait))
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except requests.exceptions.HTTPError as e:
                if attempt == max_retries - 1:
                    return None
                time.sleep(10 * (attempt + 1))
            except Exception as e:
                if attempt == max_retries - 1:
                    return None
                time.sleep(5 * (attempt + 1))
        return None

    def search_wikimedia(self, query, limit=30):
        params = {
            'action': 'query',
            'generator': 'search',
            'gsrsearch': query,
            'gsrnamespace': '6',
            'gsrlimit': str(limit),
            'prop': 'imageinfo',
            'iiprop': 'url|size|mime',
            'iiurlwidth': '640',
            'format': 'json',
        }
        resp = self._get(self.api, params=params)
        if resp is None:
            print('  search failed for {}'.format(query))
            return []
        try:
            data = resp.json()
        except Exception:
            return []
        pages = data.get('query', {}).get('pages', {})
        urls = []
        for pid, page in pages.items():
            ii = page.get('imageinfo', [{}])
            if not ii:
                continue
            mime = ii[0].get('mime', '')
            thumb = ii[0].get('thumburl') or ii[0].get('url')
            if mime.startswith('image/') and thumb:
                urls.append(self._clean_url(thumb))
        return urls

    def _valid_image(self, content):
        if len(content) < 8000:
            return False
        if content[:3] == b'\xff\xd8\xff':
            return True
        if content[:4] == b'\x89PNG':
            return True
        return False

    def download_image(self, url, filename):
        resp = self._get(url)
        if resp is None:
            return False, 'download failed'
        if not resp.headers.get('Content-Type', '').startswith('image/'):
            return False, 'not an image'
        content = resp.content
        if not self._valid_image(content):
            return False, 'too small or invalid'
        h = hashlib.md5(content).hexdigest()
        if h in self.downloaded_hashes:
            return False, 'duplicate skipped'
        (self.output_dir / filename).write_bytes(content)
        self.downloaded_hashes.add(h)
        return True, 'ok'

    def crawl_airplane_images(self):
        total = 0
        seen_urls = set()
        for kw in self.keywords:
            if total >= self.max_images:
                break
            print('\nsearch keyword: {}'.format(kw))
            urls = self.search_wikimedia(kw, limit=25)
            new_urls = []
            for u in urls:
                if u not in seen_urls:
                    new_urls.append(u)
                    seen_urls.add(u)
            print('  got {} new unique URLs (total {})'.format(len(new_urls), len(seen_urls)))
            for url in new_urls:
                if total >= self.max_images:
                    break
                filename = '{}_{}_{}.jpg'.format(self.object, int(time.time() * 1000), total)
                ok, msg = self.download_image(url, filename)
                if ok:
                    total += 1
                    print('  [{}/{}] ok {}'.format(total, self.max_images, filename))
                else:
                    print('  [{}] skip: {}'.format(total, msg))
                time.sleep(self.delay)
        print('\ndone! total downloaded {} real {} images to {}'.format(total,self.object , self.output_dir))
        return total


def main():
    print('=' * 60)
    os.chdir(Path(__file__).parent.absolute())
    crawler = AirplaneImageCrawler(output_dir='images-unmarked-2', max_images=500, delay=4, object='car',
                                   keywords=[
                                       "car",
                                       "automobile",
                                       "vehicle",
                                       "sports car",
                                       "supercar",
                                       "muscle car",
                                       "luxury car",
                                       "sedan",
                                       "hatchback",
                                       "coupe",
                                       "convertible",
                                       "roadster",
                                       "SUV",
                                       "off-road vehicle",
                                       "pickup truck",
                                       "van",
                                       "minivan",
                                       "station wagon",
                                       "electric car",
                                       "hybrid car",
                                       "autonomous car",
                                       "classic car",
                                       "vintage car",
                                       "racing car",
                                       "Formula 1 car",
                                       "rally car",
                                       "concept car",
                                       "custom car",
                                       "lowrider",
                                       "hot rod"
                                   ])
    crawler.crawl_airplane_images()
    print('\ndone! output dir: {}'.format(crawler.output_dir))


if __name__ == '__main__':
    main()
