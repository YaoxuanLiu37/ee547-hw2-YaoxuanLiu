#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote
import json, re, sys, os
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "sample_data")
PAPERS_FILE = os.path.join(DATA_DIR, "papers.json")
CORPUS_FILE = os.path.join(DATA_DIR, "corpus_analysis.json")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _warn(msg):
    sys.stderr.write(f"[{_ts()}] WARNING: {msg}\n")

def _info(msg):
    sys.stdout.write(f"[{_ts()}] INFO: {msg}\n")

def _read_json_if_exists(path, expected_type=None):
    if not os.path.exists(path):
        _warn(f"Data file not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if expected_type and not isinstance(data, expected_type):
            _warn(f"{os.path.basename(path)} has unexpected type: {type(data).__name__}, expected {expected_type.__name__}. Ignoring.")
            return None
        return data
    except Exception as e:
        _warn(f"Failed to read {path}: {e}")
        return None

def load_data():
    papers = _read_json_if_exists(PAPERS_FILE, list) or []
    corpus = _read_json_if_exists(CORPUS_FILE, dict) or {}
    missing_core = 0
    for p in papers:
        if not isinstance(p, dict) or not all(k in p for k in ("arxiv_id", "title", "authors", "categories")):
            missing_core += 1
    if missing_core:
        _warn(f"{missing_core} records missing core fields (arxiv_id/title/authors/categories).")
    _info(f"Loaded papers={len(papers)}; corpus_keys={len(corpus)}")
    return papers, corpus

PAPERS, CORPUS = load_data()

def _json_bytes(obj, status=200):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return data, status, "application/json; charset=utf-8"

def _not_found(msg="Not Found"):
    return _json_bytes({"error": msg}, 404)

def _bad_request(msg="Bad Request"):
    return _json_bytes({"error": msg}, 400)

def _server_error(msg="Internal Server Error"):
    return _json_bytes({"error": msg}, 500)

def _tokenize(text):
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(text)]

def _count_term_freq(haystack_tokens, query_terms):
    if not haystack_tokens or not query_terms:
        return 0
    freq = 0
    for term in query_terms:
        freq += sum(1 for t in haystack_tokens if t == term)
    return freq

def _categories_from_papers(papers):
    dist = {}
    for p in papers:
        for c in (p.get("categories") or []):
            dist[c] = dist.get(c, 0) + 1
    return dist

def _stats_response():
    total_papers = len(PAPERS)
    total_words = None
    unique_words = None
    top_10_words = []
    if isinstance(CORPUS, dict):
        total_words = CORPUS.get("total_words")
        unique_words = CORPUS.get("unique_words")
        top_words = CORPUS.get("top_words") or CORPUS.get("top_10_words")
        if isinstance(top_words, list):
            for item in top_words[:10]:
                if isinstance(item, dict) and "word" in item and "frequency" in item:
                    top_10_words.append({"word": item["word"], "frequency": item["frequency"]})
                elif isinstance(item, list) and len(item) >= 2:
                    top_10_words.append({"word": item[0], "frequency": item[1]})
    if total_words is None or unique_words is None or not top_10_words:
        from collections import Counter
        tokens = []
        for p in PAPERS:
            tokens.extend(_tokenize(p.get("abstract", "")))
        counter = Counter(tokens)
        total_words = sum(counter.values())
        unique_words = len(counter)
        top_10_words = [{"word": w, "frequency": f} for w, f in counter.most_common(10)]
    category_distribution = _categories_from_papers(PAPERS)
    return {
        "total_papers": total_papers,
        "total_words": total_words or 0,
        "unique_words": unique_words or 0,
        "top_10_words": top_10_words,
        "category_distribution": category_distribution,
    }

class ArxivHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send(self, body, status, content_type, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _log_req(self, path, status, extra=""):
        status_text = f"{status} " + ("OK" if status == 200 else "Not Found" if status == 404 else "Bad Request" if status == 400 else "Internal Error" if status >= 500 else "")
        msg = f"[{_ts()}] {self.command} {path} - {status_text}"
        if extra:
            msg += f" {extra}"
        print(msg)

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path or "/"
            q = parse_qs(parsed.query)
            if path == "/papers":
                return self._handle_list_papers()
            if path.startswith("/papers/"):
                arxiv_id = unquote(path[len("/papers/"):]).strip()
                if not arxiv_id:
                    body, status, ct = _bad_request("Missing arxiv_id.")
                    self._send(body, status, ct)
                    self._log_req(self.path, status)
                    return
                return self._handle_get_paper(arxiv_id)
            if path == "/search":
                return self._handle_search(q)
            if path == "/stats":
                return self._handle_stats()
            body, status, ct = _not_found("Endpoint not found.")
            self._send(body, status, ct)
            self._log_req(self.path, status)
        except Exception as e:
            body, status, ct = _server_error(str(e))
            self._send(body, status, ct)
            self._log_req(self.path, status)

    def _handle_list_papers(self):
        items = []
        for p in PAPERS:
            if not isinstance(p, dict):
                continue
            items.append({
                "arxiv_id": p.get("arxiv_id"),
                "title": p.get("title"),
                "authors": p.get("authors") or [],
                "categories": p.get("categories") or [],
            })
        body, status, ct = _json_bytes(items, 200)
        self._send(body, status, ct)
        self._log_req(self.path, status, f"({len(items)} results)")

    def _handle_get_paper(self, arxiv_id):
        found = None
        for p in PAPERS:
            if isinstance(p, dict) and str(p.get("arxiv_id")) == arxiv_id:
                found = p
                break
        if not found:
            body, status, ct = _not_found(f"Paper not found: {arxiv_id}")
            self._send(body, status, ct)
            self._log_req(self.path, status)
            return
        resp = {
            "arxiv_id": found.get("arxiv_id"),
            "title": found.get("title"),
            "authors": found.get("authors") or [],
            "abstract": found.get("abstract"),
            "categories": found.get("categories") or [],
            "published": found.get("published"),
            "abstract_stats": found.get("abstract_stats") or self._derive_abstract_stats(found.get("abstract") or "")
        }
        body, status, ct = _json_bytes(resp, 200)
        self._send(body, status, ct)
        self._log_req(self.path, status)

    def _derive_abstract_stats(self, abstract):
        tokens = _tokenize(abstract)
        total_words = len(tokens)
        unique_words = len(set(tokens))
        sentences = re.split(r"[.!?]+", abstract.strip())
        sentences = [s for s in sentences if s.strip()]
        return {"total_words": total_words, "unique_words": unique_words, "total_sentences": len(sentences)}

    def _handle_search(self, qdict):
        if "q" not in qdict or not qdict["q"]:
            body, status, ct = _bad_request("Missing query parameter q.")
            self._send(body, status, ct)
            self._log_req(self.path, status)
            return
        query_raw = qdict["q"][0].strip()
        if not query_raw:
            body, status, ct = _bad_request("Empty query.")
            self._send(body, status, ct)
            self._log_req(self.path, status)
            return
        terms = [t for t in _tokenize(query_raw) if t]
        if not terms:
            body, status, ct = _bad_request("Malformed query; no valid terms.")
            self._send(body, status, ct)
            self._log_req(self.path, status)
            return
        results = []
        for p in PAPERS:
            title = p.get("title") or ""
            abstract = p.get("abstract") or ""
            title_tokens = _tokenize(title)
            abstract_tokens = _tokenize(abstract)
            combined_tokens = title_tokens + abstract_tokens
            if not all(term in combined_tokens for term in terms):
                continue
            score = _count_term_freq(combined_tokens, terms)
            matches_in = []
            if any(term in title_tokens for term in terms):
                matches_in.append("title")
            if any(term in abstract_tokens for term in terms):
                matches_in.append("abstract")
            results.append({
                "arxiv_id": p.get("arxiv_id"),
                "title": title,
                "match_score": int(score),
                "matches_in": matches_in or [],
            })
        resp = {"query": query_raw, "results": results}
        body, status, ct = _json_bytes(resp, 200)
        self._send(body, status, ct)
        self._log_req(self.path, status, f"({len(results)} results)")

    def _handle_stats(self):
        resp = _stats_response()
        body, status, ct = _json_bytes(resp, 200)
        self._send(body, status, ct)
        self._log_req(self.path, status)

def main():
    port = 8080
    if len(sys.argv) >= 2:
        try:
            port = int(sys.argv[1])
        except Exception:
            _warn(f"Invalid port '{sys.argv[1]}', falling back to 8080.")
            port = 8080
    server = ThreadingHTTPServer(("0.0.0.0", port), ArxivHandler)
    _info(f"Server starting on port {port}. Data dir: {DATA_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _info("Server stopped.")

if __name__ == "__main__":
    main()
