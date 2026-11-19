import ipaddress
import socket

import pytest
import requests

from tools.definitions import tool_registry
from tools.webpage import MAX_PAGE_BYTES, MAX_REDIRECTS, PublicVisitWebpageTool

DNS = {
    "example.org": ["93.184.215.14"],
    "localhost": ["127.0.0.1"],
    "metadata.example": ["169.254.169.254"],
    "split.example": ["93.184.215.14", "10.0.0.5"],  # one public and one private answer
    "gateway.example": ["93.184.216.80"],  # public, but on a network a lab may want to block
}


class _FakeReply:
    def __init__(self, text="", status_code=200, location=None, encoding="utf-8"):
        self.content, self.status_code, self.encoding = text.encode(encoding), status_code, encoding
        self.headers = {"Location": location} if location else {}
        self.chunks_read = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size):
        for start in range(0, len(self.content), chunk_size):
            self.chunks_read += 1
            yield self.content[start : start + chunk_size]

    @property
    def is_redirect(self):
        return "Location" in self.headers and self.status_code in (301, 302, 303, 307, 308)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


@pytest.fixture
def web(monkeypatch):
    """Fake DNS and HTTP: returns the requests made and a list to fill with replies."""

    def fake_getaddrinfo(host, port, type=0):
        try:
            addresses = [str(ipaddress.ip_address(host))]
        except ValueError:
            if host not in DNS:
                raise socket.gaierror(f"unknown host {host}") from None
            addresses = DNS[host]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addresses]

    requests_made, replies = [], []

    def fake_get(url, timeout, allow_redirects, stream):
        requests_made.append({"url": url, "timeout": timeout, "allow_redirects": allow_redirects, "stream": stream})
        return replies.pop(0)

    monkeypatch.setattr("tools.webpage.socket.getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr("tools.webpage.requests.get", fake_get)
    return requests_made, replies


def test_a_public_page_is_fetched_and_converted_to_markdown(web):
    requests_made, replies = web
    replies.append(_FakeReply("<h1>GW190521</h1><p>A heavy merger.</p>"))
    res = PublicVisitWebpageTool().forward("https://example.org/gw190521")
    assert "GW190521" in res and "A heavy merger." in res
    assert requests_made == [
        {"url": "https://example.org/gw190521", "timeout": 20, "allow_redirects": False, "stream": True}
    ]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost:8888/tree",
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.example/",
        "http://10.1.2.3/",
        "http://192.168.0.1/admin",
        "http://100.64.0.1/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://224.0.0.251/",
        "http://split.example/",
    ],
)
def test_a_non_public_address_is_refused_without_a_request(web, url):
    requests_made, _ = web
    res = PublicVisitWebpageTool().forward(url)
    assert res.startswith("Error fetching the webpage:") and "not a public web address" in res
    assert requests_made == []


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.org/x", "gopher://example.org/", "http:///x"])
def test_only_http_and_https_urls_with_a_host_are_visited(web, url):
    requests_made, _ = web
    assert PublicVisitWebpageTool().forward(url).startswith("Error fetching the webpage:")
    assert requests_made == []


def test_a_redirect_to_an_internal_address_is_refused(web):
    requests_made, replies = web
    replies.append(_FakeReply(status_code=302, location="http://169.254.169.254/latest/meta-data/"))
    res = PublicVisitWebpageTool().forward("https://example.org/redirect")
    assert "not a public web address (169.254.169.254)" in res
    assert len(requests_made) == 1


def test_a_redirect_to_a_public_page_is_followed(web):
    requests_made, replies = web
    replies.extend([_FakeReply(status_code=301, location="/moved"), _FakeReply("<p>moved here</p>")])
    assert "moved here" in PublicVisitWebpageTool().forward("https://example.org/old")
    assert [r["url"] for r in requests_made] == ["https://example.org/old", "https://example.org/moved"]


def test_too_many_redirects_end_with_an_error(web):
    requests_made, replies = web
    replies.extend([_FakeReply(status_code=302, location="/again")] * (MAX_REDIRECTS + 1))
    res = PublicVisitWebpageTool().forward("https://example.org/loop")
    assert res.startswith("Error fetching the webpage:") and "redirects" in res
    assert len(requests_made) == MAX_REDIRECTS + 1


def test_blocked_networks_refuse_public_addresses_too(web):
    requests_made, replies = web
    tool = PublicVisitWebpageTool(blocked_networks=["93.184.216.0/24"])
    assert "not a public web address (93.184.216.80)" in tool.forward("http://gateway.example:8080/")
    assert requests_made == []
    replies.append(_FakeReply("<p>reachable</p>"))
    assert "reachable" in PublicVisitWebpageTool().forward("http://gateway.example:8080/")


def test_an_unknown_host_is_reported_like_a_failed_fetch(web):
    res = PublicVisitWebpageTool().forward("https://no-such-host.example/")
    assert res.startswith("Error fetching the webpage: cannot resolve no-such-host.example")


def test_the_registry_builds_the_public_only_tool(monkeypatch):
    tool = tool_registry.get("visit_webpage")(watcher=None, blocked_networks=["93.184.216.0/24"])._wrapped
    assert isinstance(tool, PublicVisitWebpageTool)
    assert tool.name == "visit_webpage"
    assert tool.blocked_networks == [ipaddress.ip_network("93.184.216.0/24")]
    assert tool_registry.get("visit_webpage")(watcher=None)._wrapped.max_output_length == 10**9


def test_only_the_first_part_of_a_huge_page_is_read(web):
    _, replies = web
    huge = _FakeReply("<p>" + "x" * (3 * MAX_PAGE_BYTES) + "</p>")
    replies.append(huge)
    res = PublicVisitWebpageTool().forward("https://example.org/huge")
    assert res.startswith(f"_Only the first {MAX_PAGE_BYTES // 2**20} MB of this page were read._")
    assert huge.chunks_read * 64 * 1024 < 2 * MAX_PAGE_BYTES  # it stopped reading, not read it all and cut


def test_the_page_is_decoded_with_its_declared_encoding(web):
    _, replies = web
    replies.append(_FakeReply("<p>Gr\u00fc\u00dfe</p>", encoding="latin-1"))
    assert "Gr\u00fc\u00dfe" in PublicVisitWebpageTool().forward("https://example.org/latin")
