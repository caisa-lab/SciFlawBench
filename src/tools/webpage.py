"""`visit_webpage`: smolagents' webpage fetcher, restricted to public web addresses."""

import ipaddress
import re
import socket
from collections.abc import Sequence
from urllib.parse import unquote, urljoin, urlsplit

import requests
from markdownify import markdownify
from smolagents import VisitWebpageTool

MAX_REDIRECTS = 5
TIMEOUT_S = 20
MAX_PAGE_BYTES = 10 * 1024 * 1024  # read at most this much of a page, so a huge response can't fill the memory
# URLs refused by default: the benchmark's public repository, its pages and copies hold the tasks and their answers.
BLOCKED_URLS = ("sciflawbench",)


class BlockedURLError(ValueError):
    """A URL that `visit_webpage` refuses to fetch."""


class PublicVisitWebpageTool(VisitWebpageTool):
    """smolagents' `VisitWebpageTool`, fetching only public web addresses.

    The upstream tool sends `requests.get(url)` for any URL the model gives it and follows redirects, so a task or an
    injected instruction could make the agent read the loopback interface, a private network or a cloud metadata
    service and pass the answer on (server-side request forgery). Here the URL must be http or https, and its host must
    resolve only to public addresses outside `blocked_networks`. Redirects are followed by hand, at most
    `MAX_REDIRECTS`, and every new URL is checked the same way. A refused URL returns an error message, as a failed
    fetch does. At most `MAX_PAGE_BYTES` of a page are read; the rest is converted and truncated exactly as by the
    upstream tool.

    Not covered: a host whose DNS answer changes between the check and the request (DNS rebinding), and internal
    services on public addresses (for example a university network reached over a VPN), unless their networks are
    listed in `blocked_networks`.

    A URL containing one of `blocked_urls` (case-insensitive, also after decoding %-escapes) is refused before any
    request, and so is a redirect to one. By default that is any URL with "sciflawbench" in it, as for the Tavily
    search: the benchmark's public repository holds the tasks and their answers. The refusal does not say why, so the
    agent is not told that it is being benchmarked.
    """

    def __init__(
        self,
        max_output_length: int = 40000,
        blocked_networks: Sequence[str] = (),
        blocked_urls: Sequence[str] = BLOCKED_URLS,
    ):
        super().__init__(max_output_length=max_output_length)
        self.blocked_networks = [ipaddress.ip_network(network, strict=False) for network in blocked_networks]
        self.blocked_urls = [pattern.lower() for pattern in blocked_urls if pattern]

    def forward(self, url: str) -> str:
        try:
            page, cut = self._fetch(url)
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdownify(page).strip())
            if cut:  # first, so a truncation of the content can't remove it
                note = f"_Only the first {MAX_PAGE_BYTES // 2**20} MB of this page were read._"
                markdown_content = f"{note}\n\n{markdown_content}"
            return self._truncate_content(markdown_content, self.max_output_length)
        except BlockedURLError as exc:
            return f"Error fetching the webpage: {exc}"
        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except requests.exceptions.RequestException as exc:
            return f"Error fetching the webpage: {exc}"
        except Exception as exc:
            return f"An unexpected error occurred: {exc}"

    def _fetch(self, url: str) -> tuple[str, bool]:
        """The page's text, and whether it was cut at `MAX_PAGE_BYTES`."""
        for _ in range(MAX_REDIRECTS + 1):
            self._check(url)
            with requests.get(url, timeout=TIMEOUT_S, allow_redirects=False, stream=True) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers["Location"])
                    continue
                response.raise_for_status()
                return self._read(response)
        raise requests.exceptions.TooManyRedirects(f"more than {MAX_REDIRECTS} redirects")

    @staticmethod
    def _read(response: requests.Response) -> tuple[str, bool]:
        data, cut = bytearray(), False
        for chunk in response.iter_content(chunk_size=64 * 1024):
            data += chunk
            if len(data) > MAX_PAGE_BYTES:
                data, cut = data[:MAX_PAGE_BYTES], True
                break
        try:
            return bytes(data).decode(response.encoding or "utf-8", errors="replace"), cut
        except LookupError:  # an encoding name Python doesn't know
            return bytes(data).decode("utf-8", errors="replace"), cut

    def _check(self, url: str) -> None:
        if any(pattern in unquote(url).lower() for pattern in self.blocked_urls):
            raise BlockedURLError(f"{url!r} is not available")
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise BlockedURLError(f"only http and https URLs can be visited, not {url!r}")
        host = parts.hostname
        if not host:
            raise BlockedURLError(f"{url!r} has no host")
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError as exc:
            raise BlockedURLError(f"{url!r} has an invalid port") from exc
        try:
            answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise requests.exceptions.ConnectionError(f"cannot resolve {host}: {exc}") from exc
        for answer in answers:
            address = ipaddress.ip_address(str(answer[4][0]).split("%")[0])  # drop an IPv6 zone such as %en0
            if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
                address = address.ipv4_mapped
            blocked = any(address in network for network in self.blocked_networks)
            if blocked or not address.is_global or address.is_multicast:
                raise BlockedURLError(f"{host} is not a public web address ({address})")
