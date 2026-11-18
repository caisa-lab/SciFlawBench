"""`visit_webpage`: smolagents' webpage fetcher, restricted to public web addresses."""

import ipaddress
import re
import socket
from collections.abc import Sequence
from urllib.parse import urljoin, urlsplit

import requests
from markdownify import markdownify
from smolagents import VisitWebpageTool

MAX_REDIRECTS = 5
TIMEOUT_S = 20


class BlockedURLError(ValueError):
    """A URL that `visit_webpage` refuses to fetch."""


class PublicVisitWebpageTool(VisitWebpageTool):
    """smolagents' `VisitWebpageTool`, fetching only public web addresses.

    The upstream tool sends `requests.get(url)` for any URL the model gives it and follows redirects, so a task or an
    injected instruction could make the agent read the loopback interface, a private network or a cloud metadata
    service and pass the answer on (server-side request forgery). Here the URL must be http or https, and its host must
    resolve only to public addresses outside `blocked_networks`. Redirects are followed by hand, at most
    `MAX_REDIRECTS`, and every new URL is checked the same way. A refused URL returns an error message, as a failed
    fetch does; a page that is fetched is converted and truncated exactly as by the upstream tool.

    Not covered: a host whose DNS answer changes between the check and the request (DNS rebinding), and internal
    services on public addresses (for example a university network reached over a VPN), unless their networks are
    listed in `blocked_networks`.
    """

    def __init__(self, max_output_length: int = 40000, blocked_networks: Sequence[str] = ()):
        super().__init__(max_output_length=max_output_length)
        self.blocked_networks = [ipaddress.ip_network(network, strict=False) for network in blocked_networks]

    def forward(self, url: str) -> str:
        try:
            response = self._fetch(url)
            response.raise_for_status()
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdownify(response.text).strip())
            return self._truncate_content(markdown_content, self.max_output_length)
        except BlockedURLError as exc:
            return f"Error fetching the webpage: {exc}"
        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except requests.exceptions.RequestException as exc:
            return f"Error fetching the webpage: {exc}"
        except Exception as exc:
            return f"An unexpected error occurred: {exc}"

    def _fetch(self, url: str) -> requests.Response:
        for _ in range(MAX_REDIRECTS + 1):
            self._check(url)
            response = requests.get(url, timeout=TIMEOUT_S, allow_redirects=False)
            if not response.is_redirect:
                return response
            url = urljoin(url, response.headers["Location"])
        raise requests.exceptions.TooManyRedirects(f"more than {MAX_REDIRECTS} redirects")

    def _check(self, url: str) -> None:
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
