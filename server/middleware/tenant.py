"""Tenant extraction middleware."""

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

DEFAULT_TENANT = "default"
TENANT_HEADER = "X-Tenant-ID"


class TenantMiddleware(BaseHTTPMiddleware):
    """Middleware to extract tenant ID from request headers.

    Extracts the tenant identifier from the X-Tenant-ID header and stores
    it in request.state for use by downstream handlers.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Extract tenant ID and add to request state.

        Args:
            request: The incoming request.
            call_next: The next middleware or route handler.

        Returns:
            The response from downstream handlers.
        """
        tenant_id = request.headers.get(TENANT_HEADER, DEFAULT_TENANT)
        request.state.tenant_id = tenant_id
        response = await call_next(request)
        return response


def get_tenant_id(request: Request) -> str:
    """Get tenant ID from request state.

    Args:
        request: The request with tenant state.

    Returns:
        Tenant identifier.
    """
    return getattr(request.state, "tenant_id", DEFAULT_TENANT)
