"""
WebSocket server — localhost only, Origin-validated.

Protocol (JSON):
  Browser → App   {"action":"status"}
  App → Browser   {"action":"status_result","ok":true,"ready":true}

  Browser → App   {"action":"enroll"}
  App → Browser   {"action":"enroll_result","ok":true,
                   "helper_data_b64":…,"mask_b64":…,"key_hash_b64":…}

  Browser → App   {"action":"verify","helper_data_b64":…,"mask_b64":…}
  App → Browser   {"action":"verify_result","ok":true,"c_prime_b64":…}

  App → Browser   {"action":"error","code":…,"message":…}
"""
import asyncio
import json
import threading
from typing import Callable

import websockets


class PendingAuth:
    def __init__(self, mode: str, data: dict):
        self.mode   = mode
        self.data   = data
        self.result = None
        self._event = threading.Event()

    def set_result(self, result: dict):
        self.result = result
        self._event.set()

    def wait(self, timeout: float = 60.0) -> dict | None:
        self._event.wait(timeout=timeout)
        return self.result


class WiFaKeyWSServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7825,
        allowed_origins: set[str] | None = None,
        on_auth_request: Callable[[PendingAuth], None] | None = None,
    ):
        self._host    = host
        self._port    = port
        self._origins = allowed_origins or set()
        self._on_auth = on_auth_request
        self._ready   = False

    def set_ready(self, ready: bool = True):
        self._ready = ready

    # ── Server lifecycle ──────────────────────────────────────────────────────

    async def serve_forever(self):
        async with websockets.serve(self._handle, self._host, self._port):
            print(f"[WS] Listening on ws://{self._host}:{self._port}")
            await asyncio.Future()  # run forever

    def start_in_thread(self):
        t = threading.Thread(target=self._thread_main, daemon=True, name="ws-server")
        t.start()

    def _thread_main(self):
        asyncio.run(self.serve_forever())

    # ── Connection handler ────────────────────────────────────────────────────

    async def _handle(self, ws):
        if not self._check_origin(ws):
            await ws.close(1008, "Forbidden origin")
            return

        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(ws, "invalid_json", "Invalid JSON")
                    continue

                await self._dispatch(ws, msg)
        except websockets.exceptions.ConnectionClosedOK:
            pass

    async def _dispatch(self, ws, msg: dict):
        action = msg.get("action", "")

        if action == "status":
            await ws.send(json.dumps({
                "action": "status_result",
                "ok": True,
                "ready": self._ready,
            }))
            return

        if not self._ready:
            await self._send_error(ws, "not_ready", "Models still loading")
            return

        if action == "enroll":
            await self._handle_auth(ws, "enroll", {})

        elif action == "verify":
            for field in ("helper_data_b64", "mask_b64"):
                if field not in msg:
                    await self._send_error(ws, "missing_field", f"Missing: {field}")
                    return
            await self._handle_auth(ws, "verify", {
                "helper_data_b64": msg["helper_data_b64"],
                "mask_b64":        msg["mask_b64"],
            })

        else:
            await self._send_error(ws, "unknown_action", f"Unknown action: {action}")

    async def _handle_auth(self, ws, mode: str, data: dict):
        if self._on_auth is None:
            await self._send_error(ws, "not_configured", "Auth handler not set")
            return

        pending = PendingAuth(mode, data)
        self._on_auth(pending)  # Qt signal emitted from this thread

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, pending.wait, 60.0)

        if result is None:
            await self._send_error(ws, "timeout", "Auth timed out")
            return

        if result.get("ok"):
            await ws.send(json.dumps(result))
        else:
            await self._send_error(
                ws,
                result.get("code", "auth_failed"),
                result.get("message", "Authentication failed"),
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_origin(self, ws) -> bool:
        if not self._origins:
            return True
        # websockets 14+ uses ws.request.headers; older uses ws.request_headers
        try:
            origin = ws.request.headers.get("Origin", "")
        except AttributeError:
            origin = ws.request_headers.get("Origin", "")
        return origin in self._origins

    @staticmethod
    async def _send_error(ws, code: str, message: str):
        await ws.send(json.dumps({"action": "error", "code": code, "message": message}))
