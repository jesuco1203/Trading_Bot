"""Mini interfaz web y worker para configurar alertas 3h/4h."""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from monitoring.telegram_alerts import evaluate_candles, fetch_confirmed_candles, fetch_current_price, format_alert, send_telegram
from scripts.alert_4h import next_candle_close
import time


CONFIG_PATH = Path(os.environ.get("ALERT_CONFIG_PATH", "data/alerts.json"))
HOST = os.environ.get("ALERT_HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
CHECK_DELAY = int(os.environ.get("ALERT_DELAY_SECONDS", "10"))


def load_config() -> list[dict]:
    if not CONFIG_PATH.exists():
        return []
    try:
        value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except (OSError, json.JSONDecodeError):
        logging.exception("No se pudo leer %s", CONFIG_PATH)
        return []


def save_config(alerts: list[dict]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = CONFIG_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(alerts, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(CONFIG_PATH)


HTML = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alertas Trading</title><style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;background:#101318;color:#eef2f7}
main{background:#1a2029;padding:1.4rem;border-radius:14px}h1{margin-top:0;font-size:1.35rem}
label{display:block;margin:.8rem 0 .25rem;color:#aeb8c6}input,select,button{width:100%;box-sizing:border-box;padding:.7rem;border-radius:8px;border:1px solid #3c4654;background:#11161d;color:#fff;font-size:1rem}
button{margin-top:1rem;background:#2e83f7;border:0;cursor:pointer;font-weight:600}.row{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.alert{display:flex;justify-content:space-between;gap:1rem;border-top:1px solid #38414e;padding:1rem 0}.muted{color:#aeb8c6;font-size:.9rem}.danger{background:#933d4c;width:auto;margin:0;padding:.45rem .7rem}
</style></head><body><main><h1>Alertas de velas</h1><p class="muted">La comprobación se realiza una vez después de cada cierre de 3h o 4h en UTC.</p><p>Precio actual: <strong id="current-price">—</strong> <span id="price-status" class="muted"></span></p>
<form id="form"><div class="row"><div><label>Símbolo</label><input name="symbol" value="BTC-USDT-SWAP" required></div><div><label>Timeframe</label><select name="timeframe"><option value="4h">4 horas</option><option value="3h">3 horas</option></select></div></div><div class="row"><div><label>Precio objetivo</label><input name="price" type="number" step="any" required></div><div><label>Condición</label><select name="direction"><option value="above">Cerrar por encima</option><option value="below">Cerrar por debajo</option></select></div></div>
<div class="row"><div><label>Patrón</label><select name="require_engulfing"><option value="true">Exigir envolvente</option><option value="false">Solo nivel de precio</option></select></div><div></div></div>
<button>Guardar alerta</button></form><section id="list"></section></main><script>
const list=document.querySelector('#list');const form=document.querySelector('#form');
async function load(){const r=await fetch('/api/alerts');const a=await r.json();list.innerHTML=a.length?'<h2>Alertas configuradas</h2>'+a.map((x,i)=>`<div class="alert"><div><b>${x.symbol}</b> · <b>${x.timeframe||'4h'}</b><br>${x.direction==='above'?'Por encima':'Por debajo'} de <b>${x.price}</b><br><span class="muted">${x.require_engulfing?'Con envolvente':'Solo precio'} · ${x.enabled?'Activa':'Pausada'}</span></div><button class="danger" onclick="removeAlert(${i})">Eliminar</button></div>`).join(''):'<p class="muted">No hay alertas configuradas.</p>'}
async function updatePrice(){const symbol=form.elements.symbol.value.trim();if(!symbol)return;document.querySelector('#price-status').textContent='consultando…';try{const r=await fetch('/api/price?symbol='+encodeURIComponent(symbol));const d=await r.json();if(!r.ok)throw new Error(d.error);document.querySelector('#current-price').textContent=Number(d.price).toLocaleString('en-US',{maximumFractionDigits:8});document.querySelector('#price-status').textContent='OKX · actualizado '+new Date().toLocaleTimeString()}catch(e){document.querySelector('#current-price').textContent='—';document.querySelector('#price-status').textContent=e.message}}
form.elements.symbol.addEventListener('change',updatePrice);form.elements.symbol.addEventListener('blur',updatePrice);setInterval(updatePrice,3000);updatePrice();
form.onsubmit=async e=>{e.preventDefault();const d=Object.fromEntries(new FormData(form));d.price=Number(d.price);d.require_engulfing=d.require_engulfing==='true';d.enabled=true;await fetch('/api/alerts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});form.reset();updatePrice();load()};
async function removeAlert(i){await fetch('/api/alerts/'+i,{method:'DELETE'});load()}load();
</script></body></html>"""


def authorized(handler: BaseHTTPRequestHandler) -> bool:
    password = os.environ.get("APP_PASSWORD")
    if not password:
        return True
    value = handler.headers.get("Authorization", "")
    try:
        scheme, encoded = value.split(" ", 1)
        decoded = base64.b64decode(encoded).decode("utf-8")
        return scheme.lower() == "basic" and decoded == f"admin:{password}"
    except (ValueError, UnicodeDecodeError):
        return False


class Handler(BaseHTTPRequestHandler):
    def _auth(self) -> bool:
        if authorized(self):
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Trading alerts"')
        self.end_headers()
        return False

    def _json(self, value: object, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if not self._auth(): return
        parsed = urlparse(self.path)
        if parsed.path == "/api/alerts": return self._json(load_config())
        if parsed.path == "/api/price":
            from urllib.parse import parse_qs
            symbol = parse_qs(parsed.query).get("symbol", [""])[0].strip()
            if not symbol: return self._json({"error": "símbolo requerido"}, 400)
            try:
                price = fetch_current_price(symbol)
                return self._json({"symbol": symbol, "price": price})
            except Exception: return self._json({"error": "no se pudo consultar el precio"}, 502)
        body = HTML.encode("utf-8")
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self) -> None:
        if not self._auth(): return
        if urlparse(self.path).path != "/api/alerts": return self._json({"error":"not found"}, 404)
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if data.get("direction") not in ("above", "below") or data.get("timeframe") not in ("3h", "4h") or not data.get("symbol") or float(data["price"]) <= 0:
                raise ValueError
            data["price"] = float(data["price"]); data["enabled"] = bool(data.get("enabled", True))
            alerts = load_config(); alerts.append(data); save_config(alerts); self._json(data, 201)
        except (ValueError, TypeError, json.JSONDecodeError): self._json({"error":"configuración inválida"}, 400)

    def do_DELETE(self) -> None:
        if not self._auth(): return
        parts = urlparse(self.path).path.split("/")
        if len(parts) != 4 or parts[:3] != ["", "api", "alerts"]: return self._json({"error":"not found"}, 404)
        try: index = int(parts[3]); alerts = load_config(); alerts.pop(index); save_config(alerts); self._json({"ok":True})
        except (ValueError, IndexError): self._json({"error":"alerta no encontrada"}, 404)

    def log_message(self, fmt: str, *args: object) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)


def worker() -> None:
    while True:
        items = [item for item in load_config() if item.get("enabled", True)]
        timeframes = {item.get("timeframe", "4h") for item in items}
        targets = {timeframe: next_candle_close(timeframe) for timeframe in timeframes or {"4h"}}
        target = min(targets.values())
        time.sleep(max(0, (target.timestamp() + CHECK_DELAY) - time.time()))
        due_timeframes = {timeframe for timeframe, scheduled in targets.items() if scheduled == target}
        for item in items:
            timeframe = item.get("timeframe", "4h")
            if timeframe not in due_timeframes: continue
            try:
                candles = fetch_confirmed_candles(item["symbol"], timeframe, 5)
                alert = evaluate_candles(item["symbol"], candles, level=item.get("price"), level_direction=item.get("direction"), require_engulfing=item.get("require_engulfing", True), timeframe=timeframe)
                if alert:
                    send_telegram(format_alert(alert)); logging.info("Alerta enviada: %s", item["symbol"])
            except Exception: logging.exception("Error procesando %s", item.get("symbol"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    threading.Thread(target=worker, daemon=True, name="4h-alert-worker").start()
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
