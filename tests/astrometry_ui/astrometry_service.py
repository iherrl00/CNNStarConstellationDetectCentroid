import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QWidget


@dataclass
class AstrometryConfig:
    api_url: str
    api_key: str
    publicly_visible: str
    allow_commercial_use: str
    allow_modifications: str
    submission_timeout_sec: int
    job_timeout_sec: int


class AstrometryWorker(QThread):
    log_message = pyqtSignal(str)
    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, config: AstrometryConfig, fits_bytes: bytes, filename: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config = config
        self.fits_bytes = fits_bytes
        self.filename = filename
        self.api_url = self.config.api_url.rstrip("/")
        self.headers = {"Referer": "https://nova.astrometry.net/api/login"}

    def run(self) -> None:
        try:
            session = self._login()
            subid = self._upload(session)
            job_id = self._wait_for_job_id(subid)
            final_status = self._wait_for_job_status(job_id)
            result = self._collect_all_information(subid, job_id, session, final_status)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _login(self) -> str:
        self.log_message.emit("Astrometry: autenticando")
        payload = {"apikey": self.config.api_key}
        response = requests.post(
            f"{self.api_url}/login",
            data={"request-json": json.dumps(payload)},
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        session = data.get("session")
        if not session:
            raise RuntimeError(f"Respuesta de login inválida: {data}")
        return session

    def _upload(self, session: str) -> int:
        self.log_message.emit("Astrometry: enviando archivo")
        request_json = {
            "session": session,
            "publicly_visible": self.config.publicly_visible,
            "allow_commercial_use": self.config.allow_commercial_use,
            "allow_modifications": self.config.allow_modifications,
            "crpix_center": True,
        }
        files = {
            "request-json": (None, json.dumps(request_json), "text/plain"),
            "file": (self.filename, self.fits_bytes, "application/octet-stream"),
        }
        response = requests.post(f"{self.api_url}/upload", files=files, headers=self.headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        subid = data.get("subid")
        if not subid:
            raise RuntimeError(f"Upload sin subid: {data}")
        return int(subid)

    def _wait_for_job_id(self, subid: int) -> int:
        self.log_message.emit(f"Astrometry: esperando job para subid={subid}")
        start = time.monotonic()
        while time.monotonic() - start <= self.config.submission_timeout_sec:
            data = self._safe_get_json(f"/submissions/{subid}")
            jobs = [job for job in data.get("jobs", []) if job]
            if jobs:
                return int(jobs[0])
            time.sleep(3)
        raise TimeoutError("Timeout esperando job_id")

    def _wait_for_job_status(self, job_id: int) -> str:
        self.log_message.emit(f"Astrometry: resolviendo job={job_id}")
        start = time.monotonic()
        last_status = "unknown"
        while time.monotonic() - start <= self.config.job_timeout_sec:
            data = self._safe_get_json(f"/jobs/{job_id}")
            status = str(data.get("status", "unknown"))
            last_status = status
            if status in {"success", "failure"}:
                return status
            time.sleep(3)
        return last_status

    def _collect_all_information(self, subid: int, job_id: int, session: str, final_status: str) -> Dict[str, Any]:
        self.log_message.emit("Astrometry: recuperando resultados")
        payloads: Dict[str, Any] = {
            "submission": self._safe_get_json(f"/submissions/{subid}"),
            "job": self._safe_get_json(f"/jobs/{job_id}"),
            "calibration": self._safe_get_json(f"/jobs/{job_id}/calibration"),
            "tags": self._safe_get_json(f"/jobs/{job_id}/tags"),
            "machine_tags": self._safe_get_json(f"/jobs/{job_id}/machine_tags"),
            "objects_in_field": self._safe_get_json(f"/jobs/{job_id}/objects_in_field"),
            "annotations": self._safe_get_json(f"/jobs/{job_id}/annotations"),
            "info": self._safe_get_json(f"/jobs/{job_id}/info"),
        }
        download_urls = {
            "wcs_file": f"{self._site_root()}/wcs_file/{job_id}",
            "new_fits_file": f"{self._site_root()}/new_fits_file/{job_id}",
            "annotated_display": f"{self._site_root()}/annotated_display/{job_id}",
            "extraction_image_display": f"{self._site_root()}/extraction_image_display/{job_id}",
        }
        return {
            "status": final_status,
            "session": session,
            "subid": subid,
            "job_id": job_id,
            "download_urls": download_urls,
            "api_payloads": payloads,
        }

    def _safe_get_json(self, endpoint: str) -> Dict[str, Any]:
        url = f"{self.api_url}{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            return {"_error": str(exc), "_url": url}

    def _site_root(self) -> str:
        if self.api_url.endswith("/api"):
            return self.api_url[:-4]
        return "http://nova.astrometry.net"
