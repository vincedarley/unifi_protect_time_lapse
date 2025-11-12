# app/camera_manager.py

import httpx  # type: ignore
import asyncio
import aiohttp # type: ignore
import requests
import logging
import hashlib
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

import config

# Disable SSL warnings if SSL verification is disabled
if not config.UNIFI_PROTECT_VERIFY_SSL:
    import urllib3  # type: ignore

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class Camera:
    """Data class representing a camera."""

    id: str
    name: str
    state: str
    type: str
    mac: str
    firmware_version: str
    is_connected: bool
    is_recording: bool
    supports_full_hd_snapshot: bool = False

    @classmethod
    def from_api_response(cls, camera_data: Dict[str, Any]) -> "Camera":
        """Create Camera instance from API response data."""
        # Extract feature flags
        feature_flags = camera_data.get("featureFlags", {})
        supports_full_hd = feature_flags.get("supportFullHdSnapshot", False)

        return cls(
            id=camera_data.get("id", ""),
            name=camera_data.get("name", ""),
            state=camera_data.get("state", ""),
            type=camera_data.get("type", ""),
            mac=camera_data.get("mac", ""),
            firmware_version=camera_data.get("firmwareVersion", ""),
            is_connected=camera_data.get("state") == "CONNECTED",
            is_recording=camera_data.get("isRecording", False),
            supports_full_hd_snapshot=supports_full_hd,
        )

    @property
    def safe_name(self) -> str:
        """Return a filesystem-safe version of the camera name."""
        return self.name.replace(" ", "_").replace("/", "_").replace("\\", "_")

    def get_deterministic_offset(self, offset_seconds: int) -> int:
        """
        Get consistent offset for this camera based on camera ID hash.

        Uses camera ID (immutable) instead of name (can be changed).
        This ensures the same camera always gets the same offset,
        even across container restarts and camera list changes.

        Args:
            offset_seconds: Seconds between offset slots

        Returns:
            Offset in seconds (0 to 59)
        """
        # Handle case where no distribution is needed
        if offset_seconds <= 0:
            return 0

        # Use camera ID for maximum stability - never changes
        hash_obj = hashlib.sha1(self.id.encode("utf-8"))
        hash_int = int(hash_obj.hexdigest(), 16)

        # Calculate number of possible offset slots based on config
        slots = config.FETCH_DISTRIBUTION_WINDOW_SECONDS // offset_seconds
        slot = hash_int % slots

        return slot * offset_seconds


class CameraManager:
    """Manages camera discovery and snapshot capture."""

    def __init__(self):
        self.client: httpx.AsyncClient
        self.cameras: List[Camera] = []
        self.last_camera_refresh = None
        self.camera_refresh_interval = config.CAMERA_REFRESH_INTERVAL

        # Distribution settings locked at startup - NEVER change during runtime
        self._distribution_locked = False
        self._locked_total_cameras = 0
        self._locked_use_distribution = False
        self._locked_optimal_offset = 0
        # Track in-flight captures to avoid duplicate work (camera.id, preset_name, timestamp)
        self._inflight_captures: set[tuple[str, str | None, int]] = set()
        self._inflight_lock = asyncio.Lock()

    async def __aenter__(self):
        """Async context manager entry."""
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        timeout = httpx.Timeout(config.UNIFI_PROTECT_REQUEST_TIMEOUT)

        self.client = httpx.AsyncClient(
            verify=config.UNIFI_PROTECT_VERIFY_SSL,
            limits=limits,
            timeout=timeout,
            headers={"User-Agent": "UniFi-Protect-Time-Lapse/2.0"},
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if hasattr(self, "client"):
            await self.client.aclose()

    async def refresh_cameras(self, force: bool = False) -> List[Camera]:
        """
        Refresh the list of cameras from the API.

        Args:
            force: Force refresh even if cache is still valid

        Returns:
            List of Camera objects
        """
        now = datetime.now()

        # Check if we need to refresh
        if (
            not force
            and self.last_camera_refresh
            and self.cameras
            and (now - self.last_camera_refresh).total_seconds()
            < self.camera_refresh_interval
        ):
            return self.cameras

        try:
            url = f"{config.UNIFI_PROTECT_BASE_URL}/cameras"

            response = await self.client.get(url, headers=config.get_json_headers())
            response.raise_for_status()
            cameras_data = response.json()

            # Convert API response to Camera objects
            all_cameras = [
                Camera.from_api_response(cam_data) for cam_data in cameras_data
            ]

            # Filter cameras based on configuration
            filtered_cameras = [
                camera
                for camera in all_cameras
                if config.should_process_camera(camera.name)
            ]

            self.cameras = filtered_cameras
            self.last_camera_refresh = now

            logging.info(
                f"Discovered {len(all_cameras)} total cameras, {len(filtered_cameras)} will be processed"
            )

            # Log all discovered camera names for debugging
            if all_cameras:
                camera_names = [f'"{camera.name}"' for camera in all_cameras]
                logging.info(f"Available cameras: {', '.join(camera_names)}")

            # Log camera details for cameras we'll process
            if filtered_cameras:
                for camera in filtered_cameras:
                    status = "✓" if camera.is_connected else "✗"
                    hd_support = "HD" if camera.supports_full_hd_snapshot else "SD"
                    logging.info(
                        f"  {status} {camera.name} ({camera.state}) - {camera.type} [{hd_support}]"
                    )

                # LOCK distribution settings on first discovery - NEVER change during runtime
                if not self._distribution_locked:
                    self._locked_total_cameras = len(filtered_cameras)
                    self._locked_use_distribution = (
                        config.should_use_camera_distribution(len(filtered_cameras))
                    )
                    self._locked_optimal_offset = (
                        config.calculate_optimal_offset_seconds(len(filtered_cameras))
                    )
                    self._distribution_locked = True

                    logging.info(
                        "🔒 LOCKING distribution settings based on discovered cameras:"
                    )
                    logging.info(
                        f"   Total cameras: {self._locked_total_cameras} (including disconnected)"
                    )
                    logging.info(
                        f"   Distribution enabled: {self._locked_use_distribution}"
                    )
                    if self._locked_use_distribution:
                        logging.info(
                            f"   Locked offset: {self._locked_optimal_offset}s"
                        )

                # Log camera distribution information using LOCKED settings
                if self._locked_use_distribution:
                    logging.info(
                        f"Camera distribution ENABLED: {self._locked_total_cameras} cameras, "
                        f"strategy: {config.FETCH_DISTRIBUTION_STRATEGY}, "
                        f"offset: {self._locked_optimal_offset}s (LOCKED)"
                    )

                    # Log rate limit analysis using locked settings
                    max_simultaneous_intervals = (
                        config.calculate_max_simultaneous_intervals()
                    )
                    effective_concurrent_limit = (
                        config.calculate_effective_concurrent_limit()
                    )

                    logging.info(
                        f"Rate limit analysis: "
                        f"limit={config.UNIFI_PROTECT_RATE_LIMIT} req/sec, "
                        f"effective={config.EFFECTIVE_RATE_LIMIT} req/sec, "
                        f"max_intervals={max_simultaneous_intervals}, "
                        f"concurrent_limit={effective_concurrent_limit} (LOCKED)"
                    )

                    if config.FETCH_LOG_SLOT_UTILIZATION:
                        self._log_camera_assignments(
                            filtered_cameras, self._locked_optimal_offset
                        )
                else:
                    logging.info(
                        f"Camera distribution DISABLED: {self._locked_total_cameras} cameras (LOCKED)"
                    )

                    # Check rate limit compliance without distribution using locked settings
                    config.validate_rate_limit_compliance(self._locked_total_cameras)

            else:
                logging.warning("No cameras match the current selection criteria")
                if config.CAMERA_SELECTION_MODE == "whitelist":
                    logging.warning(f"Whitelist: {config.CAMERA_WHITELIST}")
                elif config.CAMERA_SELECTION_MODE == "blacklist":
                    logging.warning(f"Blacklist: {config.CAMERA_BLACKLIST}")

            return self.cameras

        except httpx.RequestError as e:
            logging.error(f"Failed to fetch cameras: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error fetching cameras: {e}")
            raise

    def _log_camera_assignments(self, cameras: List[Camera], offset_seconds: int):
        """Log camera slot assignments for debugging."""
        connected_cameras = [cam for cam in cameras if cam.is_connected]

        if not connected_cameras:
            return

        # Group cameras by their assigned slots
        slot_assignments = defaultdict(list)
        for camera in connected_cameras:
            offset = camera.get_deterministic_offset(offset_seconds)
            slot = offset // offset_seconds
            slot_assignments[slot].append(camera.name)

        # Calculate max cameras per slot based on rate limits
        max_simultaneous_intervals = config.calculate_max_simultaneous_intervals()
        max_cameras_per_slot = config.EFFECTIVE_RATE_LIMIT // max_simultaneous_intervals

        logging.info("Camera slot assignments (deterministic):")
        for slot in sorted(slot_assignments.keys()):
            camera_names = slot_assignments[slot]
            offset = slot * offset_seconds
            logging.info(f"  Slot {slot} (+{offset}s): {', '.join(camera_names)}")

            # Warn if slot exceeds rate limit capacity
            if len(camera_names) > max_cameras_per_slot:
                logging.warning(
                    f"  ⚠️  Slot {slot} has {len(camera_names)} cameras "
                    f"(exceeds rate limit capacity of {max_cameras_per_slot})"
                )

    async def get_cameras(self, force_refresh: bool = False) -> List[Camera]:
        """
        Get the list of cameras, refreshing if necessary.

        Args:
            force_refresh: Force refresh from API

        Returns:
            List of Camera objects
        """
        if not self.cameras or force_refresh:
            await self.refresh_cameras(force=force_refresh)

        return self.cameras

    async def _get_connected_cameras(self) -> List[Camera]:
        """Return list of connected cameras to be used for capture.

        This consolidates the common pattern of fetching cameras, filtering
        out disconnected devices, and logging warnings so callers don't
        duplicate the same logic.
        """
        cameras = await self.get_cameras()

        if not cameras:
            logging.warning("No cameras available for capture")
            return []

        connected_cameras = [camera for camera in cameras if camera.is_connected]
        disconnected_cameras = [camera for camera in cameras if not camera.is_connected]

        if disconnected_cameras:
            disconnected_names = [cam.name for cam in disconnected_cameras]
            logging.warning(
                f"Skipping {len(disconnected_cameras)} disconnected cameras: {', '.join(disconnected_names)}"
            )

        if not connected_cameras:
            logging.warning("No connected cameras available for capture")
            return []

        return connected_cameras

    async def capture_snapshot(
        self, camera: Camera, output_path: str, interval: int, retry_count: int = 0
    ) -> bool:
        """
        Capture a snapshot from the specified camera.

        Args:
            camera: Camera object to capture from
            output_path: Path to save the image
            interval: Interval in seconds (for logging)
            retry_count: Current retry attempt (for internal use)

        Returns:
            True if successful, False otherwise
        """
        if not camera.is_connected:
            logging.debug(
                f"[{interval}s] Skipping {camera.name} - not connected (state: {camera.state})"
            )
            return False

        try:
            url = f"{config.UNIFI_PROTECT_BASE_URL}/cameras/{camera.id}/snapshot"

            # Build query parameters - only use highQuality if camera supports it
            params = {}
            if config.SNAPSHOT_HIGH_QUALITY and camera.supports_full_hd_snapshot:
                params["highQuality"] = "true"
                quality_note = "HQ"
            else:
                quality_note = "STD"

            # Log the request we're about to make
            logging.debug(f"[{interval}s] Requesting snapshot from {camera.name}")

            response = await self.client.get(
                url, headers=config.get_image_headers(), params=params
            )

            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "")

                if content_type.startswith("image/"):
                    # Ensure directory exists
                    import os

                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    # Write image data
                    with open(output_path, "wb") as f:
                        f.write(response.content)

                    # Verify file was written and has reasonable size
                    if (
                        os.path.exists(output_path)
                        and os.path.getsize(output_path) > 1000
                    ):
                        file_size = os.path.getsize(output_path)
                        logging.debug(
                            f"[{interval}s] ✓ Captured {camera.name} [{quality_note}] -> {output_path} ({file_size/1024:.1f}KB)"
                        )
                        return True
                    else:
                        logging.error(
                            f"[{interval}s] ✗ Image file too small or missing: {camera.name}"
                        )
                        return False
                else:
                    logging.error(
                        f"[{interval}s] ✗ Invalid content type for {camera.name}: {content_type}"
                    )
                    return False
            else:
                # Try to get error details
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", "Unknown error")
                except:
                    error_msg = f"HTTP {response.status_code}"

                logging.error(
                    f"[{interval}s] ✗ Snapshot failed for {camera.name}: {error_msg}"
                )
                return False

        except httpx.TimeoutException:
            logging.error(f"[{interval}s] ✗ Timeout capturing {camera.name}")
            return False
        except httpx.RequestError as e:
            logging.error(f"[{interval}s] ✗ Network error capturing {camera.name}: {e}")
            return False
        except Exception as e:
            logging.error(
                f"[{interval}s] ✗ Unexpected error capturing {camera.name}: {e}"
            )
            return False

    async def capture_snapshot_with_retry(
        self, camera: Camera, output_path: str, interval: int
    ) -> bool:
        """
        Capture a snapshot with retry logic.

        Args:
            camera: Camera object to capture from
            output_path: Path to save the image
            interval: Interval in seconds (for logging)

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(config.FETCH_MAX_RETRIES + 1):
            success = await self.capture_snapshot(
                camera, output_path, interval, attempt
            )

            if success:
                return True

            if attempt < config.FETCH_MAX_RETRIES:
                logging.debug(
                    f"[{interval}s] Retrying {camera.name} in {config.FETCH_RETRY_DELAY}s (attempt {attempt + 1}/{config.FETCH_MAX_RETRIES})"
                )
                await asyncio.sleep(config.FETCH_RETRY_DELAY)

        logging.error(
            f"[{interval}s] Failed to capture {camera.name} after {config.FETCH_MAX_RETRIES + 1} attempts"
        )
        return False

    async def capture_all_cameras(
        self, timestamp: int, interval: int
    ) -> Dict[str, Dict[str, bool]]:
        """
        Capture snapshots from all configured cameras concurrently.

        Args:
            timestamp: Unix timestamp for the capture
            interval: Interval in seconds (for directory structure)

        Returns:
            Dictionary mapping camera names to success status
        """
        connected_cameras = await self._get_connected_cameras()
        if not connected_cameras:
            return {}

        logging.debug(
            f"[{interval}s] Capturing from {len(connected_cameras)} connected cameras"
        )

        return await self._capture_from_set_of_cameras(connected_cameras, timestamp, interval)

    async def capture_cameras_distributed(
        self, timestamp: int, interval: int
    ) -> Dict[str, Dict[str, bool]]:
        """
        Capture snapshots from cameras using distributed timing to avoid rate limits.

        Args:
            timestamp: Base unix timestamp for the capture
            interval: Interval in seconds (for directory structure)

        Returns:
            Dictionary mapping camera names to a mapping of preset name -> success status
        """
        connected_cameras = await self._get_connected_cameras()
        if not connected_cameras:
            return {}

        # Calculate optimal offset based on LOCKED settings
        optimal_offset = (
            self._locked_optimal_offset
            if self._distribution_locked
            else config.calculate_optimal_offset_seconds(len(connected_cameras))
        )

        # Handle case where optimal offset is 0 (no distribution needed)
        if optimal_offset == 0:
            # No distribution - capture all cameras immediately
            logging.debug(
                f"[{interval}s] No distribution needed, capturing all cameras immediately"
            )
            return await self._capture_from_set_of_cameras(connected_cameras, timestamp, interval)

        # Group cameras by their offset (only if distribution is enabled)
        camera_groups: Dict[int, List[Camera]] = defaultdict(list)
        for camera in connected_cameras:
            offset = camera.get_deterministic_offset(optimal_offset)
            camera_groups[offset].append(camera)

        logging.debug(
            f"[{interval}s] Capturing {len(connected_cameras)} cameras in {len(camera_groups)} groups "
            f"with {optimal_offset}s offsets (strategy: {config.FETCH_DISTRIBUTION_STRATEGY})"
        )

        # Execute captures for each group with proper timing
        all_results = {}

        for offset, group_cameras in sorted(camera_groups.items()):
            logging.debug(
                f"[{interval}s] Capturing group at +{offset}s: {[cam.name for cam in group_cameras]}"
            )

            group_results = await self._capture_from_set_of_cameras(group_cameras, timestamp + offset, interval)
            for key, subdict in group_results.items():
                all_results.setdefault(key, {}).update(subdict) 

            # Wait before next group (if there are more groups)
            remaining_groups = len([o for o in camera_groups.keys() if o > offset])
            if remaining_groups > 0:
                await asyncio.sleep(optimal_offset)

        # Log summary
        successful = sum(1 for success in all_results.values() if success)
        total = len(all_results)
        logging.info(
            f"[{interval}s] Captured {successful}/{total} connected cameras "
            f"(distributed across {len(camera_groups)} groups, {optimal_offset}s offset)"
        )

        return all_results

    async def _capture_from_set_of_cameras(
        self,
        cameras: List[Camera], ts: int, ivl: int
    ) -> Dict[str, Dict[str, bool]]:
        # Create semaphore for all cameras
        concurrent_limit = config.calculate_effective_concurrent_limit()
        semaphore = asyncio.Semaphore(min(len(cameras), concurrent_limit))

        async def capture_camera(
            camera: Camera,
        ) -> tuple[str, Dict[str, bool]]:
            async with semaphore:
                results = await self._capture_presets_for_camera(
                    camera, ts, ivl
                )
                return camera.name, results

        # Execute all captures concurrently
        tasks = [capture_camera(camera) for camera in cameras]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results into camera -> {preset: success}
        all_results: Dict[str, Dict[str, bool]] = {}
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                camera_name, preset_results = result
                all_results[camera_name] = preset_results
            else:
                logging.error(f"Unexpected error in camera capture: {result}")

        # Log summary
        total_presets = sum(len(presets) for presets in all_results.values())
        successful_presets = sum(
            1 for presets in all_results.values() for v in presets.values() if v
        )
        logging.info(
            f"[{ivl}s] Captured {successful_presets}/{total_presets} presets across {len(all_results)} cameras"
        )

        return all_results

    async def _capture_one_preset(
        self,
        camera: Camera,
        preset_name: str | None,
        preset_number: str | int | None,
        capture_timestamp: int,
        interval: int,
    ) -> bool:
        """Helper to capture one preset for a camera.

        Handles optional PTZ move, directory creation, and snapshot capture.
        Returns True on success, False otherwise.
        """

        try:
            # Move to preset if defined
            if preset_number is not None:
                if not await self.goto_preset(camera, interval, preset_number, preset_name):
                    logging.error(f"[CAPTURE] {camera.name} {interval}s preset {preset_name} not attempted, since preset move failed.")
                    return False

            # Build output path and capture
            date_obj = datetime.fromtimestamp(capture_timestamp)
            year = date_obj.strftime("%Y")
            month = date_obj.strftime("%m")
            day = date_obj.strftime("%d")

            camera_dir_name = f"{camera.safe_name}-{preset_name}" if preset_name != "Default" else camera.safe_name
            output_dir = (
                config.IMAGE_OUTPUT_PATH
                / camera_dir_name
                / f"{interval}s"
                / year
                / month
                / day
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{camera.safe_name}_{capture_timestamp}.jpg"

            success = await self.capture_snapshot_with_retry(
                camera, str(output_path), interval
            )

            logging.info(f"[CAPTURE] {camera.name} {interval}s preset {preset_name}: {'Success' if success else 'Failed'}")
            return success
        except Exception as e:
            logging.error(f"[CAPTURE] {camera.name} {interval}s preset {preset_name} failed: {e}")
            return False

    async def _capture_presets_for_camera(
        self, camera: Camera, capture_timestamp: int, interval: int
    ) -> Dict[str, bool]:
        """Capture all presets for a camera serially.

        Returns a mapping preset_name -> success.
        Handles determining whether presets move the camera and returning to Home if configured.
        """
        results: Dict[str, bool] = {}
        presets: Dict[str, Any] = config.CAMERA_PRESETS.get(camera.name, {"Default": None})

        # Determine if any preset will move the camera
        moved_presets = any(
            (preset_number is not None and str(preset_number) != "-1")
            for preset_number in presets.values()
        )

        # Iterate presets in deterministic order, and do them serially...
        for preset_name in sorted(presets.keys()):
            preset_number = presets[preset_name]
            success = await self._capture_one_preset(
                camera, preset_name, preset_number, capture_timestamp, interval
            )
            results[preset_name] = success

        # Optionally return to Home after all presets
        if config.CAMERA_PTZ_RETURN_TO_HOME and moved_presets:
            # Allow time for any prior image captures to complete, then move to Home
            await asyncio.sleep(config.CAMERA_PTZ_PRESET_DELAY)
            returned = await self.goto_preset(camera, interval, -1, "Home")
            if not returned:
                logging.warning(f"[PTZ] Failed to return {camera.name} to Home preset (-1)")

        return results

    async def goto_preset(self, camera: Camera, interval: int, preset_number: str | int | None, preset_name: str | None = None) -> bool:
        """Move camera to a named preset via the UniFi Protect "goto" endpoint.
        Returns True on success (HTTP 200/204), False otherwise.
        If `preset_number` is None, this is a no-op and returns False.
        """
        if preset_number is None:
            return False

        # Normalize preset number to string
        preset_str = str(preset_number)

        url = f"{config.UNIFI_PROTECT_BASE_URL}/cameras/{camera.id}/ptz/goto/{preset_str}"
        headers = {
            "X-API-Key": config.UNIFI_PROTECT_API_KEY,
            "Accept": "application/json",
            "User-Agent": "UniFi-Protect-Time-Lapse/2.0",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    ssl=config.UNIFI_PROTECT_VERIFY_SSL,
                    timeout=int(config.UNIFI_PROTECT_REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status in (200, 204):
                        logging.info(
                            f"[PTZ] Moved {camera.name} at {interval}s to preset {preset_name} ({preset_str}), waiting {config.CAMERA_PTZ_PRESET_DELAY}s for capture"
                        )
                        # Allow camera time to reach the preset
                        await asyncio.sleep(config.CAMERA_PTZ_PRESET_DELAY)
                        return True
                    else:
                        logging.warning(
                            f"[PTZ] PTZ move to preset {preset_name} ({preset_str}) returned {resp.status} for {camera.name}"
                        )
                        return False

        except Exception as e:
            logging.warning(f"[PTZ] Error moving {camera.name} to preset {preset_name}: {e}")
            return False