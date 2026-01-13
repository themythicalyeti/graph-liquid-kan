"""
BarentsWatch Fish Health API Client

Phase 1.1: Biological Data Acquisition for Graph-Liquid-KAN

Retrieves the Graph Nodes (localities) with:
- Site locations with coordinates (LocalityID, lat, lon)
- Lice abundance data (AvgAdultFemaleLice, AvgMobileLice, AvgStationaryLice)
- Treatment history with event classification

Protocol: OpenID Connect with client_credentials grant
API Docs: https://www.barentswatch.no/bwapi/

The output establishes the "Graphon-Static Field" - only farms with valid
coordinates are retained, ensuring the adjacency matrix is mathematically
well-defined for the Graph Neural ODE.
"""

import os
import csv
import time
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

import requests
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config/.env.local")
load_dotenv("config/.env")
load_dotenv()


@dataclass
class Site:
    """
    Aquaculture site (Graph Node) with coordinates.

    These form the vertices of the biological graph. Only sites with
    valid coordinates are retained for the Graph Neural ODE.
    """
    locality_id: int
    name: str
    municipality: str
    county: str
    latitude: float
    longitude: float
    production_area: Optional[int] = None
    is_active: bool = True


@dataclass
class LiceReport:
    """
    Parsed lice report record (Y target variable).

    Contains the target variables for prediction:
    - avg_adult_female_lice: Primary target (regulatory threshold 0.5)
    - avg_mobile_lice: Secondary predictor
    - avg_stationary_lice: Early indicator
    """
    locality_id: int
    year: int
    week: int
    report_date: Optional[datetime]
    avg_adult_female_lice: Optional[float]
    avg_mobile_lice: Optional[float]
    avg_stationary_lice: Optional[float]
    has_reported_lice: bool
    is_mnar: bool  # Missing Not At Random flag
    sea_temperature: Optional[float]
    latitude: Optional[float]
    longitude: Optional[float]
    name: str = ""


@dataclass
class TreatmentEvent:
    """
    Parsed treatment event record (intervention variable).

    Treatment events cause discontinuities in lice dynamics that the
    model must handle. Classified into three categories for the
    treatment feature vector [is_mechanical, is_medicinal, is_cleaner_fish].
    """
    locality_id: int
    treatment_date: Optional[datetime]
    year: int
    week: int
    treatment_type: str
    is_mechanical: bool = False
    is_medicinal: bool = False
    is_cleaner_fish: bool = False


class BarentsWatchClient:
    """
    Client for BarentsWatch Fish Health API.

    Implements OAuth2 client_credentials flow and provides methods
    for fetching locality data, lice reports, and treatment history.

    Token Lifetime Handling:
    The client automatically refreshes tokens before expiration to prevent
    "Silent Data Gaps" in long-running SciML pipelines, which would cause
    ODE solver divergence in later phases.
    """

    AUTH_URL = "https://id.barentswatch.no/connect/token"
    BASE_URL = "https://www.barentswatch.no/bwapi"

    # Treatment type classification mapping
    TREATMENT_TYPES = {
        # Mechanical treatments
        "Hydrolicer": "mechanical",
        "Thermolicer": "mechanical",
        "Optilicer": "mechanical",
        "FLS": "mechanical",
        "Freshwater": "mechanical",
        "Ferskvannsbehandling": "mechanical",
        # Medicinal treatments
        "Azametifos": "medicinal",
        "Deltametrin": "medicinal",
        "Hydrogenperoksid": "medicinal",
        "Emamektin": "medicinal",
        "Slice": "medicinal",
        "Alphamax": "medicinal",
        # Cleaner fish
        "Rensefisk": "cleaner_fish",
        "Leppefisk": "cleaner_fish",
        "Rognkjeks": "cleaner_fish",
    }

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 60.0,
        rate_limit_delay: float = 1.0,
    ):
        """
        Initialize the BarentsWatch client.

        Args:
            client_id: OAuth2 client ID (or from BARENTSWATCH_CLIENT_ID env)
            client_secret: OAuth2 client secret (or from BARENTSWATCH_CLIENT_SECRET env)
            retry_attempts: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
            rate_limit_delay: Delay between API calls to avoid rate limiting
        """
        self.client_id = client_id or os.getenv("BARENTSWATCH_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("BARENTSWATCH_CLIENT_SECRET")

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "BarentsWatch credentials required. Set BARENTSWATCH_CLIENT_ID "
                "and BARENTSWATCH_CLIENT_SECRET environment variables."
            )

        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay

        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None

        self.session = requests.Session()

    def _authenticate(self) -> None:
        """Obtain OAuth2 access token using client_credentials grant."""
        logger.info("Authenticating with BarentsWatch...")

        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": "api",
        }

        response = requests.post(self.AUTH_URL, data=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        self._access_token = data["access_token"]
        expires_in = data.get("expires_in", 3600)
        # Refresh 60 seconds before expiration to prevent silent gaps
        self._token_expires = datetime.now() + timedelta(seconds=expires_in - 60)

        logger.info(f"Authentication successful. Token expires in {expires_in}s")

    def _ensure_authenticated(self) -> None:
        """Ensure we have a valid access token."""
        if self._access_token is None or datetime.now() >= self._token_expires:
            self._authenticate()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with Bearer token."""
        self._ensure_authenticated()
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }

    def _request_with_retry(
        self, method: str, url: str, **kwargs
    ) -> requests.Response:
        """Make HTTP request with automatic retry on failure."""
        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                time.sleep(self.rate_limit_delay)

                response = self.session.request(
                    method, url, headers=self._get_headers(), timeout=120, **kwargs
                )

                if response.status_code == 429:  # Too Many Requests
                    wait_time = int(response.headers.get("Retry-After", self.retry_delay))
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if response.status_code >= 500:
                    logger.warning(
                        f"Server error {response.status_code}. "
                        f"Attempt {attempt + 1}/{self.retry_attempts}"
                    )
                    time.sleep(self.retry_delay)
                    continue

                response.raise_for_status()
                return response

            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"Timeout. Attempt {attempt + 1}/{self.retry_attempts}")
                time.sleep(self.retry_delay)

            except requests.exceptions.ConnectionError as e:
                last_exception = e
                logger.warning(f"Connection error. Attempt {attempt + 1}/{self.retry_attempts}")
                time.sleep(self.retry_delay)

        raise requests.RequestException(
            f"All {self.retry_attempts} attempts failed"
        ) from last_exception

    # =========================================================================
    # Site/Locality Methods (Graph Nodes)
    # =========================================================================

    def get_localities_with_coordinates(
        self, year: Optional[int] = None, week: Optional[int] = None
    ) -> List[Site]:
        """
        Fetch all localities with coordinates from weekly data endpoint.

        This establishes the Graph Nodes for the Graph-Liquid-KAN.
        Only farms with valid coordinates are retained to ensure the
        adjacency matrix is mathematically well-defined.

        Args:
            year: Year to query (defaults to current year)
            week: Week number (defaults to current week)

        Returns:
            List of Site objects with coordinates
        """
        if year is None:
            year = datetime.now().year
        if week is None:
            week = datetime.now().isocalendar()[1]

        logger.info(f"Fetching localities with coordinates for {year} week {week}...")

        url = f"{self.BASE_URL}/v1/geodata/fishhealth/locality/{year}/{week}"
        response = self._request_with_retry("GET", url)
        data = response.json()

        if isinstance(data, dict) and "localities" in data:
            localities = data["localities"]
        else:
            localities = data

        sites = []
        invalid_count = 0

        for loc in localities:
            lat = loc.get("lat") or loc.get("latitude")
            lon = loc.get("lon") or loc.get("longitude")

            if lat is not None and lon is not None:
                try:
                    site = Site(
                        locality_id=loc.get("localityNo") or loc.get("id"),
                        name=loc.get("name", "Unknown"),
                        municipality=loc.get("municipality", ""),
                        county=loc.get("county", ""),
                        latitude=float(lat),
                        longitude=float(lon),
                        production_area=loc.get("productionArea"),
                        is_active=loc.get("hasReportedLice", True),
                    )
                    sites.append(site)
                except (ValueError, TypeError):
                    invalid_count += 1
            else:
                invalid_count += 1

        logger.info(f"Retrieved {len(sites)} localities with valid coordinates")
        if invalid_count > 0:
            logger.warning(f"Excluded {invalid_count} localities with missing/invalid coordinates")

        return sites

    # =========================================================================
    # Lice Data Methods (Target Variable Y)
    # =========================================================================

    def download_lice_data_csv(
        self,
        from_year: int,
        from_week: int,
        to_year: int,
        to_week: int,
    ) -> List[LiceReport]:
        """
        Download bulk lice data using the CSV export endpoint.

        This is the most efficient method for fetching historical data.
        The MNAR (Missing Not At Random) flag is critical - farms that don't
        report often have high lice counts and are systematically different.

        Args:
            from_year: Start year
            from_week: Start week number
            to_year: End year
            to_week: End week number

        Returns:
            List of LiceReport objects with MNAR flags
        """
        logger.info(
            f"Downloading lice data from {from_year} W{from_week} "
            f"to {to_year} W{to_week}..."
        )

        url = f"{self.BASE_URL}/v1/geodata/download/fishhealth"
        params = {
            "reporttype": "lice",
            "filetype": "csv",
            "fromyear": from_year,
            "fromweek": from_week,
            "toyear": to_year,
            "toweek": to_week,
        }

        response = self._request_with_retry("GET", url, params=params)
        content = response.content.decode("utf-8-sig")
        reader = csv.DictReader(StringIO(content))

        reports = []
        for row in reader:
            # Parse year (handle BOM variants in Norwegian CSV)
            year_val = (
                row.get("\ufeffÅr") or row.get("År") or
                row.get("\ufeff\u00c5r") or row.get("\u00c5r", "0")
            )
            year = int(year_val) if year_val else 0
            week = int(row.get("Uke", 0))

            # Calculate report date (Wednesday of the week as per protocol)
            if year and week:
                jan1 = datetime(year, 1, 1)
                report_date = jan1 + timedelta(weeks=week - 1, days=2)
            else:
                report_date = None

            # MNAR flag: HasReportedLice=False means Missing Not At Random
            has_reported = row.get("Har telt lakselus", "").lower() == "ja"

            report = LiceReport(
                locality_id=int(row.get("Lokalitetsnummer", 0)),
                year=year,
                week=week,
                report_date=report_date,
                avg_adult_female_lice=self._parse_float(row.get("Voksne hunnlus")),
                avg_mobile_lice=self._parse_float(row.get("Lus i bevegelige stadier")),
                avg_stationary_lice=self._parse_float(row.get("Fastsittende lus")),
                has_reported_lice=has_reported,
                is_mnar=not has_reported,
                sea_temperature=self._parse_float(
                    row.get("Sjøtemperatur") or row.get("Sj\u00f8temperatur")
                ),
                latitude=self._parse_float(row.get("Lat")),
                longitude=self._parse_float(row.get("Lon")),
                name=row.get("Lokalitetsnavn", ""),
            )
            reports.append(report)

        logger.info(f"Downloaded {len(reports)} lice records")

        # Count MNAR for data quality reporting
        mnar_count = sum(1 for r in reports if r.is_mnar)
        if reports:
            logger.info(f"MNAR records: {mnar_count} ({100*mnar_count/len(reports):.1f}%)")

        return reports

    # =========================================================================
    # Treatment Data Methods (Intervention Variable)
    # =========================================================================

    def get_treatments(
        self, from_year: int, to_year: int
    ) -> List[TreatmentEvent]:
        """
        Fetch treatment history for all localities.

        Treatment events are classified into:
        - Mechanical (Hydrolicer, Thermolicer, etc.)
        - Medicinal (Azametifos, Deltametrin, etc.)
        - Cleaner fish (Rensefisk, Leppefisk, etc.)

        These create the binary feature vector [is_mechanical, is_medicinal, is_cleaner_fish]
        that captures intervention dynamics.

        Args:
            from_year: Start year
            to_year: End year

        Returns:
            List of TreatmentEvent objects with binary classification
        """
        logger.info(f"Fetching treatment data from {from_year} to {to_year}...")

        url = f"{self.BASE_URL}/v1/geodata/download/fishhealth"
        params = {
            "reporttype": "treatments",
            "filetype": "csv",
            "fromyear": from_year,
            "fromweek": 1,
            "toyear": to_year,
            "toweek": 52,
        }

        response = self._request_with_retry("GET", url, params=params)
        content = response.content.decode("utf-8-sig")
        reader = csv.DictReader(StringIO(content))

        events = []
        for row in reader:
            year_val = (
                row.get("\ufeffÅr") or row.get("År") or
                row.get("\ufeff\u00c5r") or row.get("\u00c5r", "0")
            )
            year = int(year_val) if year_val else 0
            week = int(row.get("Uke", 0))

            if year and week:
                jan1 = datetime(year, 1, 1)
                treatment_date = jan1 + timedelta(weeks=week - 1, days=2)
            else:
                treatment_date = None

            treatment_type = row.get("Behandlingstype", "Unknown")
            classification = self._classify_treatment(treatment_type)

            event = TreatmentEvent(
                locality_id=int(row.get("Lokalitetsnummer", 0)),
                treatment_date=treatment_date,
                year=year,
                week=week,
                treatment_type=treatment_type,
                is_mechanical=classification == "mechanical",
                is_medicinal=classification == "medicinal",
                is_cleaner_fish=classification == "cleaner_fish",
            )
            events.append(event)

        logger.info(f"Downloaded {len(events)} treatment events")

        # Summary breakdown
        mechanical = sum(1 for e in events if e.is_mechanical)
        medicinal = sum(1 for e in events if e.is_medicinal)
        cleaner = sum(1 for e in events if e.is_cleaner_fish)
        logger.info(
            f"Treatment breakdown: {mechanical} mechanical, "
            f"{medicinal} medicinal, {cleaner} cleaner fish"
        )

        return events

    def _classify_treatment(self, treatment_type: str) -> str:
        """Classify treatment type into mechanical/medicinal/cleaner_fish."""
        for keyword, classification in self.TREATMENT_TYPES.items():
            if keyword.lower() in treatment_type.lower():
                return classification
        return "unknown"

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _parse_float(self, value: Optional[str]) -> Optional[float]:
        """Parse float from string, handling Norwegian decimal format."""
        if not value or value.strip() == "":
            return None
        try:
            return float(value.replace(",", "."))
        except (ValueError, TypeError):
            return None

    # =========================================================================
    # DataFrame Conversion Methods
    # =========================================================================

    def sites_to_dataframe(self, sites: List[Site]) -> pd.DataFrame:
        """Convert list of Site objects to DataFrame (graph_nodes_metadata.csv)."""
        records = []
        for s in sites:
            records.append({
                "locality_id": s.locality_id,
                "name": s.name,
                "municipality": s.municipality,
                "county": s.county,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "production_area": s.production_area,
                "is_active": s.is_active,
            })
        return pd.DataFrame(records)

    def lice_reports_to_dataframe(self, reports: List[LiceReport]) -> pd.DataFrame:
        """Convert list of LiceReport objects to DataFrame."""
        records = []
        for r in reports:
            records.append({
                "locality_id": r.locality_id,
                "year": r.year,
                "week": r.week,
                "report_date": r.report_date,
                "avg_adult_female_lice": r.avg_adult_female_lice if not r.is_mnar else None,
                "avg_mobile_lice": r.avg_mobile_lice if not r.is_mnar else None,
                "avg_stationary_lice": r.avg_stationary_lice if not r.is_mnar else None,
                "has_reported_lice": r.has_reported_lice,
                "is_mnar": r.is_mnar,
                "sea_temperature": r.sea_temperature,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "name": r.name,
            })
        return pd.DataFrame(records)

    def treatments_to_dataframe(self, events: List[TreatmentEvent]) -> pd.DataFrame:
        """Convert list of TreatmentEvent objects to DataFrame."""
        records = []
        for e in events:
            records.append({
                "locality_id": e.locality_id,
                "treatment_date": e.treatment_date,
                "year": e.year,
                "week": e.week,
                "treatment_type": e.treatment_type,
                "is_mechanical": int(e.is_mechanical),
                "is_medicinal": int(e.is_medicinal),
                "is_cleaner_fish": int(e.is_cleaner_fish),
            })
        return pd.DataFrame(records)
