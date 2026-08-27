"""
Readers for instrument-native powder XRD formats: PANalytical XRDML and
Bruker BRML/UXD. Ported from mholmboe/mlm-xrd (mlmnc/io), trimmed to what
peakipy needs: two-theta, intensity, and a metadata dict.
"""

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

XSI_TYPE = '{http://www.w3.org/2001/XMLSchema-instance}type'
WAVELENGTH_RANGE = (0.3, 3.0)


def read_xrdml(filepath):
    """
    Parse a PANalytical/Malvern Panalytical .xrdml file.

    Returns
    -------
    two_theta : ndarray
    intensity : ndarray
    metadata : dict
    """
    path = Path(filepath)
    tree = ET.parse(path)
    root = tree.getroot()

    namespace = _detect_xrdml_namespace(root)
    ns = {'xrdml': namespace}

    two_theta, intensity = _extract_xrdml_data(root, ns)
    if len(two_theta) == 0:
        raise ValueError(f"No measurement data found in XRDML file: {path.name}")

    metadata = _extract_xrdml_metadata(root, ns)
    metadata['source_format'] = 'xrdml'
    metadata['filename'] = path.name

    return np.asarray(two_theta, float), np.asarray(intensity, float), metadata


def _detect_xrdml_namespace(root):
    tag = root.tag
    if '}' in tag:
        return tag.split('}')[0][1:]
    return 'http://www.xrdml.com/XRDMeasurement/1.6'


def _extract_xrdml_data(root, ns):
    datapoints = root.find('.//xrdml:dataPoints', ns)
    if datapoints is None:
        datapoints = root.find('.//dataPoints')
    if datapoints is None:
        raise ValueError("No dataPoints found in XRDML file")

    # Preferred: explicit position list.
    positions_elem = datapoints.find('xrdml:positions[@axis="2Theta"]', ns)
    if positions_elem is None:
        positions_elem = datapoints.find('xrdml:positions', ns)
    if positions_elem is None:
        positions_elem = datapoints.find('positions')

    intensities_elem = datapoints.find('xrdml:intensities', ns)
    if intensities_elem is None:
        intensities_elem = datapoints.find('intensities')
    if intensities_elem is None or not intensities_elem.text:
        raise ValueError("No intensities found in dataPoints")
    intensity_data = _parse_values(intensities_elem.text)
    n_points = len(intensity_data)

    two_theta_data = []
    if positions_elem is not None:
        list_text = positions_elem.findtext('xrdml:listPositions', default=None, namespaces=ns)
        start_elem = positions_elem.find('xrdml:startPosition', ns)
        end_elem = positions_elem.find('xrdml:endPosition', ns)
        if start_elem is None:
            start_elem = positions_elem.find('startPosition')
        if end_elem is None:
            end_elem = positions_elem.find('endPosition')

        if list_text:
            two_theta_data = _parse_values(list_text)
        elif start_elem is not None and end_elem is not None:
            start_pos = float(start_elem.text)
            end_pos = float(end_elem.text)
            two_theta_data = list(np.linspace(start_pos, end_pos, n_points))

    if not two_theta_data:
        raise ValueError("Could not resolve 2Theta axis in XRDML file")

    return two_theta_data, intensity_data


def _parse_values(text):
    values = []
    for token in text.strip().split():
        if token.lower() == 'nan':
            values.append(0.0)
        else:
            try:
                values.append(float(token))
            except ValueError:
                values.append(0.0)
    return values


def _extract_xrdml_metadata(root, ns):
    metadata = {}

    wavelength_elem = root.find('.//xrdml:usedWavelength/xrdml:kAlpha1', ns)
    if wavelength_elem is not None and wavelength_elem.text:
        metadata['wavelength'] = float(wavelength_elem.text)

    sample_elem = root.find('.//xrdml:sample', ns)
    if sample_elem is not None:
        name_elem = sample_elem.find('.//xrdml:name', ns)
        if name_elem is not None and name_elem.text:
            metadata['sample_name'] = name_elem.text

    tube_elem = root.find('.//xrdml:xRayTube', ns)
    if tube_elem is not None:
        anode_elem = tube_elem.find('.//xrdml:anodeMaterial', ns)
        if anode_elem is not None and anode_elem.text:
            metadata['anode'] = anode_elem.text

    return metadata


def read_uxd(filepath):
    """
    Parse a Bruker .uxd (text) file.

    Returns
    -------
    two_theta : ndarray
    intensity : ndarray
    metadata : dict
    """
    path = Path(filepath)
    with open(path, 'r', errors='replace') as f:
        lines = f.readlines()

    data_start = None
    metadata = {'source_format': 'uxd'}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('_2THETA_INTENSITY') or stripped.startswith('_2THETACOUNTS'):
            data_start = i + 1
            break
        elif stripped.startswith('_') and ' ' in stripped:
            key, value = stripped.split(' ', 1)
            metadata[key.lstrip('_')] = value.strip()

    if data_start is None:
        raise ValueError(f"No data section found in UXD file: {path.name}")

    two_theta_list, intensity_list = [], []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith('_'):
            continue
        parts = re.split(r'[\s,;]+', stripped)
        if len(parts) < 2:
            continue
        try:
            two_theta_list.append(float(parts[0]))
            intensity_list.append(float(parts[1]))
        except ValueError:
            continue

    if not two_theta_list:
        raise ValueError(f"No numeric two-theta/intensity rows found in UXD file: {path.name}")

    return np.asarray(two_theta_list, float), np.asarray(intensity_list, float), metadata


def read_brml(filepath):
    """
    Parse a Bruker .brml file (a zip archive of XML documents).

    Returns
    -------
    two_theta : ndarray
    intensity : ndarray (counts)
    metadata : dict
    """
    path = Path(filepath)
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path.name} is not a valid .brml (zip) archive: {exc}") from exc

    with z:
        names = z.namelist()
        raw_names = _brml_raw_data_names(z, names)
        if not raw_names:
            raise ValueError(f"{path.name} contains no RawData*.xml entries")

        metadata = {'source_format': 'brml', 'filename': path.name}
        metadata.update(_brml_instrument_info(z, names))

        for name in raw_names:
            root = ET.fromstring(z.read(name))
            for route in root.findall('./DataRoutes/DataRoute'):
                info = route.find('ScanInformation')
                scan_name = (info.get('ScanName') or '') if info is not None else ''
                if 'NonAmbientModeData' in scan_name:
                    continue

                rows = _brml_rows(route)
                if len(rows) < 2:
                    continue

                idx = _brml_datum_indices(route)
                tt_i, in_i = _brml_pick_columns(rows, idx, route)
                en_i = idx.get('enabled')
                keep = [r for r in rows if en_i is None or en_i >= len(r) or r[en_i] != 0]
                if len(keep) < 2:
                    continue

                two_theta = np.array([r[tt_i] for r in keep], dtype=float)
                intensity = np.array([r[in_i] for r in keep], dtype=float)
                metadata['intensity_unit'] = 'counts'
                if scan_name:
                    metadata['scan_name'] = scan_name

                order = np.argsort(two_theta)
                return two_theta[order], intensity[order], metadata

    raise ValueError(f"{path.name}: found XML entries but no scan with usable data")


def _brml_raw_data_names(z, names):
    listed = []
    for n in names:
        if not n.endswith('DataContainer.xml'):
            continue
        folder = n.rsplit('/', 1)[0] if '/' in n else ''
        try:
            root = ET.fromstring(z.read(n))
        except ET.ParseError:
            continue
        for child in root.findall('./RawDataReferenceList/'):
            ref = (child.text or '').strip()
            if not ref:
                continue
            for cand in (ref, f'{folder}/{ref}' if folder else ref):
                if cand in names and cand not in listed:
                    listed.append(cand)
    if listed:
        return listed
    return sorted(n for n in names if re.search(r'RawData\d*\.xml$', n))


def _brml_instrument_info(z, names):
    out = {}
    for n in names:
        if not n.endswith('MeasurementContainer.xml'):
            continue
        head = z.read(n).decode('utf-8', 'replace')
        m = re.search(r'<WaveLengthAlpha1[^>]*Value="([^"]+)"', head)
        if m:
            try:
                val = float(m.group(1))
                if WAVELENGTH_RANGE[0] <= val <= WAVELENGTH_RANGE[1]:
                    out['wavelength'] = val
            except ValueError:
                pass
        m = re.search(r'<TubeMaterial[^>]*Value="([^"]+)"', head)
        if m:
            out['anode'] = m.group(1)
        break
    return out


def _brml_datum_indices(route):
    idx = {}
    for view in route.findall('./DataViews/RawDataView'):
        start = int(view.get('Start', 0))
        logic = view.get('LogicName') or ''
        kind = view.get(XSI_TYPE) or ''
        if logic == 'MeasuredTime':
            idx['time'] = start
        elif logic == 'AbsorptionFactor':
            idx['enabled'] = start
        elif kind == 'VaryingRawDataView':
            for i, field in enumerate(view.findall('./Varying/FieldDefinitions')):
                name = field.get('FieldName') or field.get('AxisId') or ''
                if name == 'TwoTheta':
                    idx['two_theta'] = start + i
                elif name == 'Theta':
                    idx['theta'] = start + i
        elif kind == 'RecordedRawDataView':
            rec = view.find('Recording')
            if rec is not None and rec.get('LogicName') == 'ScanCounter':
                idx['intensity'] = start
    return idx


def _brml_axis_range(route):
    for field in route.findall('.//Varying/FieldDefinitions[@FieldName="TwoTheta"]/Restriction'):
        lo = field.find('Minimum')
        hi = field.find('Maximum')
        try:
            return (float(lo.text) if lo is not None and lo.text else None,
                    float(hi.text) if hi is not None and hi.text else None)
        except ValueError:
            return None, None
    return None, None


def _brml_rows(route):
    out = []
    for datum in route.findall('Datum'):
        if not datum.text:
            continue
        try:
            out.append([float(x) for x in datum.text.split(',')])
        except ValueError:
            continue
    return out


def _brml_pick_columns(rows, idx, route):
    width = min(len(r) for r in rows)
    tt_i, in_i = idx.get('two_theta'), idx.get('intensity')
    if tt_i is None or tt_i >= width:
        lo, hi = _brml_axis_range(route)
        tt_i = None
        for c in range(width):
            col = np.array([r[c] for r in rows])
            if col.size > 1 and np.all(np.diff(col) > 0):
                if lo is None or hi is None or (lo - 1 <= col[0] and col[-1] <= hi + 1):
                    tt_i = c
                    break
        if tt_i is None:
            raise ValueError("BRML: no DataViews entry for TwoTheta and no "
                              "column that looks like an increasing angle axis")
    if in_i is None or in_i >= width:
        in_i = width - 1
    return tt_i, in_i


#: extension (lowercase, with dot) -> reader(filepath) -> (x, y, metadata)
XRD_READERS = {
    '.xrdml': read_xrdml,
    '.uxd': read_uxd,
    '.brml': read_brml,
}


def is_xrd_format(filepath):
    """True if filepath has a known instrument-native XRD extension."""
    return Path(filepath).suffix.lower() in XRD_READERS


def read_xrd_file(filepath):
    """
    Dispatch to the right reader based on file extension.

    Returns
    -------
    x : ndarray
    y : ndarray
    metadata : dict
    """
    suffix = Path(filepath).suffix.lower()
    reader = XRD_READERS.get(suffix)
    if reader is None:
        raise ValueError(f"Unsupported XRD format: {suffix}")
    return reader(filepath)
