#!/usr/bin/env python3
"""
Convert MassSpecGym parquet files to mzML format.

This script reads parquet files containing MS/MS spectra data and converts them
to the standard mzML format for mass spectrometry data.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MzMLWriter:
    """Writer class for creating mzML files from spectrum data."""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.spectra = []
        
    def parse_array_string(self, array_str: str) -> np.ndarray:
        """Parse comma-separated string into numpy array."""
        if pd.isna(array_str) or array_str == '':
            return np.array([])
        return np.array([float(x) for x in str(array_str).split(',')])
    
    def add_spectrum(self, spec_data: Dict):
        """Add a spectrum to the writer."""
        self.spectra.append(spec_data)
    
    def write(self):
        """Write all spectra to mzML file."""
        # Create root element
        root = ET.Element('mzML', {
            'xmlns': 'http://psi.hupo.org/ms/mzml',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:schemaLocation': 'http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd',
            'version': '1.1.0'
        })
        
        # Add cvList
        cv_list = ET.SubElement(root, 'cvList', {'count': '2'})
        ET.SubElement(cv_list, 'cv', {
            'id': 'MS',
            'fullName': 'Proteomics Standards Initiative Mass Spectrometry Ontology',
            'version': '4.1.0',
            'URI': 'https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo'
        })
        ET.SubElement(cv_list, 'cv', {
            'id': 'UO',
            'fullName': 'Unit Ontology',
            'version': '09:04:2014',
            'URI': 'http://obo.cvs.sourceforge.net/*checkout*/obo/obo/ontology/phenotype/unit.obo'
        })
        
        # Add fileDescription
        file_desc = ET.SubElement(root, 'fileDescription')
        file_content = ET.SubElement(file_desc, 'fileContent')
        ET.SubElement(file_content, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000579',
            'name': 'MS1 spectrum',
            'value': ''
        })
        ET.SubElement(file_content, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000580',
            'name': 'MSn spectrum',
            'value': ''
        })
        
        source_file_list = ET.SubElement(file_desc, 'sourceFileList', {'count': '1'})
        source_file = ET.SubElement(source_file_list, 'sourceFile', {
            'id': 'SF1',
            'name': Path(self.output_path).name,
            'location': str(Path(self.output_path).parent)
        })
        ET.SubElement(source_file, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000569',
            'name': 'SHA-1',
            'value': ''
        })
        ET.SubElement(source_file, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000777',
            'name': 'spectrum identifier nativeID format',
            'value': ''
        })
        
        # Add softwareList
        software_list = ET.SubElement(root, 'softwareList', {'count': '1'})
        software = ET.SubElement(software_list, 'software', {
            'id': 'parquet_to_mzml_converter',
            'version': '1.0.0'
        })
        ET.SubElement(software, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000799',
            'name': 'custom unreleased software tool',
            'value': ''
        })
        
        # Add instrumentConfigurationList
        instrument_list = ET.SubElement(root, 'instrumentConfigurationList', {'count': '1'})
        instrument = ET.SubElement(instrument_list, 'instrumentConfiguration', {'id': 'IC1'})
        ET.SubElement(instrument, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000031',
            'name': 'instrument model',
            'value': ''
        })
        
        # Add dataProcessingList
        data_proc_list = ET.SubElement(root, 'dataProcessingList', {'count': '1'})
        data_proc = ET.SubElement(data_proc_list, 'dataProcessing', {'id': 'DP1'})
        proc_method = ET.SubElement(data_proc, 'processingMethod', {
            'order': '1',
            'softwareRef': 'parquet_to_mzml_converter'
        })
        ET.SubElement(proc_method, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000544',
            'name': 'Conversion to mzML',
            'value': ''
        })
        
        # Add run
        run = ET.SubElement(root, 'run', {
            'id': 'run1',
            'defaultInstrumentConfigurationRef': 'IC1',
            'startTimeStamp': datetime.now().isoformat()
        })
        
        # Add spectrumList
        spectrum_list = ET.SubElement(run, 'spectrumList', {
            'count': str(len(self.spectra)),
            'defaultDataProcessingRef': 'DP1'
        })
        
        # Add each spectrum
        for idx, spec in enumerate(self.spectra):
            self._add_spectrum_element(spectrum_list, idx, spec)
        
        # Write to file with pretty printing
        xml_str = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent='  ')
        
        with open(self.output_path, 'w') as f:
            f.write(pretty_xml)
        
        logger.info(f"Wrote {len(self.spectra)} spectra to {self.output_path}")
    
    def _add_spectrum_element(self, parent: ET.Element, index: int, spec: Dict):
        """Add a single spectrum element."""
        mzs = spec['mzs']
        intensities = spec['intensities']
        
        spectrum = ET.SubElement(parent, 'spectrum', {
            'id': f"scan={index + 1}",
            'index': str(index),
            'defaultArrayLength': str(len(mzs))
        })
        
        # Add spectrum metadata
        ET.SubElement(spectrum, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000580',
            'name': 'MSn spectrum',
            'value': ''
        })
        ET.SubElement(spectrum, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000511',
            'name': 'ms level',
            'value': '2'
        })
        ET.SubElement(spectrum, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000128',
            'name': 'profile spectrum',
            'value': ''
        })
        ET.SubElement(spectrum, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000130',
            'name': 'positive scan',
            'value': ''
        })
        
        # Add scan list
        scan_list = ET.SubElement(spectrum, 'scanList', {'count': '1'})
        ET.SubElement(scan_list, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000795',
            'name': 'no combination',
            'value': ''
        })
        
        scan = ET.SubElement(scan_list, 'scan')
        
        # Add instrument type if available
        if spec.get('instrument_type'):
            ET.SubElement(scan, 'cvParam', {
                'cvRef': 'MS',
                'accession': 'MS:1000044',
                'name': 'dissociation method',
                'value': spec['instrument_type']
            })
        
        # Add collision energy if available
        if spec.get('collision_energy') and not pd.isna(spec['collision_energy']):
            ET.SubElement(scan, 'cvParam', {
                'cvRef': 'MS',
                'accession': 'MS:1000045',
                'name': 'collision energy',
                'value': str(spec['collision_energy']),
                'unitCvRef': 'UO',
                'unitAccession': 'UO:0000266',
                'unitName': 'electronvolt'
            })
        
        # Add precursor list
        precursor_list = ET.SubElement(spectrum, 'precursorList', {'count': '1'})
        precursor = ET.SubElement(precursor_list, 'precursor')
        selected_ion_list = ET.SubElement(precursor, 'selectedIonList', {'count': '1'})
        selected_ion = ET.SubElement(selected_ion_list, 'selectedIon')
        
        # Add precursor m/z
        if spec.get('precursor_mz') and not pd.isna(spec['precursor_mz']):
            ET.SubElement(selected_ion, 'cvParam', {
                'cvRef': 'MS',
                'accession': 'MS:1000744',
                'name': 'selected ion m/z',
                'value': str(spec['precursor_mz']),
                'unitCvRef': 'MS',
                'unitAccession': 'MS:1000040',
                'unitName': 'm/z'
            })
        
        # Add activation
        activation = ET.SubElement(precursor, 'activation')
        ET.SubElement(activation, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000133',
            'name': 'collision-induced dissociation',
            'value': ''
        })
        
        # Add binary data arrays
        binary_data_array_list = ET.SubElement(spectrum, 'binaryDataArrayList', {'count': '2'})
        
        # m/z array
        mz_array = ET.SubElement(binary_data_array_list, 'binaryDataArray', {
            'encodedLength': str(len(mzs) * 8)
        })
        ET.SubElement(mz_array, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000514',
            'name': 'm/z array',
            'value': '',
            'unitCvRef': 'MS',
            'unitAccession': 'MS:1000040',
            'unitName': 'm/z'
        })
        ET.SubElement(mz_array, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000523',
            'name': '64-bit float',
            'value': ''
        })
        ET.SubElement(mz_array, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000576',
            'name': 'no compression',
            'value': ''
        })
        mz_binary = ET.SubElement(mz_array, 'binary')
        mz_binary.text = self._encode_array(mzs)
        
        # intensity array
        int_array = ET.SubElement(binary_data_array_list, 'binaryDataArray', {
            'encodedLength': str(len(intensities) * 8)
        })
        ET.SubElement(int_array, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000515',
            'name': 'intensity array',
            'value': '',
            'unitCvRef': 'MS',
            'unitAccession': 'MS:1000131',
            'unitName': 'number of detector counts'
        })
        ET.SubElement(int_array, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000523',
            'name': '64-bit float',
            'value': ''
        })
        ET.SubElement(int_array, 'cvParam', {
            'cvRef': 'MS',
            'accession': 'MS:1000576',
            'name': 'no compression',
            'value': ''
        })
        int_binary = ET.SubElement(int_array, 'binary')
        int_binary.text = self._encode_array(intensities)
        
        # Add user params for additional metadata
        if spec.get('identifier'):
            ET.SubElement(spectrum, 'userParam', {
                'name': 'identifier',
                'value': str(spec['identifier'])
            })
        if spec.get('smiles'):
            ET.SubElement(spectrum, 'userParam', {
                'name': 'smiles',
                'value': str(spec['smiles'])
            })
        if spec.get('inchikey'):
            ET.SubElement(spectrum, 'userParam', {
                'name': 'inchikey',
                'value': str(spec['inchikey'])
            })
        if spec.get('formula'):
            ET.SubElement(spectrum, 'userParam', {
                'name': 'formula',
                'value': str(spec['formula'])
            })
        if spec.get('adduct'):
            ET.SubElement(spectrum, 'userParam', {
                'name': 'adduct',
                'value': str(spec['adduct'])
            })
    
    def _encode_array(self, arr: np.ndarray) -> str:
        """Encode numpy array to base64 string."""
        import base64
        return base64.b64encode(arr.astype(np.float64).tobytes()).decode('ascii')


def convert_parquet_to_mzml(input_file: Path, output_file: Path, max_spectra: int = None):
    """
    Convert a single parquet file to mzML format.
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output mzML file
        max_spectra: Maximum number of spectra to convert (None for all)
    """
    logger.info(f"Reading {input_file}...")
    df = pd.read_parquet(input_file)
    
    if max_spectra:
        df = df.head(max_spectra)
    
    logger.info(f"Converting {len(df)} spectra...")
    
    writer = MzMLWriter(str(output_file))
    
    for idx, row in df.iterrows():
        spec_data = {
            'identifier': row.get('identifier'),
            'mzs': writer.parse_array_string(row['mzs']),
            'intensities': writer.parse_array_string(row['intensities']),
            'smiles': row.get('smiles'),
            'inchikey': row.get('inchikey'),
            'formula': row.get('formula'),
            'precursor_mz': row.get('precursor_mz'),
            'adduct': row.get('adduct'),
            'instrument_type': row.get('instrument_type'),
            'collision_energy': row.get('collision_energy'),
        }
        writer.add_spectrum(spec_data)
    
    writer.write()
    logger.info(f"Successfully converted {input_file.name} to {output_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert MassSpecGym parquet files to mzML format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Input parquet file or directory containing parquet files'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory for mzML files'
    )
    parser.add_argument(
        '--max-spectra',
        type=int,
        default=None,
        help='Maximum number of spectra to convert per file (default: all)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.parquet',
        help='File pattern to match (default: *.parquet)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of parquet files
    if input_path.is_file():
        parquet_files = [input_path]
    elif input_path.is_dir():
        parquet_files = sorted(input_path.glob(args.pattern))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    if not parquet_files:
        logger.error(f"No parquet files found matching pattern '{args.pattern}' in {input_path}")
        return
    
    logger.info(f"Found {len(parquet_files)} parquet file(s) to convert")
    
    # Convert each file
    for parquet_file in parquet_files:
        output_file = output_dir / f"{parquet_file.stem}.mzML"
        try:
            convert_parquet_to_mzml(parquet_file, output_file, args.max_spectra)
        except Exception as e:
            logger.error(f"Error converting {parquet_file.name}: {e}", exc_info=True)
            continue
    
    logger.info("Conversion complete!")


if __name__ == '__main__':
    main()
