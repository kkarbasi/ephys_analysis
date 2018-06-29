"""
Copyright (c) 2016 David Herzfeld

Written by David J. Herzfeld <herzfeldd@gmail.com>
"""

import struct 
import os
import sys
import numpy as np

def load(filename):
    """Load an FHD binary file"""
    if not os.path.isfile(filename):
        raise RuntimeError('File {:s} does not exist'.format(filename))
    # Open the file
    fp = open(filename, 'rb')
    
    header = _read_header(fp)
    root = _read_group(fp)
    fp.close()

    return root

def _read_header(fp):
    read_bytes = fp.read(1024)
    magic_header = struct.unpack('@3s', read_bytes[0:3])[0].decode('utf-8')
    if magic_header != 'fhd':
        raise RuntimeError('FHD header is invalid')
    header = {};
    if (sys.version_info > (3, 0)):
        header['major_version'] = struct.unpack('@B', bytes([read_bytes[4]]))[0]
        header['minor_version'] = struct.unpack('@B', bytes([read_bytes[5]]))[0]
        header['minor_minor_version'] = struct.unpack('@B', bytes([read_bytes[6]]))[0]
        header['pointer_size'] = struct.unpack('@B', bytes([read_bytes[7]]))[0]
        header['num_pointer_entries'] = struct.unpack('@B', bytes([read_bytes[8]]))[0]
    return header

def _read_group(fp, data={}):
    current_location = fp.tell()

    group_location = struct.unpack('@Q', fp.read(8))[0]
    if group_location != current_location:
        raise RuntimeError('Group location is invalid')
    element_type = struct.unpack('@B', fp.read(1))[0]
    if element_type != 0:
        raise RuntimeError('Element type is invalid for group')
    name_length = struct.unpack('@H', fp.read(2))[0]
    name = struct.unpack('@{:d}s'.format(name_length), fp.read(name_length))[0].decode('utf-8')
    parent = struct.unpack('@Q', fp.read(8))[0]

    if name == '/':
        data = _read_linked_list(fp, {})
    else:
        data[name] = _read_linked_list(fp, {})
    return data

def _read_linked_list(fp, data={}):
    next_location = struct.unpack('@Q', fp.read(8))[0]
    previous_location = struct.unpack('@Q', fp.read(8))[0]
    max_entries = struct.unpack('@Q', fp.read(8))[0]
    num_contents = struct.unpack('@Q', fp.read(8))[0]
    if num_contents > max_entries:
        raise RuntimeError('Number of contents for linked list exceeds max entries')
    child_locations = struct.unpack('@{:d}Q'.format(num_contents), fp.read(8*num_contents))

    for i in range(0, num_contents):
        fp.seek(child_locations[i])
        child_location = struct.unpack('@Q', fp.read(8))[0]
        if child_location != child_locations[i]:
            raise RuntimeError('Child location is invalid')
        child_element_type = struct.unpack('@B', fp.read(1))[0]
        fp.seek(child_locations[i])
        if child_element_type == 0:
            data = _read_group(fp, data)
        elif child_element_type == 1:
            data = _read_attribute(fp, data)
        else:
            data = _read_dataset(fp, data) 
    # Load the next set of entries in our list
    if next_location > 0:
        fp.seek(next_location)
        data = _read_linked_list(fp, data)
    return data

def _data_type_to_struct_symbol(data_type):
    if data_type == 0:
        return ('d', 8, np.float64) # Double
    elif data_type == 1:
        return ('f', 4, np.float32) # Float
    elif data_type == 2:
        return ('B', 1, np.uint8) # uint8_t
    elif data_type == 3:
        return ('H', 2, np.uint16) # uint16_t
    elif data_type == 4:
        return ('I', 4, np.uint32) # uint32_t
    elif data_type == 5:
        return ('Q', 8, np.uint64) # uint64_t
    elif data_type == 6:
        return ('b', 1, np.int8) # int8_t
    elif data_type == 7:
        return ('h', 2, np.int16) # int16_t
    elif data_type == 8:
        return ('i', 4, np.int32) # int32_t
    elif data_type == 9:
        return ('q', 8, np.int64) # int64_t
    elif data_type == 10:
        return ('s', 1, np.uint8) # string (char)
    elif data_type == 11:
        return ('Q', 8, np.uint64) # pointer (uint64_t)

def _read_attribute(fp, data={}):
    current_location = fp.tell()
    location = struct.unpack('@Q', fp.read(8))[0]
    if location != current_location:
        raise RuntimeError('Attribute location is invalid')
    element_type = struct.unpack('@B', fp.read(1))[0]
    if element_type != 1:
        raise RuntimeError('Element type is invalid for attribute')
    name_length = struct.unpack('@H', fp.read(2))[0]
    name = struct.unpack('@{:d}s'.format(name_length), fp.read(name_length))[0].decode('utf-8')
    parent = struct.unpack('@Q', fp.read(8))[0]
    num_dimensions = int(struct.unpack('@B', fp.read(1))[0])
    dimensions = struct.unpack('@{:d}Q'.format(num_dimensions), fp.read(8*num_dimensions))
    data_type = struct.unpack('@B', fp.read(1))[0]
    total_elements = 1
    for i in range(0, num_dimensions):
        total_elements *= dimensions[i]
    struct_symbol, size, _ = _data_type_to_struct_symbol(data_type)
    data[name] = struct.unpack('@{:d}{:s}'.format(total_elements, struct_symbol), fp.read(size * total_elements))
    if data_type == 10: # string
        data[name] = data[name][0].decode('utf-8')
    else:
        data[name] = np.array(data[name])
        data[name] = np.reshape(data[name], dimensions)
    
    return data

def _read_dataset(fp, data={}):
    current_location = fp.tell()
    location = struct.unpack('@Q', fp.read(8))[0]
    if location != current_location:
        raise RuntimeError('Dataset location is invalid')
    element_type = struct.unpack('@B', fp.read(1))[0]
    if element_type != 2:
        raise RuntimeError('Element type is invalid for dataset')
    name_length = struct.unpack('@H', fp.read(2))[0]
    name = struct.unpack('@{:d}s'.format(name_length), fp.read(name_length))[0].decode('utf-8')
    parent = struct.unpack('@Q', fp.read(8))[0]
    num_dimensions = int(struct.unpack('@B', fp.read(1))[0])
    dimensions = struct.unpack('@{:d}Q'.format(num_dimensions), fp.read(8*num_dimensions))
    data_type = struct.unpack('@B', fp.read(1))[0]
    long_dimension = struct.unpack('@Q', fp.read(8))[0]
    data[name] = _read_dataset_linked_list(fp, dimensions, long_dimension, data_type)
    return data

def _read_dataset_linked_list(fp, dimensions, long_dimension, data_type):
    total_dimensions = list(dimensions)
    total_dimensions.append(long_dimension)
    struct_symbol, size, dtype = _data_type_to_struct_symbol(data_type)
    if data_type != 11:  # This is a pointer
        data = np.zeros(total_dimensions, dtype=dtype)
    else:
        data = [None for i in range(np.prod(total_dimensions))]
    index = 0
    while True:
        next_location = struct.unpack('@Q', fp.read(8))[0]
        previous_location = struct.unpack('@Q', fp.read(8))[0]
        max_entries = struct.unpack('@Q', fp.read(8))[0]
        num_contents = struct.unpack('@Q', fp.read(8))[0]
        if num_contents > max_entries:
            raise RuntimeError('Number of contents for linked list exeeds max entries')
        child_locations = struct.unpack('@{:d}Q'.format(num_contents), fp.read(8*num_contents))

        for i in range(0, num_contents):
            fp.seek(child_locations[i])
            long_axis = struct.unpack('@Q', fp.read(8))[0]
            total_data = long_axis
            for j in range(0, len(dimensions)):
                total_data *= dimensions[j]
            temp = np.array(struct.unpack('@{:d}{:s}'.format(total_data, struct_symbol), fp.read(total_data * size)))
            if data_type == 11:  # This is a pointer
                for j in range(0, len(temp)):
                    fp.seek(temp[j])
                    sub_group = _read_group(fp, data={})
                    data[index+j] = sub_group[list(sub_group)[0]]
            else:
                new_dimensions = list(dimensions)
                new_dimensions.append(long_axis)
                temp = np.reshape(temp, new_dimensions)
                data[..., index:index+long_axis] = temp
            index = index + long_axis
        if next_location == 0:
            break
        else:
            fp.seek(next_location)
    return data


if __name__ == '__main__':
    load(sys.argv[1])
     
