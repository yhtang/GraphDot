import struct


def flatten(abstract_layout, depth=0):
    """
    Convert nested structures into native C++ layout with proper padding
    """

    padded_layout = ''
    padded_size = 0
    alignment = 1

    i = 0
    while i < len(abstract_layout):
        c = abstract_layout[i]
        if c == '[':
            step, sublayout = flatten(abstract_layout[i+1:], depth + 1)
            alignment = max(alignment, sublayout['align'])
            padding = (sublayout['align'] - padded_size) % sublayout['align']
            padded_layout += 'x' * padding + sublayout['layout']
            padded_size += padding + sublayout['size']
            i += step + 1
        elif c == ']':
            return i + 1, {'layout': padded_layout,
                           'size': padded_size,
                           'align': alignment}
        else:
            c_size = struct.calcsize(c)
            alignment = max(alignment, c_size)
            padding = (c_size - padded_size) % c_size
            padded_layout += 'x' * padding + c
            padded_size += padding + c_size
            i += 1

    padding = (alignment - padded_size) % alignment
    padded_layout += 'x' * padding
    padded_size += padding

    return padded_layout, alignment
